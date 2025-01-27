import asyncio
import itertools
import logging
from asyncio import Task
from typing import Dict, Optional, Set

from istream_player.config.config import PlayerConfig
from istream_player.core.abr import ABRController
from istream_player.core.buffer import BufferManager
from istream_player.core.bw_meter import BandwidthMeter
from istream_player.core.downloader import DownloadManager, DownloadRequest, DownloadType
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.core.scheduler import Scheduler, SchedulerEventListener
from istream_player.models import AdaptationSet
import pdb
import time

@ModuleOption(
    "scheduler_360", default=True, requires=["segment_downloader", BandwidthMeter, BufferManager, MPDProvider, ABRController]
)
class Scheduler360(Module, Scheduler):
    log = logging.getLogger("Scheduler360")

    def __init__(self, *, select_as: str = "-", mode_scheduler: str = "all_tiles"):
        super().__init__()

        self.adaptation_sets: Optional[Dict[int, AdaptationSet]] = None
        self.started = False
        self.mode_scheduler = mode_scheduler

        self._task: Optional[Task] = None
        self._index = 0
        self._representation_initialized: Set[str] = set()
        self._current_selections: Optional[Dict[int, int]] = None

        self._dropped_index = None
        _select_as = select_as.split("-")
        if len(_select_as) == 1 and _select_as[0].isdecimal():
            self.selected_as_start = int(_select_as[0])
            self.selected_as_end = int(_select_as[0])
        elif (
            len(_select_as) == 2
            and (_select_as[0].isdecimal() or _select_as[0] == "")
            and (_select_as[1].isdecimal() or _select_as[1] == "")
        ):
            self.selected_as_start = int(_select_as[0]) if _select_as[0] != "" else None
            self.selected_as_end = int(_select_as[1]) if _select_as[1] != "" else None
        else:
            raise Exception("select_as should be of the format '<uint>-<uint>' or '<uint>'.")

    async def setup(
        self,
        config: PlayerConfig,
        segment_downloader: DownloadManager,
        bandwidth_meter: BandwidthMeter,
        buffer_manager: BufferManager,
        mpd_provider: MPDProvider,
        abr_controller: ABRController,
    ):
        self.max_buffer_duration = config.buffer_duration
        self.update_interval = config.static.update_interval
        self.time_factor = config.time_factor

        self.download_manager = segment_downloader
        self.bandwidth_meter = bandwidth_meter
        self.buffer_manager = buffer_manager
        self.abr_controller = abr_controller
        self.mpd_provider = mpd_provider
        self.fov_groups_number = config.fov_groups_number

    def segment_limits(self, adap_sets: Dict[int, AdaptationSet]) -> tuple[int, int]:
        ids = [
            [[seg_id for seg_id in repr.segments.keys()] for repr in as_val.representations.values()]
            for as_val in adap_sets.values()
        ]
        ids = itertools.chain(*ids)
        ids = list(itertools.chain(*ids))
        return min(ids), max(ids)

    async def run(self):
        self.log.debug("Waiting for MPD file")
        await self.mpd_provider.available()
        self.log.debug("MPD file available")
        assert self.mpd_provider.mpd is not None
        self.adaptation_sets = self.select_adaptation_sets(self.mpd_provider.mpd.adaptation_sets)
        # print(f"{self.adaptation_sets=}")

        # Start from the min segment index
        first_segment, last_segment = self.segment_limits(self.adaptation_sets)
        self._index = first_segment
        while self.mpd_provider.mpd.type == "dynamic" or self._index <= last_segment:

            # Calculate required buffer level
            exp_seg_dur = max(
                rep.segments[self._index].duration
                for adap_set in self.adaptation_sets.values()
                for rep in adap_set.representations.values()
                if self._index in rep.segments
            )
            req_buffer_level = self.max_buffer_duration - exp_seg_dur
            #print("req_buffer_level", req_buffer_level)
            #print("buffer_level", self.buffer_manager.buffer_level)

            # Check buffer level
            if self.buffer_manager.buffer_level > req_buffer_level:
                async with self.buffer_manager.buffer_change_cond:
                    await self.buffer_manager.buffer_change_cond.wait_for(
                        lambda: self.buffer_manager.buffer_level <= req_buffer_level
                    )


            mod_scheduler = self.mode_scheduler
            if mod_scheduler == "fov_tiles":
                all_tiles = {}
                previous_selection = {}
                fov_groups = self.abr_controller.fov_group_selection(self.adaptation_sets)
                # Run the ABR controller for each FoV group
                time_start = time.time()
                for fov_group_id in range(1, self.fov_groups_number +1):
                    self.log.info(f"--- Running for FoV group {fov_group_id} and index {self._index} ---")
                    self.log.info(f"FoV Groups: {fov_groups}")
                    selections = self.abr_controller.update_selection(self.adaptation_sets, self._index, fov_groups, fov_group_id, mod_scheduler)
                    #self.log.debug(f"Downloading index {self._index} at {selections}")
                    adap_bw = {as_id[0]: self.bandwidth_meter.bandwidth for as_id in selections.keys()}
                
                    # Get segments to download for each adaptation set (only for the current FoV group)
                    selections_download = {key: value for key, value in selections.items() if key[1] == fov_group_id}
                    try:
                        segments = {
                            adaptation_set_id[0]: self.adaptation_sets[adaptation_set_id[0]].representations[selection].segments[self._index]
                            for adaptation_set_id, selection in selections_download.items()
                        }
                    except KeyError:
                        # No more segments left
                        self.log.info("No more segments left")
                        self._end = True
                        await self.buffer_manager.close()
                        return
                    

                    for listener in self.listeners:
                        #self.log.debug(f"Downloading index {self._index} at {selections}")
                        await listener.on_segment_download_start(self._index, adap_bw, segments)

                    # duration = 0
                    time_start_dl = time.time()
                    urls = []
                    bitrates = []
                    

                    for adaptation_set_id, selection in selections_download.items():

                        adaptation_set = self.adaptation_sets[adaptation_set_id[0]]
                        representation = adaptation_set.representations[selection]
                        init_url = representation.initialization
                        if init_url is not None and init_url not in self._representation_initialized:
                            await self.download_manager.download(DownloadRequest(init_url, DownloadType.STREAM_INIT))
                            await self.download_manager.wait_complete(init_url)
                            #self.log.info(f"Segment Initialization downloaded: {init_url}")
                            self._representation_initialized.add(init_url)
                        try:
                            segment = representation.segments[self._index]
                        except IndexError:
                            self.log.info("Segments ended")
                            self._end = True
                            await self.buffer_manager.close()
                            return
                        urls.append(segment.url)
                        bitrates.append(segment.bitrate)
                        
                        self.log.info(f"Downloading segment {segment.url}")
                        await self.download_manager.download(DownloadRequest(segment.url, DownloadType.SEGMENT))
                        
                        # duration = segment.duration

                    # Wait for all tile downloads to finish in FOV group
                    results = [await self.download_manager.wait_complete(url) for url in urls]
                    # self.log.info(f"Completed downloading from urls {urls}")
                    time_end_dl = time.time()

                    self.log.info(f"Time taken for complete downloading tile {fov_groups[fov_group_id]}: {time_end_dl - time_start_dl}")

                    # pdb.set_trace()

                    for listener in self.listeners:
                        await listener.on_segment_tile_download_complete(self._index, segments, fov_groups)
                        #self.log.info(f"Segment download complete for index {self._index} and fov group {fov_group_id}")

                    # Add the new tiles in FOV group in all_tiles
                    all_tiles.update(segments)
                    #self.log.info(f"all_tiles: {all_tiles.keys()}")
                time2 = time.time()
                self.log.info(f" 🕐 Time taken for finishing index {self._index}: {time2 - time_start}  🕐")
                self.log.info(f"Enqueueing buffer for index {self._index} and tiles {all_tiles.keys()}")
                await self.buffer_manager.enqueue_buffer(all_tiles)
                self._index += 1

            # await self.buffer_manager.close()
            elif mod_scheduler == "all_tiles":
          
                # assert self.mpd_provider.mpd is not None
                # if self.mpd_provider.mpd.type == "dynamic":
                #     await self.mpd_provider.update()
                #     self.adaptation_sets = self.select_adaptation_sets(self.mpd_provider.mpd.adaptation_sets)

                # first_segment, last_segment = self.segment_limits(self.adaptation_sets)
                # self.log.info(f"{first_segment=}, {last_segment=}")

                # if self._index < first_segment:
                #     self.log.info(f"Segment {self._index} not in mpd, Moving to next segment")
                #     self._index += 1
                #     continue

                # if self.mpd_provider.mpd.type == "dynamic" and self._index > last_segment:
                #     self.log.info(f"Waiting for more segments in mpd : {self.mpd_provider.mpd.type}")
                #     await asyncio.sleep(self.time_factor * self.update_interval)
                #     continue

                #Download one segment from each adaptation set
                start_time_all_tiles = time.time()
                if self._index == self._dropped_index:
                    selections = self.abr_controller.update_selection_lowest(self.adaptation_sets)
                else:
                    self.log.info(f"------ Updating selections for index {self._index} ----------")
                    fov_groups = self.abr_controller.fov_group_selection(self.adaptation_sets)
                    self.log.info(f"FoV Groups: {fov_groups}")
                    selections = self.abr_controller.update_selection(self.adaptation_sets, self._index, fov_groups, 1 , mod_scheduler)
                    #pdb.set_trace()
                    #self.log.info(f"Selections: {selections}")
                self.log.info(f"Downloading index {self._index} at {selections}")
                self._current_selections = selections

                # All adaptation sets take the current bandwidth
                adap_bw = {as_id[0]: self.bandwidth_meter.bandwidth for as_id in selections.keys()}
                

                # Get segments to download for each adaptation set
                try:
                    segments = {
                        adaptation_set_id[0]: self.adaptation_sets[adaptation_set_id[0]].representations[selection].segments[self._index]
                        for adaptation_set_id, selection in selections.items()
                    }
                except KeyError:
                    # No more segments left
                    self.log.info("No more segments left")
                    self._end = True
                    await self.buffer_manager.close()
                    return
                
                for listener in self.listeners:
                    await listener.on_segment_download_start(self._index, adap_bw, segments)

                # duration = 0
                urls = []
                for adaptation_set_id, selection in selections.items():
                    adaptation_set = self.adaptation_sets[adaptation_set_id[0]]
                    representation = adaptation_set.representations[selection]
                    init_url = representation.initialization
                    if init_url is not None and init_url not in self._representation_initialized:
                        await self.download_manager.download(DownloadRequest(init_url, DownloadType.STREAM_INIT))
                        await self.download_manager.wait_complete(init_url)
                        #self.log.info(f"Segment Initialization downloaded: {init_url}")
                        self._representation_initialized.add(init_url)
                    try:
                        segment = representation.segments[self._index]
                    except IndexError:
                        self.log.info("Segments ended")
                        self._end = True
                        await self.buffer_manager.close()
                        return
                    urls.append(segment.url)
                    await self.download_manager.download(DownloadRequest(segment.url, DownloadType.SEGMENT))
                    # duration = segment.duration
                # self.log.info(f"Waiting for completion urls {urls}")
                results = [await self.download_manager.wait_complete(url) for url in urls]
                #self.log.info(f"Completed downloading from urls {urls}")

                if any([result is None for result in results]):
                    # Result is None means the stream got dropped
                    self._dropped_index = self._index
                    continue
                for listener in self.listeners:
                    await listener.on_segment_tile_download_complete(self._index, segments, fov_groups)
                time_end_all_tiles = time.time()
                self.log.info(f"🕐 Time taken for downloading all tiles segment {self._index}: {time_end_all_tiles - start_time_all_tiles}")
                self._index += 1
                await self.buffer_manager.enqueue_buffer(segments)

        await self.buffer_manager.close()

    def select_adaptation_sets(self, adaptation_sets: Dict[int, AdaptationSet]):
        as_ids = adaptation_sets.keys()
        start = self.selected_as_start if self.selected_as_start is not None else min(as_ids)
        end = self.selected_as_end if self.selected_as_end is not None else max(as_ids)
        return {as_id: as_val for as_id, as_val in adaptation_sets.items() if as_id >= start and as_id <= end}

    async def stop(self):
        await self.download_manager.close()
        if self._task is not None:
            self._task.cancel()

    def add_listener(self, listener: SchedulerEventListener):
        if listener not in self.listeners:
            self.listeners.append(listener)

    async def cancel_task(self, index: int):
        """
        Cancel current downloading task, and move to the next one

        Parameters
        ----------
        index: int
            The index of segment to cancel
        """

        raise Exception("Not supported")

        # If the index is the the index of currently downloading segment, ignore it
        # if self._index != index or self._current_selections is None:
        #     return

        # # Do not cancel the task for the first index
        # if index == 0:
        #     return

        # assert self.adaptation_sets is not None
        # for adaptation_set_id, selection in self._current_selections.items():
        #     segment = self.adaptation_sets[adaptation_set_id].representations[selection].segments[self._index]
        #     self.log.debug(f"Stop current downloading URL: {segment.url}")
        #     await self.download_manager.stop(segment.url)

    async def drop_index(self, index):
        raise Exception("Not supported")
        # self._dropped_index = index
