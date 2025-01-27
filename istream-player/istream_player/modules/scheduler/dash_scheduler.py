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


@ModuleOption(
    "scheduler", default=True, requires=["segment_downloader", BandwidthMeter, BufferManager, MPDProvider, ABRController]
)
class SchedulerImpl(Module, Scheduler):
    log = logging.getLogger("SchedulerImpl")

    def __init__(self, *, select_as: str = "-"):
        super().__init__()

        self.adaptation_sets: Optional[Dict[int, AdaptationSet]] = None
        self.started = False

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

            # Check buffer level
            if self.buffer_manager.buffer_level > req_buffer_level:
                async with self.buffer_manager.buffer_change_cond:
                    await self.buffer_manager.buffer_change_cond.wait_for(
                        lambda: self.buffer_manager.buffer_level <= req_buffer_level
                    )

            assert self.mpd_provider.mpd is not None
            if self.mpd_provider.mpd.type == "dynamic":
                await self.mpd_provider.update()
                self.adaptation_sets = self.select_adaptation_sets(self.mpd_provider.mpd.adaptation_sets)

            first_segment, last_segment = self.segment_limits(self.adaptation_sets)
            self.log.info(f"{first_segment=}, {last_segment=}")

            if self._index < first_segment:
                self.log.info(f"Segment {self._index} not in mpd, Moving to next segment")
                self._index += 1
                continue

            if self.mpd_provider.mpd.type == "dynamic" and self._index > last_segment:
                self.log.info(f"Waiting for more segments in mpd : {self.mpd_provider.mpd.type}")
                await asyncio.sleep(self.time_factor * self.update_interval)
                continue

            # Download one segment from each adaptation set
            if self._index == self._dropped_index:
                selections = self.abr_controller.update_selection_lowest(self.adaptation_sets)
            else:
                selections = self.abr_controller.update_selection(self.adaptation_sets, self._index)
            self.log.info(f"Downloading index {self._index} at {selections}")
            self._current_selections = selections

            # All adaptation sets take the current bandwidth
            adap_bw = {as_id: self.bandwidth_meter.bandwidth for as_id in selections.keys()}

            # Get segments to download for each adaptation set
            try:
                segments = {
                    adaptation_set_id: self.adaptation_sets[adaptation_set_id].representations[selection].segments[self._index]
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
                adaptation_set = self.adaptation_sets[adaptation_set_id]
                representation = adaptation_set.representations[selection]
                init_url = representation.initialization
                if init_url is not None and init_url not in self._representation_initialized:
                    await self.download_manager.download(DownloadRequest(init_url, DownloadType.STREAM_INIT))
                    await self.download_manager.wait_complete(init_url)
                    self.log.info(f"Segment Initialization downloaded: {init_url}")
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
            # self.log.info(f"Completed downloading from urls {urls}")
            if any([result is None for result in results]):
                # Result is None means the stream got dropped
                self._dropped_index = self._index
                continue
            for listener in self.listeners:
                await listener.on_segment_download_complete(self._index, segments)
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
