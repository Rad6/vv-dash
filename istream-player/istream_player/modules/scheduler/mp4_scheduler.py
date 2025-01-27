import logging

from istream_player.config.config import PlayerConfig
from istream_player.core.buffer import BufferManager
from istream_player.core.downloader import DownloadManager, DownloadRequest, DownloadType
from istream_player.core.module import Module, ModuleOption
from istream_player.core.scheduler import Scheduler, SchedulerEventListener
from istream_player.models.mpd_objects import Segment


@ModuleOption("mp4_scheduler", default=True, requires=["segment_downloader", BufferManager])
class MP4SchedulerImpl(Module, Scheduler):
    log = logging.getLogger("MP4SchedulerImpl")

    def __init__(self):
        super().__init__()
        self._end = False

    async def setup(
        self,
        config: PlayerConfig,
        segment_downloader: DownloadManager,
        buffer_manager: BufferManager,
    ):
        self.download_manager = segment_downloader
        self.buffer_manager = buffer_manager
        self.init_bw = config.static.max_initial_bitrate
        self.url = config.input

        if not self.url.endswith(".mp4"):
            raise Exception("Please provide a '.mp4' file input")

    async def run(self):
        segments = {0: Segment(0, self.url, "", 3, 0, 0, 0, 0, 10000000, 0)}
        # Notify listeners
        for listener in self.listeners:
            await listener.on_segment_download_start(1, {0: self.init_bw}, segments)

        # Schedule before download starts to stream
        await self.buffer_manager.enqueue_buffer(segments)

        # Schedule download
        await self.download_manager.download(DownloadRequest(self.url, DownloadType.SEGMENT))

        # Wait for download to finish
        await self.download_manager.wait_complete(self.url)

        # Notify download complete
        for listener in self.listeners:
            await listener.on_segment_download_complete(1, segments)

        self.log.debug("Ending schdeduler")
        self._end = True
        await self.buffer_manager.close()

    def add_listener(self, listener: SchedulerEventListener):
        if listener not in self.listeners:
            self.listeners.append(listener)

    async def stop(self):
        await self.cancel_task(1)

    async def cancel_task(self, index: int):
        """
        Cancel current downloading task, and move to the next one

        Parameters
        ----------
        index: int
            The index of segment to cancel
        """

        await self.download_manager.stop(self.url)

    async def drop_index(self, index):
        self._dropped_index = index
