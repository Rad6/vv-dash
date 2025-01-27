import logging
import time
from typing import Dict
from collections import deque

from istream_player.config.config import PlayerConfig
from istream_player.core.bw_meter import BandwidthMeter, DownloadStats
from istream_player.core.downloader import DownloadEventListener, DownloadManager
from istream_player.core.module import Module, ModuleOption


@ModuleOption("bw_cont", requires=["segment_downloader"])
class BandwidthMeterCont(Module, BandwidthMeter, DownloadEventListener):
    log = logging.getLogger("BandwidthMeterCont")

    def __init__(self):
        super().__init__()

        # Stores recent packets within window (req_time, res_time, size_in_bytes)
        self.packets: deque[(float, int)] = deque()
        self.packets_total_bytes = 0
        self.packets_total_time = 0
        self.stats: Dict[str, DownloadStats] = {}

    async def setup(
        self, config: PlayerConfig, segment_downloader: DownloadManager, **kwargs
    ):
        self._bw = config.static.max_initial_bitrate
        self.smooth_factor = config.static.smoothing_factor
        # Discard packets with higher delay
        self.max_packet_delay = config.static.max_packet_delay
        # Minimum window size in seconds
        self.window_min = 0.5
        # Window size in seconds
        self.window = config.static.cont_bw_window

        segment_downloader.add_listener(self)

    async def on_transfer_start(self, url) -> None:
        self.log.debug(f"Transfer started : {url}")
        start_time = time.time()
        self.stats[url] = DownloadStats(start_time=start_time)

    async def on_bytes_transferred(
        self, length: int, url: str, position: int, size: int, content
    ) -> None:
        res_time = time.time()
        req_time = max(
            self.packets[-1][1] if len(self.packets) > 0 else 0,
            self.stats[url].start_time,
        )
        packet_delay = res_time - req_time
        if packet_delay > self.max_packet_delay:
            raise Exception(
                f"Packet received with delay={packet_delay} > max_packet_delay={self.max_packet_delay}"
            )

        self.packets.append((req_time, res_time, length))
        self.packets_total_bytes += length
        self.packets_total_time += packet_delay

        while (
            len(self.packets) > 2
            and (self.packets[-1][1] - self.packets[0][0]) > self.window
        ):
            req, res, psize = self.packets.popleft()
            self.packets_total_bytes -= psize
            self.packets_total_time -= res - req

        self._bw = 8 * self.packets_total_bytes / self.packets_total_time
        for listener in self.listeners:
            await listener.on_bandwidth_update(self._bw)

        # Update stats
        stats = self.stats.get(url)
        if stats.first_byte_at is None:
            stats.first_byte_at = res_time
        stats.last_byte_at = res_time
        stats.received_bytes += length
        stats.total_bytes = size

    async def on_transfer_end(self, size: int, url: str) -> None:
        t = time.time()
        stats = self.stats[url]
        if stats.stop_time is None:
            stats.stop_time = t
        if stats.stopped_bytes == 0:
            stats.stopped_bytes = stats.received_bytes

    async def on_transfer_canceled(self, url: str, position: int, size: int) -> None:
        return await self.on_transfer_end(position, url)

    @property
    def bandwidth(self) -> float:
        return self._bw

    def get_stats(self, url: str) -> DownloadStats:
        return self.stats[url]

    # def update_bandwidth(self):
    #     assert (
    #         self.transmission_end_time is not None
    #         and self.transmission_start_time is not None
    #     )
    #     self._bw = self._bw * self.smooth_factor + (8 * self.bytes_transferred) / (
    #         self.transmission_end_time - self.transmission_start_time
    #     ) * (1 - self.smooth_factor)
    #     # print(f"Bandwith updated : {self._bw}")
    #     # if self.last_cont_bw is not None:
    #     #     self._bw = (self._bw + self.last_cont_bw)/2
    #     # self._bw = self.last_cont_bw
    #     self.extra_stats = {
    #         "_bw": self._bw,
    #         "smooth_factor": self.smooth_factor,
    #         "bytes_transferred": self.bytes_transferred,
    #         "transmission_end_time": self.transmission_end_time,
    #         "transmission_start_time": self.transmission_start_time,
    #     }
    #     # self.log.info(f"************* Updated stats : {self.extra_stats}")
