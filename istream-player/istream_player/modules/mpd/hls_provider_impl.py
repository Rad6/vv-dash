import logging
import os
import time
from typing import Dict, Optional

import m3u8

from istream_player.config.config import PlayerConfig
from istream_player.core.downloader import (
    DownloadManager,
    DownloadRequest,
    DownloadType,
)
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.models.mpd_objects import MPD, AdaptationSet, Representation, Segment
from istream_player.utils.async_utils import AsyncResource


class HLSParser:
    log = logging.getLogger("HLSParser")

    def __init__(self, downloader: DownloadManager) -> None:
        self.downloader = downloader

    async def fetch(self, url: str):
        await self.downloader.download(DownloadRequest(url, DownloadType.MANIFEST))
        content, size = await self.downloader.wait_complete(url)
        text = content.decode("utf-8")
        playlist = m3u8.loads(text)

        adap_set = AdaptationSet(0, "video", None, 0, 0, "", {}, {})
        mpd = MPD(text, url, "static", 0, 0, 0, {0: adap_set}, {})

        base_url = os.path.dirname(url)

        if len(playlist.playlists) > 0:
            repr_id = 0
            repr_by_quality = sorted(playlist.playlists, key=lambda pl: pl.stream_info.bandwidth)
            for sub_playlist in playlist.playlists:
                repr_url = os.path.join(base_url, sub_playlist.uri)
                await self.downloader.download(DownloadRequest(repr_url, DownloadType.MANIFEST))
                content, size = await self.downloader.wait_complete(repr_url)
                repr_pl = m3u8.loads(content.decode("utf-8"))
                str_info = sub_playlist.stream_info
                if str_info.frame_rate is None or str_info.resolution is None:
                    self.log.info(
                        f"Frame rate or Resolution is missing from the m3u8 playlist '{sub_playlist.uri}'. Skipping playlist"
                    )
                    continue
                repr = self.parse_representation(
                    repr_pl,
                    base_url,
                    0,
                    repr_id,
                    repr_by_quality.index(sub_playlist),
                    str_info.bandwidth,
                    str_info.frame_rate,
                    str_info.codecs,
                    str_info.resolution[0],
                    str_info.resolution[1],
                )
                adap_set.representations[repr_id] = repr
                repr_id += 1
        elif len(playlist.segments) > 0:
            repr = self.parse_representation(playlist, base_url, 0, 0, 0, 0, 0, "", 0, 0)
            adap_set.representations[0] = repr
        else:
            raise Exception("m3u8 playlist contains neither segments nor sub playlists")

        return mpd

    def parse_representation(
        self,
        playlist: m3u8.M3U8,
        base_url: str,
        as_id: int,
        repr_id: int,
        quality: int,
        bitrate: int,
        framerate: float,
        codecs: str,
        width: int,
        height: int,
    ):
        # if len(playlist.segment_map) != 1:
        #     raise Exception(f"Exactly 1 EXT-X-MAP expected : {len(playlist.segment_map)} found")

        seg_map = playlist.segment_map[0] if len(playlist.segment_map) > 0 else None
        # assert seg_map is not None
        if seg_map is not None and seg_map.byterange is not None:
            raise Exception("EXT-X-MAP byterange not supported")
        init_url = os.path.join(base_url, seg_map.uri) if seg_map is not None else None
        segments: Dict[int, Segment] = {}
        start_time = 0
        seg_index = 0
        for segment in playlist.segments:
            # if int(framerate * segment.duration) != (framerate * segment.duration):
            #     raise Exception(f"Fractional framerate not supiported : {framerate * segment.duration}")
            segments[seg_index] = Segment(
                seg_index,
                url=os.path.join(base_url, segment.uri),
                init_url=init_url,
                duration=segment.duration,
                start_time=start_time,
                as_id=as_id,
                repr_id=repr_id,
                quality=quality,
                bitrate=bitrate,
                num_frames=int(framerate * segment.duration),
            )
            seg_index += 1
            start_time += segment.duration
        return Representation(
            id_=repr_id,
            mime_type="video/mp4",
            codecs=codecs,
            bandwidth=bitrate,
            width=width,
            height=height,
            initialization=init_url,
            segments=segments,
            attrib={},
        )


@ModuleOption("hls", default=True, requires=["mpd_downloader"])
class HLSProviderImpl(Module, MPDProvider):
    log = logging.getLogger("HLSProviderImpl")

    def __init__(self):
        super().__init__()
        self.last_updated = 0
        self._mpd_res: AsyncResource[Optional[MPD]] = AsyncResource(None)

    async def setup(self, config: PlayerConfig, mpd_downloader: DownloadManager, **kwargs):
        self.update_interval = config.static.update_interval
        self.download_manager = mpd_downloader
        self.mpd_url = config.input
        self.parser = HLSParser(mpd_downloader)

    @property
    def mpd(self) -> Optional[MPD]:
        return self._mpd_res.get()

    async def available(self) -> MPD:
        value = await self._mpd_res.value_non_none()
        assert value is not None
        return value

    async def update(self):
        if self.mpd is not None and (time.time() - self.last_updated) < self.update_interval:
            return
        mpd = await self.parser.fetch(self.mpd_url)
        await self._mpd_res.set(mpd)

        self.last_updated = time.time()

    async def run(self):
        assert self.mpd_url is not None
        await self.update()
        assert self.mpd is not None

    async def stop(self):
        self.log.info("Stopping HLS Provider")
        await self.download_manager.close()
