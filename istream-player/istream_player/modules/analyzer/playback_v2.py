import asyncio
import logging
from typing import Dict

from istream_player.config.config import PlayerConfig
from istream_player.core.analyzer import Analyzer
from istream_player.core.buffer import BufferManager
from istream_player.core.downloader import DownloadEventListener, DownloadManager
from istream_player.core.module import Module, ModuleOption
from istream_player.core.player import Player, PlayerEventListener
from istream_player.models.mpd_objects import Segment
from istream_player.models.player_objects import State


@ModuleOption("playback_v2", requires=["segment_downloader", Player, BufferManager])
class PlaybackV2(Module, Analyzer, DownloadEventListener, PlayerEventListener):
    log = logging.getLogger("PlaybackV2")

    def __init__(self) -> None:
        super().__init__()
        from istream_renderer import IStreamRenderer
        self.renderer = IStreamRenderer()

    async def setup(
        self, config: PlayerConfig, segment_downloader: DownloadManager, player: Player, buffer_manager: BufferManager
    ):
        segment_downloader.add_listener(self)
        player.add_listener(self)
        buffer_manager._closed.on_value(True, lambda _: self.renderer.close())  # type: ignore
        self.renderer.setup(config)

    async def run(self) -> None:
        await self.renderer.run()
        self.log.debug("Playback finished")

    async def cleanup(self) -> None:
        self.renderer.end()

    async def on_bytes_transferred(self, length: int, url: str, position: int, size: int, content: bytes) -> None:
        self.renderer.decode(content, length)
        await asyncio.sleep(0)

    async def on_segment_playback_start(self, segments: Dict[int, Segment]):
        if len(segments) > 1:
            raise Exception("Multi-tile playback not supported")
        seg = list(segments.values())[0]
        self.renderer.play_frames(seg.num_frames)

    async def on_state_change(self, position: float, old_state: State, new_state: State):
        if new_state == State.END:
            self.log.debug("Ending playback")
            self.renderer.end()
