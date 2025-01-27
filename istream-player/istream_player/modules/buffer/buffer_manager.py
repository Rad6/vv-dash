import asyncio
import logging
from typing import Dict, Tuple

from istream_player.config.config import PlayerConfig
from istream_player.core.buffer import BufferManager
from istream_player.core.module import Module, ModuleOption
from istream_player.models.mpd_objects import Segment
from istream_player.utils.async_utils import AsyncResource

# Item1 : Map of adaptation_id and segments, Item2 : Max duration of selected segments
QueueItemType = Tuple[Dict[int, Segment], float]


@ModuleOption("buffer_manager", default=True)
class BufferManagerImpl(Module, BufferManager):
    log = logging.getLogger("BufferManager")

    def __init__(self) -> None:
        super().__init__()
        self._buffer_level: float = 0
        self._segments: asyncio.Queue[QueueItemType] = asyncio.Queue()
        self._buffer_change_cond: asyncio.Condition = asyncio.Condition()
        self._closed: AsyncResource[bool] = AsyncResource(False)

    async def publish_buffer_level(self):
        for listener in self.listeners:
            await listener.on_buffer_level_change(self.buffer_level)

    async def setup(self, config: PlayerConfig):
        pass

    async def run(self) -> None:
        await self.publish_buffer_level()

    async def enqueue_buffer(self, segments: Dict[int, Segment]) -> None:
        max_duration = max(map(lambda s: s[1].duration, segments.items()))
        await self._segments.put((segments, max_duration))
        self._buffer_level += max_duration
        await self.publish_buffer_level()
        async with self._buffer_change_cond:
            self._buffer_change_cond.notify_all()
        self.log.debug(f"Buffer level increased to {self.buffer_level}")

    #segments[0].start_time = 0
    #segments[0].finish_time = 10

    def get_next_segment(self) -> QueueItemType:
        return self._segments._queue[0]  # type: ignore

    async def dequeue_buffer(self):
        segments, max_duration = await self._segments.get()
        self._buffer_level -= max_duration
        await self.publish_buffer_level()
        async with self._buffer_change_cond:
            self._buffer_change_cond.notify_all()
            # return segments, max_duration

    @property
    def buffer_level(self):
        return self._buffer_level

    @property
    def buffer_change_cond(self) -> asyncio.Condition:
        return self._buffer_change_cond

    def is_empty(self) -> bool:
        return self._segments.empty()

    async def close(self) -> None:
        await self._closed.set(True)
        async with self.buffer_change_cond:
            self.buffer_change_cond.notify_all()

    def is_closed(self) -> bool:
        return self._closed.get()
