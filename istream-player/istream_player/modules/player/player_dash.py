import asyncio
import logging

from istream_player.config.config import PlayerConfig
from istream_player.core.buffer import BufferManager
from istream_player.core.module import Module, ModuleOption
from istream_player.core.player import Player, PlayerEventListener
from istream_player.core.scheduler import Scheduler
from istream_player.models import State


@ModuleOption("dash_player", default=True, requires=[BufferManager, Scheduler])
class DASHPlayer(Module, Player):
    log = logging.getLogger("DASHPlayer")

    def __init__(self):
        super().__init__()

        # State related
        self._state = State.IDLE

        # Playback related
        self._playback_started = False
        self._position = 0.0

    async def setup(self, config: PlayerConfig, buffer_manager: BufferManager, scheduler: Scheduler):
        self.min_start_buffer_duration = config.min_start_duration
        self.min_rebuffer_duration = config.min_rebuffer_duration
        self.time_factor = config.time_factor

        self.buffer_manager = buffer_manager
        self.scheduler = scheduler

    @property
    def state(self) -> State:
        return self._state

    async def _switch_state(self, old_state: State, new_state: State):
        for listener in self.listeners:
            await listener.on_state_change(self._position, old_state, new_state)

    def stop(self) -> None:
        raise NotImplementedError

    def pause(self) -> None:
        raise NotImplementedError

    def add_listener(self, listener: PlayerEventListener):
        if listener not in self.listeners:
            self.listeners.append(listener)

    # @critical_task()
    async def run(self):
        """
        The main loop.
        This method coordinate work between different components.
        """
        # Start the scheduler
        self._state = State.BUFFERING
        first_start_time = None
        await self._switch_state(self._state, State.BUFFERING)

        while self._state != State.END:
            # Wait for minimum segments to be available
            if self._state == State.BUFFERING:
                min_buffer_duration = self.min_rebuffer_duration if self._playback_started else self.min_start_buffer_duration
                if self.buffer_manager.buffer_level < min_buffer_duration:
                    self.log.debug(f"Waiting for {min_buffer_duration=}, {self.buffer_manager.buffer_level=}")
                    async with self.buffer_manager.buffer_change_cond:
                        await self.buffer_manager.buffer_change_cond.wait_for(
                            lambda: self.buffer_manager.buffer_level >= min_buffer_duration or self.buffer_manager.is_closed()
                        )
                if self.buffer_manager.buffer_level > 0:
                    await self._switch_state(self._state, State.READY)
                    self._state = State.READY
                else:
                    await self._switch_state(self._state, State.READY)
                    self._state = State.END
                    continue

            # Play segment
            segments, duration = self.buffer_manager.get_next_segment()

            if first_start_time is None:
                first_start_time = min(map(lambda s: s.start_time, segments.values()))

            self.log.debug(f"Playing segment of duration : {duration}")

            self._position = min(map(lambda s: s.start_time, segments.values())) - first_start_time
            for listener in self.listeners:
                await listener.on_position_change(self._position)
            await self._switch_state(self._state, State.READY)
            for listener in self.listeners:
                await listener.on_segment_playback_start(segments)
            await asyncio.sleep(self.time_factor * duration)
            self._position += duration
            for listener in self.listeners:
                await listener.on_position_change(self._position)
            await self.buffer_manager.dequeue_buffer()

            # Update for next round
            if self.buffer_manager.is_empty():
                self.log.debug("Buffer manager is empty")
                if self.buffer_manager.is_closed():
                    self.log.debug("Scheduler has ended")
                    await self._switch_state(self._state, State.END)
                    self._state = State.END
                else:
                    await self._switch_state(self._state, State.BUFFERING)
                    self._state = State.BUFFERING
            else:
                await self._switch_state(self._state, State.READY)
