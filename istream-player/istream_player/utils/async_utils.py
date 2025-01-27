import asyncio
import functools
import inspect
import traceback
from asyncio.exceptions import CancelledError
from typing import Callable, Dict, Generic, TypeVar


def critical_task(ignore_exc: list[type[BaseException]] = [CancelledError]):
    def wrapper(func):
        @functools.wraps(func)
        async def wrapped(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:  # noqa
                if e.__class__ not in ignore_exc:
                    traceback.print_exc()
                    exit(1)

        return wrapped

    return wrapper


async def try_await(cor):
    if inspect.iscoroutine(cor):
        await cor


T = TypeVar("T")


class AsyncResource(Generic[T]):
    def __init__(self, default: T) -> None:
        self._value: T = default
        self.watchers: Dict[T, Callable] = {}
        self.event_non_none = asyncio.Event()
        if default is not None:
            self.event_non_none.set()

    def get(self) -> T:
        return self._value

    async def set(self, val: T):
        self._value = val
        if val is not None and not self.event_non_none.is_set():
            self.event_non_none.set()
        elif val is None and self.event_non_none.is_set():
            self.event_non_none.clear()
        if val in self.watchers:
            cb = self.watchers[val]
            if inspect.iscoroutinefunction(cb):
                await cb(val)
            else:
                cb(val)

    async def value_non_none(self):
        await self.event_non_none.wait()
        return self._value

    def on_value(self, val: T, cb: Callable):
        self.watchers[val] = cb
