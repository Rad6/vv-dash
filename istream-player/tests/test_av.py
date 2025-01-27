from concurrent.futures import Future
from threading import Thread
from time import sleep
import av
import av.datasets
import asyncio
import cv2
import numpy as np


class AsyncBuffer(asyncio.Queue):
    """
    Minimal buffer which *only* implements the read() method.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop = None):
        super().__init__()
        
        if loop is None:
            loop = asyncio.get_event_loop()
            
        self.loop = loop
        self.curr = bytes()
        self.offset = 0

    def read(self, n):
        data = bytearray()
        
        while len(data) < n:
            assert self.offset <= len(self.curr)
            if self.offset >= len(self.curr):
                future = asyncio.run_coroutine_threadsafe(self.get(), self.loop)
                self.curr = future.result()
                self.offset = 0
            
            taking = min((n - len(data)), len(self.curr)-self.offset)
            data.extend(self.curr[self.offset:self.offset+taking])
            self.offset += taking
            
        return bytes(data)

video_paths = [
    # "dataset/av1/bbb_250ms_30s_i_crf30.mp4"
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_init.mp4",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_1.m4s",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_2.m4s",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_3.m4s",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_4.m4s",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_5.m4s",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_6.m4s",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_7.m4s",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_8.m4s",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_9.m4s",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_10.m4s",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_11.m4s",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_12.m4s",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_13.m4s",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_14.m4s",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_15.m4s",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_16.m4s",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_17.m4s",
    "dataset/dash/bbb/bbb_1000ms_30s_i_crf30/segment_18.m4s",
]

async def slow_reader(buffer: AsyncBuffer):
    for segment in video_paths:
        with open(segment, "rb") as f:
            await buffer.put(f.read())
            await asyncio.sleep(0.24)
            
            # while data := f.read(10_000_000):
            #     await buffer.put(data)
            # total += len(data)
            # print(f"Total bytes pushed: {total}")
            # await asyncio.sleep(1)
    # print("File closed")


def player(buffer: AsyncBuffer):
    cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)
    
    with av.open(buffer) as container:
        stream = container.streams.video[0]

        for frame in container.decode(stream):
            print(frame)
            rgb = frame.to_rgb().to_ndarray()
            
            cv2.imshow("Video", rgb)
            cv2.waitKey(1000//24)
            
    cv2.destroyAllWindows()
    

async def main():
    data_buffer = AsyncBuffer()
    
    reader_task = asyncio.create_task(slow_reader(data_buffer))
    player_task = asyncio.to_thread(player, data_buffer)
    
    await asyncio.gather(reader_task, player_task)

if __name__ == "__main__":
    asyncio.run(main())
