from functools import lru_cache
import pygame
import asyncio
from collections import defaultdict, deque
import heapq
import logging
from pprint import pprint
from queue import Queue
import sys
from asyncio import create_subprocess_exec
from asyncio.subprocess import PIPE
import time
from typing import Dict, Optional, Tuple
from os.path import basename
import cv2
import numpy as np

from equilib import Equi2Pers


from istream_player.config.config import PlayerConfig
from istream_player.core.analyzer import Analyzer
from istream_player.core.downloader import DownloadEventListener, DownloadManager
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.core.player import Player, PlayerEventListener
from istream_player.core.scheduler import Scheduler, SchedulerEventListener
from istream_player.models import Segment, State


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
        self.curr_time = 0
        self.offset = 0
        self.is_buffering = False

    def read(self, n):

        assert self.offset <= len(self.curr)
        if self.offset == len(self.curr):
            future = asyncio.run_coroutine_threadsafe(self.get(), self.loop)
            self.curr_time, self.curr = future.result()
            self.offset = 0

        taking = min(n, len(self.curr) - self.offset)
        data = self.curr[self.offset : self.offset + taking]
        self.offset += taking

        # print(f"Chunk returned of size : {len(data)} bytes")

        return bytes(data)


async def drain_buffer(buffer: asyncio.Queue):
    while not buffer.empty():
        buffer.get_nowait()


@lru_cache(1)
def e2p_maps(FOV, height, width):
    wFOV = FOV
    hFOV = float(height) / width * wFOV

    w_len = np.tan(np.radians(wFOV / 2.0))
    h_len = np.tan(np.radians(hFOV / 2.0))


    x_map = np.ones([height, width], np.float32)
    y_map = np.tile(np.linspace(-w_len, w_len,width), [height,1])
    z_map = -np.tile(np.linspace(-h_len, h_len,height), [width,1]).T

    D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = np.stack((x_map,y_map,z_map),axis=2)/np.repeat(D[:, :, np.newaxis], 3, axis=2)
    
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)

    xyz = xyz.reshape([height * width, 3]).T

    return (xyz, y_axis, z_axis)

@lru_cache(1)
def e2p_lonlat(equ_w, equ_h, FOV, height, width, THETA, PHI):
    equ_cx = (equ_w - 1) / 2.0
    equ_cy = (equ_h - 1) / 2.0

    xyz, y_axis, z_axis = e2p_maps(FOV, height, width)

    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

    xyz = np.dot(R1, xyz)
    xyz = np.dot(R2, xyz).T
    lat = np.arcsin(xyz[:, 2])
    lon = np.arctan2(xyz[:, 1] , xyz[:, 0])

    lon = lon.reshape([height, width]) / np.pi * 180
    lat = -lat.reshape([height, width]) / np.pi * 180

    lon = lon / 180 * equ_cx + equ_cx
    lat = lat / 90  * equ_cy + equ_cy

    return lon.astype(np.float32), lat.astype(np.float32)


def E2P(img, FOV, THETA, PHI, height, width):
    #
    # THETA is left/right angle, PHI is up/down angle, both in degree
    #

    equ_w = img.shape[1]
    equ_h = img.shape[0]
    
    lon, lat = e2p_lonlat(equ_w, equ_h, FOV, height, width, THETA, PHI)

    persp = cv2.remap(img, lon, lat, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    return persp
        

@ModuleOption(
    "playback", requires=[Player, "segment_downloader", MPDProvider, Scheduler]
)
class Playback(
    Module, Analyzer, PlayerEventListener, DownloadEventListener, SchedulerEventListener
):
    log = logging.getLogger("Playback")

    def __init__(
        self,
        *,
        stats="true",
        scale=None,
        equi2pers="false",
        border="0",
        tile_timeout="100",
    ) -> None:
        super().__init__()
        # self.segment_buffers: Dict[str, AsyncBuffer] = {}
        self.segment_data: Dict[str, bytearray] = {}
        self.init_adapids = set()

        self.decoders: Dict[int, asyncio.Task] = {}
        self.decoder_buffers: Dict[int, AsyncBuffer] = {}
        self.frame_buffers: Dict[int, Queue] = {}
        self.stats: bool = stats.lower() == "true"
        self.tile_offset: Dict[int, Tuple[int, int]] = {}
        if scale is not None:
            self.scale: Optional[Tuple[int, int]] = tuple(
                map(int, scale.lower().split("x"))
            )
        else:
            self.scale: Optional[Tuple[int, int]] = None
        self.equi2pers = equi2pers.lower() == "true"
        self.border = int(border)
        self.tile_timeout: float = int(tile_timeout) / 1000

    async def setup(
        self,
        config: PlayerConfig,
        player: Player,
        segment_downloader: DownloadManager,
        mpd_provider: MPDProvider,
        scheduler: Scheduler,
    ):
        self.mpd_provider = mpd_provider
        player.add_listener(self)
        segment_downloader.add_listener(self)
        scheduler.add_listener(self)
        self.player_module = player

    async def run(self) -> None:
        await self.mpd_provider.available()
        self.tiles = {}
        for adapId, adapSet in self.mpd_provider.mpd.adaptation_sets.items():
            if adapSet.content_type != "video":
                continue
            # _, col, row, tile_cols, tile_rows, grid_cols, grid_rows
            self.tiles[adapId] = adapSet.srd
        self.log.info(f"{self.tiles=}")
        assert len(self.tiles) >= 1, "No video tile to play"

        if len(self.tiles) > 1 and self.scale is None:
            self.scale = (1920, 1080)
            self.log.info(f"Setting default scale for multi tile video: {self.scale=}")

        loop = asyncio.get_event_loop()
        for adapId, srd in self.tiles.items():
            if srd is not None:
                _, col, row, tile_cols, tile_rows, grid_cols, grid_rows = srd
                top_left = (col * self.scale[0]) // grid_cols, (
                    row * self.scale[1]
                ) // grid_rows
                bottom_right = ((col + tile_cols) * self.scale[0]) // grid_cols, (
                    (row + tile_rows) * self.scale[1]
                ) // grid_rows
                self.tile_offset[adapId] = top_left
            else:
                self.tile_offset[adapId] = (0, 0)

            self.decoder_buffers[adapId] = AsyncBuffer()
            self.frame_buffers[adapId] = Queue()
            self.decoders[adapId] = asyncio.create_task(
                asyncio.to_thread(
                    self.decoder,
                    self.decoder_buffers[adapId],
                    self.frame_buffers[adapId],
                    (
                        (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
                        if srd is not None
                        else None
                    ),
                    loop,
                )
            )
        self.player_task = asyncio.create_task(self.player())

    async def cleanup(self) -> None:
        self.log.info(f"Closing {len(self.decoders)} decoders")
        for adapId, decoder in self.decoders.items():
            decoder.cancel()
        self.log.info("Closing video")
        self.player_task.cancel()

    async def on_bytes_transferred(
        self, chunk_size: int, url: str, position: int, seg_size: int, content
    ) -> None:
        if url not in self.segment_data:
            self.segment_data[url] = bytearray(seg_size)

        offset = position - chunk_size
        self.segment_data[url][offset:position] = content

    async def on_segment_playback_start(self, segments: Dict[int, Segment]):
        """Callback executed when a segment is played by the player

        Args:
            segment (Segment): The playback segment
        """
        seg_time = time.time()

        for adapid, segment in segments.items():
            seg_data = self.segment_data.pop(segment.url)
            if adapid not in self.init_adapids:
                self.init_adapids.add(adapid)
                seg_data = self.segment_data.pop(segment.init_url) + seg_data

            if adapid in self.decoder_buffers:
                await self.decoder_buffers[adapid].put((seg_time, seg_data))

    def decoder(
        self,
        buffer: AsyncBuffer,
        frame_buffer: Queue,
        tile_scale: Optional[tuple[int, int]],
        loop: asyncio.AbstractEventLoop,
    ):
        import av

        with av.open(buffer) as container:
            stream = container.streams.video[0]
            seg_time = -1
            pts_start = -1

            for frame in container.decode(stream):
                rgb = frame.to_rgb().to_ndarray()
                if tile_scale is not None:
                    rgb = cv2.resize(rgb, tile_scale)
                if self.border:
                    w, h = rgb.shape[1], rgb.shape[0]
                    cv2.rectangle(rgb, (0, 0), (w, h), (0, 0, 255), self.border)
                
                # rgb = rgb.transpose((1, 0, 2))
                # Lock frame buffer so that it dose not become empty in between
                with frame_buffer.mutex:
                    if seg_time != buffer.curr_time:
                        # Clear queue thread safe
                        frame_buffer.queue.clear()
                        seg_time = buffer.curr_time
                        pts_start = frame.pts

                    frame_time = stream.time_base * (frame.pts - pts_start) + seg_time
                    frame_buffer.queue.append((frame_time, rgb))

    def write_stats(
        self, img, txt, offset=(0, 0), anchor=(0, 0), padding=10
    ):  # Anchor at 0%, 0%
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (255, 255, 255)
        thickness = 2
        lineType = cv2.LINE_AA
        txt_size = cv2.getTextSize(txt, font, fontScale, thickness)[0]
        rect_size = txt_size[0] + padding * 2, txt_size[1] + padding * 2
        anchor = int(rect_size[0] * -anchor[0]), int(rect_size[1] * -anchor[1])
        rect_pos = offset[0] + anchor[0], offset[1] + anchor[1]
        txt_pos = rect_pos[0] + padding, rect_pos[1] + padding + txt_size[1]
        img = cv2.rectangle(
            img,
            rect_pos,
            (rect_pos[0] + rect_size[0], rect_pos[1] + rect_size[1]),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(
            img,
            txt,
            txt_pos,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )
    
    async def player(self):
        self.log.info("Pygame initializing")
        pygame.init()
        display = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
        pygame.display.set_caption('IStream Player')
        self.log.info("Pygame display initialized")
        stat_fps = 0
        stat_window = 1
        # Keeps the render times upto stat_window second ago
        render_times = defaultdict(deque)
        render_window = 0.010  # 10 millisecond

        img = (
            np.zeros((1920, 1080, 3), np.uint8)
            if self.scale is None
            else np.zeros((self.scale[1], self.scale[0], 3), np.uint8)
        )

        if self.equi2pers:
            THETA = 0
            PHI = 0

        tile_display_info: Dict[int, (float, (int, int), (int, int))] = {}
        self.window_size = None

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    exit(0)
                elif event.type == pygame.VIDEORESIZE:
                    # There's some code to add back window content here.
                    self.window_size = event.w, event.h
                    pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN and self.equi2pers:
                    if event.key == pygame.K_RIGHT:
                        THETA = (THETA + 10) % 360
                    elif event.key == pygame.K_LEFT:
                        THETA = (THETA - 10) % 360
                    elif event.key == pygame.K_UP:
                        PHI = (PHI + 10) % 360
                    elif event.key == pygame.K_DOWN:
                        PHI = (PHI - 10) % 360
                    
            frames = {}
            display_time = time.time()

            for adapId, frame_buffer in self.frame_buffers.items():
                with frame_buffer.mutex:
                    q = frame_buffer.queue
                    while len(q) > 0 and (display_time - q[0][0]) > render_window:
                        q.popleft()

                    if len(q) == 0 or (q[0][0] - display_time) > render_window:
                        # Frame is later than render window
                        continue
                    tile_img = frames[adapId] = frame_buffer.queue.popleft()[1]
                x1, y1 = self.tile_offset[adapId]
                x2, y2 = x1 + tile_img.shape[1], y1 + tile_img.shape[0]
                tile_display_info[adapId] = (display_time, (x1, y1), (x2, y2))
                render_times[adapId].append(display_time)
                while (display_time - render_times[adapId][0]) > stat_window:
                    render_times[adapId].popleft()

            # If there are frames to update
            if len(frames) > 0:
                if self.scale is None:
                    assert (
                        len(frames) == 1
                    ), "Cannot display multi tile without scale"
                    # Single tile video
                    img = frames[0]
                else:
                    # Merge tiles
                    for adapId, tile_img in frames.items():
                        _, (x1, y1), (x2, y2) = tile_display_info[adapId]
                        img[y1:y2, x1:x2, :] = tile_img

            if self.tile_timeout > 0:
                for adapId, (
                    tile_update_time,
                    (x1, y1),
                    (x2, y2),
                ) in tile_display_info.items():
                    if (display_time - tile_update_time) > self.tile_timeout:
                        img[y1:y2, x1:x2, :] = np.zeros((y2 - y1, x2 - x1, 3))

            final_img = img

            # Equirectanguler to perspective Project
            if self.equi2pers:
                # final_img = final_img.transpose((2, 0, 1))
                # final_img = equi2pers(
                #     equi=final_img,
                #     rots=rots,
                # )
                # final_img = np.transpose(final_img, (1, 2, 0))
                final_img = E2P(final_img, 90, THETA, PHI, self.scale[1], self.scale[0])
            else:
                final_img = final_img.copy()

            # If there are stats
            if self.stats:
                tile_fps = [
                    len(q) / (q[-1] - q[0])
                    for q in render_times.values()
                    if len(q) > 1
                ]
                if len(tile_fps) > 0:
                    stat_fps = np.mean(tile_fps)
                
                self.write_stats(final_img, f"FPS: {stat_fps:.2f}")
            # self.log.info("Stats written")

            # If the video is buffering
            if self.player_module.state == State.BUFFERING:
                final_img = (final_img * 0.75).astype("uint8")
                self.write_stats(
                    final_img,
                    "Buffering ...",
                    offset=(final_img.shape[1] // 2, final_img.shape[0] // 2),
                    anchor=(0.5, 0.5),
                )

            # if pygame.display.get_surface().get_size() != self.scale:
            #     pygame.display.set_mode(self.scale, pygame.RESIZABLE)
            final_img_pos = (0, 0)
            if self.window_size is not None:
                # Vertical snap
                h = self.window_size[1]
                w = (self.window_size[1] * final_img.shape[1])//final_img.shape[0]
                if w > self.window_size[0]:
                    w = self.window_size[0]
                    h = (self.window_size[0] * final_img.shape[0]) // final_img.shape[1]
                final_img = cv2.resize(final_img, (w, h))
                final_img_pos = ((self.window_size[0]-final_img.shape[1]) // 2), ((self.window_size[1]-final_img.shape[0]) // 2)
            final_img = final_img.transpose((1, 0, 2))
            surf = pygame.surfarray.make_surface(final_img)
            display.blit(surf, final_img_pos)
            pygame.display.update()
            await asyncio.sleep(0.01)
        self.log.info("Quitting Pygame")
        pygame.quit()
