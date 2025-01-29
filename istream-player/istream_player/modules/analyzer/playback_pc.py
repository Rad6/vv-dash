import asyncio
from collections import defaultdict, deque
import heapq
import logging
import math
import multiprocessing
from pprint import pprint
import sys
from asyncio import create_subprocess_exec
from asyncio.subprocess import PIPE
import time
from typing import Dict, Optional, Tuple
from os.path import basename
# import cv2
import numpy as np
import DracoPy
import open3d as o3d
from concurrent.futures import ThreadPoolExecutor
import os
from io import BytesIO
from os.path import join
import subprocess
import pickle


from istream_player.config.config import PlayerConfig
from istream_player.core.analyzer import Analyzer
from istream_player.core.downloader import DownloadEventListener, DownloadManager
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.core.player import Player, PlayerEventListener
from istream_player.core.scheduler import Scheduler, SchedulerEventListener
from istream_player.models import Segment, State
from scripts.drv_decode import parse_drv


async def drain_buffer(buffer: asyncio.Queue):
    while not buffer.empty():
        buffer.get_nowait()


@ModuleOption(
    "playback_pc", requires=[Player, "segment_downloader", MPDProvider, Scheduler]
)
class Playback_pc(
    Module, Analyzer, PlayerEventListener, DownloadEventListener, SchedulerEventListener
):
    log = logging.getLogger("Playback_pc")

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
        self.segment_size: Dict[str, int] = {}
        self.init_adapids = set()

        # Downloaded chunks
        self.decoder_buffer: multiprocessing.Queue = None
        # Decoded frames
        self.frame_buffer: multiprocessing.Queue = None
        self.decoder_task: asyncio.Task = None
        self.player_task: asyncio.Task = None
        self.total_decoding = 0
        self.manager = None
        self.time_intervals = None

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
        # self.decoder_pool = ThreadPoolExecutor(max_workers=10)
        self.fps = 0
        self.dump_results_path = join(config.run_dir, "data") if config.run_dir else None
        self.manager = multiprocessing.Manager()
        self.time_intervals = self.manager.list()
        self.tmc2rs_dir = config.tmc2_decoder_path


    async def run(self) -> None:
        await self.mpd_provider.available()
        
        loop = asyncio.get_event_loop()

        self.decoder_buffer = multiprocessing.Queue()
        self.frame_buffer = multiprocessing.Queue()
        # self.decoder_task = asyncio.create_task(
        #     asyncio.to_thread(
        #         # self.decoder,
        #         self.decoder_buffer,
        #         self.frame_buffer
        #     )
        # )
        self.player_task = asyncio.create_task(asyncio.to_thread(self.player))


    async def cleanup(self) -> None:
        self.log.info(f"Closing decoder")
        self.calc_total_time()
        # self.decoder_task.cancel()
        self.log.info("Closing video")
        self.player_task.cancel()


    def calc_total_time(self):
        sorted_intervals = sorted(self.time_intervals, key=lambda x: x[0])
        merged = []

        for interval in sorted_intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(list(interval))
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        
        total_time = sum(end - start for start, end in merged)
        print(f"TOTAL DECODING TIME: {total_time}")

    # make a function call inside this function that checks if the new receiving chunk will complete a frame,
    # if so, make a decoding thread for that frame
    async def on_bytes_transferred(
        self, chunk_size: int, url: str, position: int, seg_size: int, content: bytes
    ) -> None:
        if url not in self.segment_data:
            self.segment_data[url] = bytearray(seg_size)

        offset = position - chunk_size
        self.segment_data[url][offset:position] = content



    async def on_segment_download_complete(self, index: int, segments: Dict[int, Segment]):
        assert len(segments) == 1, f"Can play only 1 stream. Found {len(segments)}"

        print(self.mpd_provider.mpd.adaptation_sets[1].representations[1].codecs)
        print(type(self.mpd_provider.mpd.adaptation_sets[1].representations[1].codecs))
        if self.mpd_provider.mpd.adaptation_sets[1].representations[1].codecs == "DRCO":
            timescale, frames = parse_drv(BytesIO(self.segment_data[list(segments.values())[0].url]))
            procs = []
            for pts, frame in frames.items():
                p = multiprocessing.Process(target=self.decoder, args=(index, pts, frame, ))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
        elif self.mpd_provider.mpd.adaptation_sets[1].representations[1].codecs == "VPCC":
            frame_rate = self.mpd_provider.mpd.adaptation_sets[1].frame_rate.split("/")

            fps = int(int(frame_rate[0])/int(frame_rate[1]))

            print(f"Frame rate: {fps} fps")

            timescale, frames = parse_drv(BytesIO(self.segment_data[list(segments.values())[0].url]))

            tmc2rs_in_dir = os.path.join(self.tmc2rs_dir, "in")
            tmc2rs_out_dir = os.path.join(self.tmc2rs_dir, "out")

            self.ensure_empty_directory(tmc2rs_in_dir)
            self.ensure_empty_directory(tmc2rs_out_dir)
            
            for pts, seg in frames.items():
                with open(os.path.join(tmc2rs_in_dir, f"seg{pts}.bin"), "wb") as f:
                    pickle.dump(seg, f)

                command = [
                "cargo",
                "run",
                "--bin",
                "decoder",
                "--",
                "-i",
                f"{tmc2rs_in_dir}/seg{pts}.bin",
                "-o",
                f"{tmc2rs_out_dir}"
                ]

                try:
                    # Run the command
                    result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=self.tmc2rs_dir)
                    
                    # Print the output
                    print("Command executed successfully!")
                    print("Output:", result.stdout)
                    print("Error (if any):", result.stderr)
                    ply_files = [f for f in sorted(os.listdir(tmc2rs_out_dir)) if f.endswith('.ply')]
                    if not ply_files:
                        print("No .ply files found in the directory.")
                        return
                    for ply_file in enumerate(ply_files):
                        ply_path = os.path.join(tmc2rs_out_dir, ply_file)
                        
                        point_cloud = o3d.geometry.PointCloud()
                        # Load the point cloud from the .ply file
                        pcd = o3d.io.read_point_cloud(ply_path)
                        
                        # Update the geometry
                        point_cloud.points = pcd.points
                        point_cloud.colors = pcd.colors if pcd.has_colors() else None

                        self.frame_buffer.put((index, pts, point_cloud, "VPCC"))
                        os.remove(ply_path)
                except subprocess.CalledProcessError as e:
                    print("Command failed with error code", e.returncode)
                    print("Error output:", e.stderr)
        else:
            sys.exit("Unsupported Codec")
            


    def decoder(
        self,
        index,
        pts,
        frame_content
    ):
        print("IN DECODER")
        # TODO: check the current playback position before decoding, to make sure you are not decoding sth that is behind

        decoded_frame = DracoPy.decode(frame_content)
        print(f"DECODED FRAME {pts}")
        self.frame_buffer.put((index, pts, decoded_frame, "DRCO"))
        # print("DECODED WITH PID: " + str(os.getpid()))
        

    def player(self):
        # pass
        # Read frames from self.frame_buffer
        # Update frame in window

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        curr_frame_num = -1
        start_time = time.time()

        ctr = vis.get_view_control()

        index, pts, frame_data, codec = self.frame_buffer.get()
        vis.add_geometry(frame_data if codec=="DRCO" else self.to_pointcloud(frame_data), reset_bounding_box=True)

        while True:
            time.sleep(0.0001)
            vis.poll_events()
            vis.update_renderer()
            frame_num = math.floor((time.time() - start_time)*self.fps)
            if frame_num != curr_frame_num:
                curr_frame_num = frame_num
                vis.clear_geometries()

                index, pts, frame_data, codec = self.frame_buffer.get()
                vis.add_geometry(frame_data if codec=="DRCO" else self.to_pointcloud(frame_data), reset_bounding_box=True)


    def to_pointcloud (self, frame_data):
        pcd = o3d.geometry.PointCloud()
        # Set the points from the numpy array
        pcd.points = o3d.utility.Vector3dVector(frame_data.points)
        # Set the colors from the numpy array
        pcd.colors = o3d.utility.Vector3dVector(frame_data.colors.astype(np.float32)/255)
        return pcd

    def ensure_empty_directory(self, directory_path):
        """
        Ensure the directory exists and is empty. If it doesn't exist, create it. If it exists, empty it.
        
        Args:
            directory_path (str): Path to the directory.
        """
        if not os.path.exists(directory_path):
            # Create the directory if it doesn't exist
            os.makedirs(directory_path)
            print(f"Directory created: {directory_path}")
        else:
            # Empty the directory if it exists
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove file or symbolic link
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove directory
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            print(f"Directory emptied: {directory_path}")
