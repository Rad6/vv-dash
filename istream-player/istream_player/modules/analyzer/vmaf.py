import asyncio
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
from threading import Thread
from typing import Dict, List
from os.path import basename, join, exists

from matplotlib import pyplot as plt
import matplotlib
from istream_player.config.config import PlayerConfig
from istream_player.core.analyzer import Analyzer
from istream_player.core.module import Module, ModuleOption
from istream_player.core.player import Player, PlayerEventListener
from istream_player.models.mpd_objects import Segment
from istream_player.models.player_objects import State
from istream_player.modules.analyzer.ffmpeg import Ffmpeg

matplotlib.use("Agg")


@ModuleOption("vmaf", requires=["file_saver", Player])
class VMAFAnalyzer(Module, Analyzer, PlayerEventListener):
    log = logging.getLogger("VMAFAnalyzer")

    def __init__(self, *, src_dir: str) -> None:
        super().__init__()
        self.segments: Dict[int, List[Segment]] = defaultdict(list)
        self.src_dir = src_dir
        self.ref_dir = join(src_dir, "encoded")
        Path(self.ref_dir).mkdir(exist_ok=True)

        # Results
        self.seg_metrics: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 0))
        self.as_metrics: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 0))

    def setup(self, config: PlayerConfig, file_saver: Analyzer, player: Player):
        assert config.run_dir is not None, "--run_dir is required by VMAFAnalyzer"
        self.plots_dir = config.plots_dir

        player.add_listener(self)
        self.downloaded_dir = join(config.run_dir, "downloaded")
        self.dis_dir = join(config.run_dir, "encoded")
        self.vmaf_dir = join(config.run_dir, "vmaf")
        Path(self.dis_dir).mkdir(exist_ok=True, parents=True)
        Path(self.vmaf_dir).mkdir(exist_ok=True, parents=True)

    async def on_segment_playback_start(self, segments: Dict[int, Segment]):
        for as_id, segment in segments.items():
            self.segments[as_id].append(segment)
        await asyncio.sleep(0)

    async def on_state_change(self, position: float, old_state: State, new_state: State):
        if new_state != State.END:
            return
        await self.run_vmaf()
        await asyncio.sleep(0)

    async def decode_segment(self, segment: Segment, base_dir: str, out_dir: str):
        # -e: exteneded
        out_path = join(out_dir, f"seg_{segment.as_id:03d}_{segment.index:03d}_{segment.repr_id:03d}_e.mp4")
        if exists(out_path):
            return out_path
        filter = f"scale=1920:1080,tpad=stop_mode=clone:stop=-1,trim=end={segment.duration}"
        paths = []
        if segment.init_url is not None:
            paths.append(join(base_dir, basename(segment.init_url)))
        paths.append(join(base_dir, basename(segment.url)))

        self.log.info(f"{paths=}")

        await Ffmpeg.decode_segment(paths, out_path, filter)
        return out_path

    async def run_vmaf(self):
        for as_id, segments in self.segments.items():
            for segment in segments:
                dis_path = await self.decode_segment(segment, self.downloaded_dir, self.dis_dir)
                ref_path = await self.decode_segment(segment, self.src_dir, self.ref_dir)
                vmaf_path = join(self.vmaf_dir, f"vmaf_{segment.as_id}_{segment.index}_{segment.repr_id}.json")
                if not exists(vmaf_path):
                    vmaf = await Ffmpeg.calculate_vmaf(ref_path, dis_path)
                    with open(vmaf_path, "w") as f:
                        f.write(json.dumps(vmaf, indent=4))
                else:
                    with open(vmaf_path) as f:
                        vmaf = json.load(f)

                self.seg_metrics[segment.index]["total_frames"] += len(vmaf["frames"])
                self.as_metrics[segment.as_id]["total_frames"] += len(vmaf["frames"])

                for frame in vmaf["frames"]:
                    self.seg_metrics[segment.index]["vmaf"] += frame["metrics"]["vmaf"]
                    self.seg_metrics[segment.index]["psnr"] += frame["metrics"]["psnr_y"]
                    self.seg_metrics[segment.index]["ssim"] += frame["metrics"]["float_ssim"]

                    self.as_metrics[segment.as_id]["vmaf"] += frame["metrics"]["vmaf"]
                    self.as_metrics[segment.as_id]["psnr"] += frame["metrics"]["psnr_y"]
                    self.as_metrics[segment.as_id]["ssim"] += frame["metrics"]["float_ssim"]

    async def cleanup(self):
        print("\n\n\nQuality by adaptation sets")
        print("%-14s%-12s%-10s%-10s%-10s" % ("Adap set ID", "Num frames", "VMAF", "SSIM", "PSNR"))
        for as_id, metric in self.as_metrics.items():
            t = metric["total_frames"]
            print("%-14d%-12d%-10.2f%-10.2f%-10.2f" % (as_id, t, metric["vmaf"] / t, metric["ssim"] / t, metric["psnr"] / t))

        print("\n\n\nQuality by segments")
        print("%-14s%-12s%-10s%-10s%-10s" % ("Segment ID", "Num frames", "VMAF", "SSIM", "PSNR"))
        for seg_id, metric in self.seg_metrics.items():
            t = metric["total_frames"]
            print("%-14d%-12d%-10.2f%-10.2f%-10.2f" % (seg_id, t, metric["vmaf"] / t, metric["ssim"] / t, metric["psnr"] / t))

        await asyncio.sleep(0)
        if self.plots_dir is not None:
            self.save_plots()

    def save_plots(self):
        def plot_seg_vmaf(ax: plt.Axes):
            xs = [i for i in self.seg_metrics.keys()]
            ys = [m["vmaf"] / m["total_frames"] for m in self.seg_metrics.values()]
            line1 = ax.plot(xs, ys)
            ax.set_xlim(0)
            ax.set_ylim(top=100)
            ax.set_ylabel("VMAF")
            ax.set_xlabel("Segment")
            return (*line1,)

        def plot_seg_ssim(ax: plt.Axes):
            xs = [i for i in self.seg_metrics.keys()]
            ys = [m["ssim"] / m["total_frames"] for m in self.seg_metrics.values()]
            line1 = ax.plot(xs, ys)
            ax.set_xlim(0)
            # ax.set_ylim(top=100)
            ax.set_ylabel("SSIM")
            ax.set_xlabel("Segment")
            return (*line1,)

        def plot_seg_psnr(ax: plt.Axes):
            xs = [i for i in self.seg_metrics.keys()]
            ys = [m["psnr"] / m["total_frames"] for m in self.seg_metrics.values()]
            line1 = ax.plot(xs, ys)
            ax.set_xlim(0)
            # ax.set_ylim(top=100)
            ax.set_ylabel("PSNR")
            ax.set_xlabel("Segment")
            return (*line1,)

        def save_fig(output_file, lines_fn):
            fig, ax1 = plt.subplots()
            lines = lines_fn(ax1)
            labels = [line.get_label() for line in lines]
            fig.legend(lines, labels)
            fig.savefig(output_file)

        def save_all():
            assert self.plots_dir is not None
            save_fig(os.path.join(self.plots_dir, "vmaf.pdf"), plot_seg_vmaf)
            save_fig(os.path.join(self.plots_dir, "psnr.pdf"), plot_seg_psnr)
            save_fig(os.path.join(self.plots_dir, "ssim.pdf"), plot_seg_ssim)

        if self.plots_dir is not None:
            Path(self.plots_dir).mkdir(parents=True, exist_ok=True)
            Thread(target=save_all).start()
