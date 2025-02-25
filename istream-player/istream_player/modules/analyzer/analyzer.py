from dataclasses import asdict, dataclass
import datetime
import io
import json
import logging
import os
from pathlib import Path
import sys
from os.path import join
from threading import Thread
import traceback
from typing import Any, Dict, List, Optional, TextIO, Tuple
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt

from istream_player.config.config import PlayerConfig
from istream_player.core.analyzer import Analyzer
from istream_player.core.buffer import BufferEventListener, BufferManager
from istream_player.core.bw_meter import BandwidthMeter, BandwidthUpdateListener
from istream_player.core.module import Module, ModuleOption
from istream_player.core.player import Player, PlayerEventListener
from istream_player.core.scheduler import Scheduler, SchedulerEventListener
from istream_player.models import State
from istream_player.models.mpd_objects import Segment


@dataclass
class AnalyzerSegment:
    index: int
    url: str
    repr_id: int
    adap_set_id: int
    bitrate: int
    duration: float
    init_url: Optional[str]

    start_time: Optional[float] = None
    stop_time: Optional[float] = None
    dl_time: Optional[float] = None
    first_byte_at: Optional[float] = None
    last_byte_at: Optional[float] = None

    quality: Optional[int] = None
    segment_throughput: Optional[float] = None
    adaptation_throughput: Optional[float] = None

    total_bytes: Optional[int] = None
    received_bytes: Optional[int] = None
    stopped_bytes: Optional[int] = None

    @property
    def stop_ratio(self) -> Optional[float]:
        if self.total_bytes is not None and self.stopped_bytes is not None:
            return self.stopped_bytes / self.total_bytes
        else:
            return None

    @property
    def ratio(self) -> Optional[float]:
        if self.received_bytes is not None and self.total_bytes is not None:
            return self.received_bytes / self.total_bytes
        else:
            return None


@dataclass
class BufferLevel:
    time: float
    level: float


@dataclass
class Stall:
    time_start: float
    time_end: float


@ModuleOption("data_collector", default=True, requires=[BandwidthMeter, Scheduler, Player, BufferManager])
class PlaybackAnalyzer(
    Module, Analyzer, PlayerEventListener, SchedulerEventListener, BandwidthUpdateListener, BufferEventListener
):
    log = logging.getLogger("PlaybackAnalyzer")

    def __init__(self, *, live_plot: Optional[str] = None):
        self._start_time = datetime.datetime.now().timestamp()
        self._buffer_levels: List[BufferLevel] = []
        self._throughputs: List[Tuple[float, int]] = []
        self._cont_bw: List[Tuple[float, int]] = []
        self._states: List[Tuple[float, State, float]] = []
        self._segments_by_url: Dict[str, AnalyzerSegment] = {}
        self._position = 0
        self._stalls: List[Stall] = []
        self._first_segment = float("inf")
        self._last_segment = 0

        self.live_plot = float(live_plot) if live_plot is not None else None

        # FIXME: Move playback start event to some better place
        # if self.config.recorder:
        #     self.config.recorder.write_event(ExpEvent_PlaybackStart(int(self._start_time * 1000)))

    async def setup(
        self,
        config: PlayerConfig,
        bandwidth_meter: BandwidthMeter,
        scheduler: Scheduler,
        player: Player,
        buffer_manager: BufferManager,
        **kwargs,
    ):
        self.bandwidth_meter = bandwidth_meter
        self.dump_results_path = join(config.run_dir, "data") if config.run_dir else None
        self.plots_dir = config.plots_dir

        # segment_downloader.add_listener(self)
        bandwidth_meter.add_listener(self)
        scheduler.add_listener(self)
        player.add_listener(self)
        buffer_manager.add_listener(self)

    async def cleanup(self) -> None:
        try:
            self.save(sys.stdout)
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            self.log.error(f"Failed to save analysis : {e}")

    def to_relative_time(self, t: float):
        return t - self._start_time

    def curr_relative_time(self):
        return datetime.datetime.now().timestamp() - self._start_time

    async def on_position_change(self, position):
        self._position = position

    async def on_state_change(self, position: float, old_state: State, new_state: State):
        self._states.append((self.curr_relative_time(), new_state, position))

    async def on_buffer_level_change(self, buffer_level):
        self._buffer_levels.append(BufferLevel(self.curr_relative_time(), buffer_level))

    async def on_segment_download_start(self, index, adap_bw: Dict[int, float], segments: Dict[int, Segment]):
        self._first_segment = min(self._first_segment, index)
        self._last_segment = max(self._last_segment, index)
        for as_id, segment in segments.items():
            self._segments_by_url[segment.url] = AnalyzerSegment(
                index=index,
                url=segment.url,
                init_url=segment.init_url,
                repr_id=segment.repr_id,
                adap_set_id=as_id,
                adaptation_throughput=adap_bw[as_id],
                quality=segment.quality,
                bitrate=segment.bitrate,
                duration=segment.duration,
            )

    async def on_segment_download_complete(self, index: int, segments: Dict[int, Segment]):
        for segment in segments.values():
            stat = self.bandwidth_meter.get_stats(segment.url)
            assert (
                stat.stop_time is not None
                and stat.start_time is not None
                and stat.first_byte_at is not None
                and stat.last_byte_at is not None
            )
            analyzer_segment = self._segments_by_url[segment.url]

            analyzer_segment.stop_time = self.to_relative_time(stat.stop_time)
            analyzer_segment.start_time = self.to_relative_time(stat.start_time)
            analyzer_segment.dl_time = analyzer_segment.stop_time - analyzer_segment.start_time
            analyzer_segment.first_byte_at = self.to_relative_time(stat.first_byte_at)
            analyzer_segment.last_byte_at = self.to_relative_time(stat.last_byte_at)

            analyzer_segment.received_bytes = stat.received_bytes
            analyzer_segment.total_bytes = stat.total_bytes
            analyzer_segment.stopped_bytes = stat.stopped_bytes

            analyzer_segment.segment_throughput = (stat.received_bytes * 8) / (stat.stop_time - stat.start_time)
        if self.live_plot is not None and self.plots_dir is not None:
            self.save_plots(window=self.live_plot)

    async def on_bandwidth_update(self, bw: int) -> None:
        entry = (self.curr_relative_time(), bw)
        self._throughputs.append(entry)
        self._cont_bw.append(entry)

    def save(self, output: io.TextIOBase | TextIO) -> None:
        bitrates = []

        last_quality = None
        quality_switches = 0

        total_stall_duration = 0
        total_stall_num = 0
        total_idle_duration = 0

        if len(self._states) > 0 and self._states[-1][1] != State.END:
            self._states.append((self.curr_relative_time(), State.END, self._position))

        headers = ("Index", "Start", "End", "Quality", "Bitrate", "Adap-Th", "Seg-Th", "Ratio", "URL")
        output.write("%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-20s\n" % headers)
        prev_end_time = None
        for segment in sorted(self._segments_by_url.values(), key=lambda s: s.index):
            if prev_end_time is not None:
                total_idle_duration += max(segment.start_time - prev_end_time, 0)
            prev_end_time = segment.stop_time
            if last_quality is None:
                # First segment
                last_quality = segment.quality
            else:
                if last_quality != segment.quality:
                    last_quality = segment.quality
                    quality_switches += 1
            output.write(
                "%-10d%-10.2f%-10.2f%-10d%-10d%-10d%-10d%-10.2f%-20s\n"
                % (
                    segment.index,
                    segment.start_time,
                    segment.stop_time,
                    segment.quality,
                    segment.bitrate,
                    segment.adaptation_throughput,
                    segment.segment_throughput,
                    segment.ratio,
                    segment.url,
                )
            )
            bitrates.append(segment.bitrate)
        output.write("\n")

        # Stalls
        output.write("Stalls:\n")
        output.write("%-6s%-6s%-6s\n" % ("Start", "End", "Duration"))
        buffering_start = None
        self._stalls = []
        startup_delay = None
        for time, state, position in self._states:
            if state == State.BUFFERING and buffering_start is None:
                buffering_start = time

            elif state == State.READY and buffering_start is not None:
                duration = time - buffering_start
                output.write("%-6.2f%-6.2f%-6.2f\n" % (buffering_start, time, duration))

                if startup_delay is None:
                    startup_delay = duration
                else:
                    self._stalls.append(Stall(buffering_start, time))
                    total_stall_num += 1
                    total_stall_duration += duration
                buffering_start = None

        output.write("\n")
        # Stall summary
        output.write(f"Number of Stalls: {total_stall_num}\n")
        output.write(f"Total seconds of stalls: {total_stall_duration}\n")

        # Average bitrate
        average_bitrate = sum(bitrates) / len(bitrates) if len(bitrates) > 0 else 0
        output.write(f"Average bitrate: {average_bitrate:.2f} bps\n")

        # Number of quality switches
        output.write(f"Number of quality switches: {quality_switches}\n")

        if self.plots_dir is not None:
            self.save_plots()

        self.dump_results(
            self._segments_by_url,
            total_stall_num,
            total_stall_duration,
            total_idle_duration,
            startup_delay,
            average_bitrate,
            quality_switches,
            self._states,
            self._cont_bw,
        )

    def dump_results(
        self,
        segments: Dict[str, AnalyzerSegment],
        num_stall,
        dur_stall,
        dur_idle,
        startup_delay,
        avg_bitrate,
        num_quality_switches,
        states,
        cont_bw,
    ):
        data = {
            "segments": list(map(asdict, segments.values())),
            "stalls": list(map(asdict, self._stalls)),
            "num_stall": num_stall,
            "dur_stall": dur_stall,
            "dur_idle": dur_idle,
            "startup_delay": startup_delay,
            "avg_bitrate": avg_bitrate,
            "num_quality_switches": num_quality_switches,
            "states": [{"time": time, "state": str(state), "position": pos} for time, state, pos in states],
            "bandwidth_estimate": [{"time": bw[0], "bandwidth": bw[1]} for bw in cont_bw],
            "buffer_level": list(map(asdict, self._buffer_levels)),
        }

        if self.dump_results_path is not None:
            PlaybackAnalyzer.save_file(self.dump_results_path, data)  # type: ignore
        else:
            json.dump(data["segments"], sys.stdout, indent=4)

    @staticmethod
    def save_file(path: str, data: dict[str, Any]):
        extra_index = 1
        final_path = f"{path}-{extra_index}.json"
        while os.path.exists(final_path):
            extra_index += 1
            final_path = f"{path}-{extra_index}.json"

        print(f"Writing results in file {final_path}")
        with open(final_path, "w") as f:
            f.write(json.dumps(data))

    def save_plots(self, window: Optional[float] = None):
        def add_stall_bg(ax: plt.Axes, y_bottom, y_top):
            for stall in self._stalls:
                ax.add_patch(
                    Rectangle(
                        (stall.time_start, y_bottom),
                        stall.time_end - stall.time_start,
                        y_top - y_bottom,
                        facecolor=(1, 0, 0, 0.3),
                    )
                )

        def plot_bws(ax: plt.Axes):
            xs = [i[0] for i in self._throughputs]
            ys = [i[1] / 1000 for i in self._throughputs]
            lines1 = ax.plot(xs, ys, color="red", label="Throughput")
            ax.set_ylim(0)
            if window is not None:
                max_x = self.curr_relative_time()
                min_x = max_x - window
                ax.set_xlim(min_x, max_x)
            else:
                ax.set_xlim(0)
            ax.set_xlabel("Time (second)")
            ax.set_ylabel("Bandwidth (kbps)", color="red")
            return (*lines1,)

        def plot_bufs(ax: plt.Axes):
            xs = [i.time for i in self._buffer_levels]
            ys = [i.level for i in self._buffer_levels]
            line1 = ax.step(xs, ys, color="blue", label="Buffer", where="post")
            y_top = max(ys)
            y_bottom = min(ys)
            if window is not None:
                max_x = self.curr_relative_time()
                min_x = max_x - window
                ax.set_xlim(min_x, max_x)
            else:
                ax.set_xlim(0)
            ax.set_ylim(y_bottom, y_top)
            ax.set_ylabel("Buffer (second)", color="blue")
            ax.set_xlabel("Time (second)")
            add_stall_bg(ax, y_bottom, y_top)
            # line2 = ax.hlines(1.5, 0, 20, linestyles="dashed", label="Panic buffer")
            return (*line1,)

        def plot_quality(ax: plt.Axes):
            segments = sorted(self._segments_by_url.values(), key=lambda s: s.index)
            xs = [s.index for s in segments]
            ys = [s.quality or 0 for s in segments]
            line1 = ax.step(xs, ys, color="blue", label="Buffer", where="post")
            if window is not None:
                max_x = self.curr_relative_time()
                min_x = max_x - window
                ax.set_xlim(min_x, max_x)
            else:
                ax.set_xlim(0)
            ax.set_ylim(0)
            ax.set_ylabel("Qaulity", color="blue")
            ax.set_xlabel("Segment")
            add_stall_bg(ax, 0, max(ys))
            # line2 = ax.hlines(1.5, 0, 20, linestyles="dashed", label="Panic buffer")
            return (*line1,)

        def save_fig(output_file, lines_fn):
            fig, ax1 = plt.subplots()
            lines = lines_fn(ax1)
            labels = [line.get_label() for line in lines]
            fig.legend(lines, labels)
            fig.savefig(output_file)

        def save_all():
            assert self.plots_dir is not None
            save_fig(os.path.join(self.plots_dir, "bw.pdf"), plot_bws)
            save_fig(os.path.join(self.plots_dir, "buffer.pdf"), plot_bufs)
            save_fig(os.path.join(self.plots_dir, "quality.pdf"), plot_quality)

        if self.plots_dir is not None:
            Path(self.plots_dir).mkdir(parents=True, exist_ok=True)
            Thread(target=save_all).start()
