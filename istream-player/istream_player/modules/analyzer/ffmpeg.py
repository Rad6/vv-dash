from functools import cache
from genericpath import exists
import json
import os
from pathlib import Path
import asyncio
import subprocess
import sys
import time
from typing import List, Optional
from os.path import join, dirname


class Ffmpeg:
    @staticmethod
    async def decode_segment(input_segments: List[str], output: str, video_format: str):
        print(f"Joining segments : {input_segments}")
        bin_read, bin_write = os.pipe()

        await asyncio.create_subprocess_exec(*["cat", *input_segments], stdout=bin_write)
        os.close(bin_write)

        print(["ffmpeg", "-err_detect", "ignore_err", "-i", "-", "-vf", video_format, "-y", output])
        proc2 = await asyncio.create_subprocess_exec(
            *["ffmpeg", "-err_detect", "ignore_err", "-i", "-", "-vf", video_format, "-y", output],
            stdin=bin_read,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr,
        )
        os.close(bin_read)

        await proc2.stdout.read()  # type: ignore

    @staticmethod
    def concat_segments(input_segments: List[str], output: str, filters: List[str] = []):
        tmp_path = f"/tmp/segments-{round(time.time() * 1000)}.txt"
        print(f"Segments list file : {tmp_path}")
        with open(tmp_path, "w") as f:
            for seg in input_segments:
                f.write(f"file '{seg}'\n")
        subprocess.check_call(
            ["ffmpeg", "-f", "concat", "-safe", "0", "-i", tmp_path, "-c", "copy", "-y", output],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    @staticmethod
    def generate_buffering_segment(frame_rate: float, duration: float, output_path: Optional[str] = None):
        if output_path is None:
            output_path = f"/tmp/buff-{frame_rate}-{duration}.mp4"
        subprocess.check_call(
            [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                f"color=size=1920x1080:duration={duration}:rate={frame_rate}:color=black",
                "-vf",
                "drawtext=fontfile=$FONTFILE:fontsize=60:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:text='BUFFERING ...'",
                output_path,
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        return output_path

    @staticmethod
    def cut_video(source_path: str, start: float, end: float, output: Optional[str] = None):
        if output is None:
            output = f"/tmp/cut-{round(time.time() * 1000)}.mp4"
        subprocess.check_call(
            ["ffmpeg", "-ss", f"{start}", "-i", source_path, "-c", "copy", "-t", f"{end-start}", output],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        return output

    @staticmethod
    def get_frames(video_path):
        return subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-show_frames", "-of", "json", video_path], stderr=sys.stderr
        ).decode()

    @staticmethod
    @cache
    def video_details(video_path):
        return json.loads(
            subprocess.check_output(
                ["ffprobe", "-v", "quiet", "-show_streams", "-of", "json", video_path], stderr=sys.stderr
            ).decode()
        )

    @staticmethod
    def get_frame_rate(video_path: str) -> float:
        frame_rate = Ffmpeg.video_details(video_path)["streams"][0]["r_frame_rate"].split("/")
        if len(frame_rate) == 2:
            return float(frame_rate[0]) / float(frame_rate[1])
        elif len(frame_rate) == 1:
            return float(frame_rate[0])
        else:
            raise Exception(f"Unknow frame rate format {frame_rate}")

    @staticmethod
    def get_duration(video_path: str) -> float:
        return float(Ffmpeg.video_details(video_path)["streams"][0]["duration"])

    @staticmethod
    def get_video_with_resolution(source_path: str, resolution: str, output_path: Optional[str] = None):
        if output_path is None:
            output_path_comp = source_path.rsplit(".", 1)
            output_path_comp[0] += f"_{resolution}"
            output_path = ".".join(output_path_comp)

        if exists(output_path):
            return output_path

        res = {"1080": "1920x1080", "720": "1280x720", "480": "854x480", "360": "640x360", "240": "480x240"}[resolution]

        subprocess.check_call(
            ["ffmpeg", "-y", "-i", source_path, "-s", f"{res}", output_path], stdout=sys.stdout, stderr=sys.stderr
        )

        return output_path

    @staticmethod
    def encode_hevc_video(source_path: str, bitrate: int, resolution: str, seg_length: float, output: Optional[str] = None):
        if output is None:
            output = join(dirname(source_path), "DROP0", f"{bitrate}bps", "video.hevc")
        Path(dirname(output)).mkdir(parents=True, exist_ok=True)

        scaled_source_path = Ffmpeg.get_video_with_resolution(source_path, resolution)
        assert scaled_source_path is not None

        frame_rate = Ffmpeg.get_frame_rate(source_path)
        keyint = seg_length * frame_rate

        x265_params = (
            f"no-open-gop=1:no-scenecut=1:preset=veryslow:keyint={keyint}:min-keyint={keyint}:"
            + f"bitrate={bitrate//1000}:strict-cbr=1:vbv-maxrate={bitrate//1000}:vbv-bufsize={bitrate//2000}:b-adapt=0:bframes=16"
        )
        subprocess.check_call(
            ["ffmpeg", "-y", "-i", scaled_source_path, "-c:v", "libx265", "-an", "-x265-params", x265_params, output],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        return output

    @staticmethod
    def create_dash_playlist(source_path: str, seg_length: float, output_dir: Optional[str] = None):
        if output_dir is None:
            output_dir = join(dirname(source_path), "dash")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        mp4_path = join(output_dir, "video.mp4")

        if not exists(mp4_path):
            # Convert to mp4
            subprocess.check_call(
                ["ffmpeg", "-y", "-i", source_path, "-c", "copy", mp4_path], stdout=sys.stdout, stderr=sys.stderr
            )

        # MP4Box -dash 1000 -rap -segment-name segment_
        subprocess.check_call(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{output_dir}:/work",
                "sambaiz/mp4box",
                "-dash",
                f"{seg_length*1000}",
                "-rap",
                "-segment-name",
                "segment_",
                "video.mp4",
            ],
            cwd=output_dir,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    @staticmethod
    async def calculate_vmaf(ref_path: str, dis_path: str):
        proc = await asyncio.create_subprocess_exec(
            *[
                "ffmpeg",
                "-i",
                ref_path,
                "-i",
                dis_path,
                "-v",
                "error",
                "-lavfi",
                "libvmaf",
                "-filter_complex",
                """[0:v]scale=1920x1080:flags=bicubic[ref];
                [1:v]scale=1920x1080:flags=bicubic[main];
                [main][ref]libvmaf=psnr=true:ssim=true:log_fmt=json:log_path=/dev/stdout:n_threads=12""",
                "-f",
                "null",
                "-",
            ],
            stderr=sys.stderr,
            stdout=asyncio.subprocess.PIPE
        )
        assert proc.stdout is not None
        text = await proc.stdout.read()
        text = text.decode()
        output = json.loads(text)
        return output
