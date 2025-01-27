import argparse
import json
from pathlib import Path
from pprint import pprint
import sys
from os.path import join
import os


def parse_drv(segment_content: bytes):

    version = tuple(segment_content.read(3))
    assert version == (1, 0, 0), "Only DRV version 1.0.0 is supported."
    header_format = segment_content.read(4).decode("ascii")
    assert header_format.upper() == "JSON", "Only JSON header format is supported."

    header_size = int.from_bytes(segment_content.read(4), "big")
    header = json.loads(segment_content.read(header_size).decode("ascii"))
    # pprint(header)

    pos = 0
    frame_num = 0

    # key:pts, value:frame content
    frames_contents = {}
    timescale = header["timescale"]
    for frame in header["frames"]:
        offset = frame["offset"]
        size = frame["size"]
        pts = frame["pts"]

        assert offset == pos, "Invalid data found. Frame offset does not match"

        frames_contents[pts] = segment_content.read(size)
        # with open(join(args.outDir, f"frame_{frame_num:05d}.drc"), "wb") as f:
        #     f.write(segment_content(size))
        pos += size
        frame_num += 1
        
    return timescale, frames_contents

