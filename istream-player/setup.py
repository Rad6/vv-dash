#!/usr/bin/env python3

import os
from setuptools import find_packages, setup

dependencies = [
    "wsproto",
    "aiohttp",
    "requests",
    "aioquic==0.9.21",
    "pyyaml",
    #"sslkeylog",
    "pytest",
    "parameterized",
    "matplotlib",
    "m3u8",
    "pygments",
    "pyav",
    "pygame",
    "pyequilib",
    "opencv-python",
    "scikit-build",
    # "torch",
    "DracoPy",
    "open3d"
    "pyomo",
    "numpy==1.26.4"
]

if os.environ.get("headless", "").lower() == "false":
    dependencies.append('istream_renderer @ file://localhost/%s/istream_cpp/' % os.getcwd().replace('\\', '/'))
    # dependencies.append('istream_decoder @ file://localhost/%s/cpp_decoder/' % os.getcwd().replace('\\', '/'))

setup(
    name="istream-player",
    version="0.3.0",
    description="IStream DASH Player",
    packages=find_packages(),
    entry_points={"console_scripts": ["iplay=istream_player.main:main"]},
    install_requires=dependencies,
)
