import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import socket
import subprocess
from time import time
from typing import List

from istream_player.core.mpd_provider import MPDProvider

from istream_player.modules.analyzer.exp_events import ExpEvent_BwSwitch
from istream_player.modules.analyzer.exp_recorder import ExpWriter
from istream_player.config.config import PlayerConfig
from istream_player.core.analyzer import Analyzer
from istream_player.core.module import Module, ModuleOption
from istream_player.modules.analyzer.event_logger import EventLogger
from istream_player.modules.analyzer.exp_recorder import ExpWriterJson

IF_NAME = "eth0"
NETEM_LIMIT = 1000


class NetworkConfig:
    def __init__(self, bw, latency, drop, sustain, recorder, log, eval_params={}):
        self.bw = bw
        self.latency = latency
        self.drop = drop
        self.sustain = sustain
        self.recorder: ExpWriter = recorder
        self.log = log
        self.eval_params = eval_params
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP

    def run_in_container(self, script: str):
        self.log.info("Running inside container: " + script)
        self.sock.sendto(script.encode(), ("server", 4444))

    def run_on_host(self, script):
        subprocess.check_call(script, shell=True, executable="/bin/bash", stderr=subprocess.STDOUT)
        self.log.info("Running on host: " + script)

    async def apply(self, if_name):
        bw = eval(self.bw, self.eval_params)
        script = f"""
        set -e
        tc qdisc change dev {if_name} root netem limit 100000 rate {bw}kbit delay {self.latency}ms loss {float(self.drop) * 0:.3f}%
        """
        t = round(time() * 1000)
        executor = ThreadPoolExecutor()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, self.run_in_container, script)
        self.recorder.write_event(ExpEvent_BwSwitch(t, float(bw), float(self.latency), float(self.drop)))

    def setup(self, if_name):
        bw = eval(self.bw, self.eval_params)
        script = f"""
        set -e
        tc qdisc del dev {if_name} root || true
        tc qdisc add dev {if_name} root netem rate {bw}kbit delay {self.latency}ms loss {float(self.drop) * 0:.3f}%
        """
        self.run_in_container(script)


@ModuleOption("network_manager", requires=["progress_logger", MPDProvider], daemon=True)
class NetworkManager(Module, Analyzer):
    log = logging.getLogger("NetworkManager")

    def __init__(self):
        self.force_stop = False
        self.delay = 1
        self.timeline: List[NetworkConfig] = []
        self.if_name = "eth0"

    async def setup(self, config: PlayerConfig, event_logger: EventLogger, mpd_provider: MPDProvider):
        self.mpd_provider = mpd_provider
        try:
            self.crf = int(config.input.split("/")[-2][-2:])
        except Exception:
            self.crf = None

        assert config.bw_profile is not None, "Bandwidth profile is needed by network manager"
        self.recorder: ExpWriterJson = event_logger.recorder    # type: ignore
        self.bw_profile: str = config.bw_profile
        last_line = ""
        for line in self.bw_profile.splitlines():
            if line == last_line:
                self.timeline[-1].sustain += self.delay
                continue
            last_line = line
            [bw, latency, drop, *_] = line.strip().split(" ")
            if len(line.strip().split(" ")) > 3:
                delay = int(line.strip().split(" ")[3])
            else:
                delay = self.delay
            self.timeline.append(
                NetworkConfig(
                    bw,
                    latency,
                    drop,
                    delay,
                    self.recorder,
                    log=self.log,
                    eval_params={"crf": self.crf},
                )
            )

        self.timeline[0].setup(self.if_name)

    async def run(self):
        self.log.debug("Waiting for MPD file")
        await self.mpd_provider.available()
        self.log.debug("MPD file available")
        assert self.mpd_provider.mpd is not None

        for config in self.timeline:
            config.eval_params["mpd"] = self.mpd_provider.mpd
            await config.apply(self.if_name)
            self.log.info(f"Sustain Network Config for {config.sustain} seconds")
            for s in range(config.sustain):
                await asyncio.sleep(1)

    # async def cleanup(self):
    #     self.log.info("Stopping Network Manager in background")
