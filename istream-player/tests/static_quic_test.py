# test_with_pytest.py


import logging
import unittest
from pathlib import Path
from unittest.mock import patch

from parameterized import parameterized

from istream_player.config.config import PlayerConfig
from istream_player.core.module_composer import PlayerComposer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)20s %(levelname)8s:%(message)s")


# @unittest.skip("Aioquic SSL Error")
class StaticTest(unittest.IsolatedAsyncioTestCase):
    def make_config(self, abr: str):
        return PlayerConfig(
            input="https://localhost:443/videos/av1-1sec/Aspen/output.mpd",
            run_dir="./runs/test",
            mod_abr='dash',
            mod_downloader="quic",
            log_level='debug',
            # mod_analyzer=["data_collector"],
            # time_factor=0
        )
        # config.static.max_initial_bitrate = 100_000

    async def _asyncSetUp(self):
        import docker

        client = docker.from_env()
        self.container: Container = client.containers.run(  # type: ignore
            "server_aioquic",
            command=["bash", "/etc/nginx/common/container-start.sh", "quic"],
            name="aioquic_test",
            remove=False,
            cap_add=["net_admin"],
            ports={"443": "443"},
            auto_remove=False,
            detach=True,
            volumes={
                str(Path(__file__).parent.parent.joinpath("dataset")): {"bind": "/etc/nginx/html", "mode": "ro"},
                str(Path(__file__).parent.parent.joinpath("runs", "test")): {"bind": "/run", "mode": "rw"},
            },
        )
        print(self.container)

    async def _asyncTearDown(self) -> None:
        # self.container.stop()
        pass

    @parameterized.expand([["default"]])
    async def test_static_tcp(self, abr: str):
        # save_file_patcher = patch("istream_player.modules.analyzer.analyzer.PlaybackAnalyzer.save_file")
        # save_file_mock = save_file_patcher.start()

        composer = PlayerComposer()
        composer.register_core_modules()

        async with composer.make_player(self.make_config(abr)) as player:
            await player.run()

        # save_file_patcher.stop()
        # save_file_mock.assert_called_once()
        # [path, data] = save_file_mock.call_args.args
        # print(json.dumps(data, indent=4))
        # assert len(data["segments"]) == 4


if __name__ == "__main__":
    unittest.main()
