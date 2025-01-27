# test_with_pytest.py


# import logging
import unittest
from unittest.mock import patch

from parameterized import parameterized

from istream_player.config.config import PlayerConfig
from istream_player.core.module_composer import PlayerComposer

# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)20s %(levelname)8s:\t%(message)s")


class StaticTest(unittest.IsolatedAsyncioTestCase):
    def make_config(self):
        config = PlayerConfig(
            input="dataset/videos-raw/aspen_1080p_av1_open.mp4",
            run_dir="./runs/test",
            mod_abr='none',
            # mod_analyzer=["data_collector:plots_dir=./runs/test/plots"],
            mod_analyzer=['playback_v2'],
            # mod_downloader="local:bw=10_000_000",
            mod_scheduler="mp4_scheduler",
            time_factor=1,
            log_level="debug",
            mod_mpd='none',
            mod_player='dash_player',
            min_start_duration=1
        )
        # config.static.max_initial_bitrate = 100_000
        return config

    async def test_static(self):
        # save_file_mock = save_file_patcher.start()

        composer = PlayerComposer()
        composer.register_core_modules()

        async with composer.make_player(self.make_config()) as player:
            await player.run()


if __name__ == "__main__":
    unittest.main()
