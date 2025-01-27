# test_with_pytest.py


# import logging
import unittest
from unittest.mock import patch

from parameterized import parameterized

from istream_player.config.config import PlayerConfig
from istream_player.core.module_composer import PlayerComposer

# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)20s %(levelname)8s:\t%(message)s")


class StaticTest(unittest.IsolatedAsyncioTestCase):
    def make_config(self, abr: str):
        config = PlayerConfig(
            input="./dataset/videos/hevc-1sec/Aspen/output.mpd",
            run_dir="./runs/test",
            # plots_dir="./runs/test/plots",
            mod_abr=abr,
            mod_analyzer=[
                "playback_v2",
                # "data_collector:plots_dir=./runs/test/plots",
                "file_saver",
                "vmaf:src_dir=./dataset/videos/hevc-1sec/Aspen/"
            ],
            mod_downloader="local:bw=10_000_000",
            log_level='debug',
            time_factor=1,
        )
        config.static.max_initial_bitrate = 100_000
        return config

    @parameterized.expand([["fixed:quality=-1"]])
    async def test_static(self, abr: str):
        save_file_patcher = patch("istream_player.modules.analyzer.analyzer.PlaybackAnalyzer.save_file")
        # save_file_mock = save_file_patcher.start()

        composer = PlayerComposer()
        composer.register_core_modules()

        async with composer.make_player(self.make_config(abr)) as player:
            await player.run()

        save_file_patcher.stop()
        # save_file_mock.assert_called_once()
        # [path, data] = save_file_mock.call_args.args
        # print(json.dumps(data, indent=4))
        # assert len(data["segments"]) == 4


if __name__ == "__main__":
    unittest.main()
