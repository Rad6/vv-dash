# test_with_pytest.py


# import logging
import unittest


from istream_player.config.config import PlayerConfig
from istream_player.core.module_composer import PlayerComposer

# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)20s %(levelname)8s:\t%(message)s")


class StaticTest(unittest.IsolatedAsyncioTestCase):
    def make_config(self):
        config = PlayerConfig(
            input="dataset/videos-hls/av1-1sec/Aspen/output.m3u8",
            run_dir="./runs/test",
            mod_abr='dash',
            # mod_analyzer=["data_collector:plots_dir=./runs/test/plots"],
            mod_analyzer=['playback_v2', 'data_collector', 'file_saver', 'vmaf:src_dir=dataset/videos-hls/av1-1sec/Aspen'],
            mod_downloader="local:bw=1_000_000",
            mod_scheduler="scheduler",
            time_factor=1,
            log_level="info",
            mod_mpd='hls',
            mod_player='dash_player'
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
