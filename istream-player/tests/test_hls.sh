#!/bin/bash

iplay -i https://demo.unified-streaming.com/k8s/features/stable/video/tears-of-steel/tears-of-steel.ism/.m3u8 \
    --run_dir ./runs/test/ --log_level debug \
    --mod_abr hybrid \
    --mod_downloader tcp \
    --mod_analyzer playback_v2 \
    --mod_analyzer data_collector:plots_dir=./runs/test/plots \
    --mod_mpd hls