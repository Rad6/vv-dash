# vv-dash


## How to build

If you're aiming to use Google Draco, please make sure you have [Google Draco](https://github.com/google/draco), and if you're aiming to use V-PCC, please sure you have [TMC2 V-PCC encoder](https://github.com/MPEGGroup/mpeg-pcc-tmc2) and [TMS2-RS V-PCC Fast Decoder](https://github.com/benclmnt/tmc2-rs) built on your system (in the same directory as VV-DASH).

VV-DASH supports any point cloud-based videos. Please download your videos (for instance, from [8i VFB dataset](https://plenodb.jpeg.org/pc/8ilabs)) and then place them in ./pc-video directory.

For preparing the encoded video bitrate ladder pipeline, please follow this order encoder -> dvv-packager -> dash-packager. (each include their own config file for customizations)

```bash
# Build all modules
docker compose build
```

## How to run

```bash
# run an experiment
bash run.sh
```

## Add Bandwidth Scenario
You can define any bandwidth scenario in the `ha-proxy/bw-config.json` file. Durations are specified in seconds, and bandwidths are in Kbps. The provided sample sets the bandwidth to 400 Mbps from time 0 to 5 seconds, 200 Mbps from 10 to 15 seconds, and 550 Mbps for the remaining duration.
```json
{
    "bandwidths": [
      400000, 200000, 550000
    ],
    "durations": 5
}
```
## Player Documentation
For the player help, please use the `entrypoint: ["iplay", "-h"]` entrypoint provided in the compose file which is commented out.

## Change DASH Parameters
You can find and configure any of the DASH parameters (including buffer duration, minimum buffer level, smoothing factor, initial startup bandwidth, etc.) in [config file](/istream-player/istream_player/config/config.py).