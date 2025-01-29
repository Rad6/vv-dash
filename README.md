# vv-dash


## How to build

```bash
# Build all modules
docker compose build
```

When using V-PCC coded videos, please make sure you have TMC2-RS decoder built ([from this link](https://github.com/benclmnt/tmc2-rs)) and then put the entire directory in the istream-player directory.

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