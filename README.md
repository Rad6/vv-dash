# vv-dash

VV-DASH is a framework for volumetric video (dynamic point clouds) adaptive streaming over DASH. It encapsulates the entire streaming setup, from preparing the dataset (encoding into a bitrate ladder and DASH packaging) to streaming over an actual emulated network and decoding on the client side. VV-DASH currently integrates Google Draco and MPEG-VPCC as encoder/decoder options.

For a detailed overview of features and analysis, please refer to the paper: [https://doi.org/10.1145/3712676.3718339](https://doi.org/10.1145/3712676.3718339)


## How to Run

We provide two options for running VV-DASH:

1. Using self-contained Docker images from Docker Hub
    - This option allows for easy integration with no dataset preparation required.
    - A sample dataset (Longdress from 8i VFB) is already downloaded, encoded, DASH-packaged, and included in the server Docker image.
    - No build process is required.
    - However, it offers only a fixed configuration.
2. Building from Dockerfiles
    - If you prefer a customized setup, you can build VV-DASH from scratch using our provided Dockerfiles.
    - This option allows you to choose your dataset, encoding options, network emulation scenarios, DASH client settings, etc.
    - However, it will require you to have the encoding softwares installed for preparing the dataset.

In the following, we'll discuss instructions for both options.
## Option 1: Using Pre-built Docker Images
If you're wishing to use the self-contained docker images, the instructions for running the setup is as easy as running this single command:


```bash
bash run.sh
```

This will start three containers: Server, HAProxy, and DASH Client.
 - The server already contains the Longdress video dataset, encoded using Google Draco into five quality levels, DASH-packaged, and DVV-segmented.
 - The HAProxy applies a predefined bandwidth scenario, as detailed in the `ha-proxy/bw-config.json` file.
 - Once the Server and HAProxy are running, the DASH Client (iStream Player) will begin streaming the video. At the end of the session, it will output analytical data based on the streaming performance.

## Option 2: Building from Dockerfiles
If you wish to build the modules from scratch, you will first need to install the required encoders.
- To use Google Draco, ensure you have [Google Draco](https://github.com/google/draco) installed.
- To use V-PCC, install [TMC2 V-PCC encoder](https://github.com/MPEGGroup/mpeg-pcc-tmc2) and [TMS2-RS V-PCC Fast Decoder](https://github.com/benclmnt/tmc2-rs) built on your system (in the same directory as VV-DASH).

VV-DASH supports any point cloud-based videos. You can download datasets from sources such as [8i VFB dataset](https://plenodb.jpeg.org/pc/8ilabs).

### Endoing Pipeline
To prepare an encoded video bitrate ladder, follow this sequence:

Encoder → DVV Packager → DASH Packager

Each step includes its own configuration file for customization.
1. Encode Dataset into a Bitrate Ladder: Navigate to the `./encoder` directory and choose your encoder (Draco or V-PCC). Modify `config.json` with the required settings for your dataset, then run:
    ```bash
    # Encode dataset into a bitrate ladder
    python encode.py
    ```

2. Create DVV Segments: Navigate to the `./dvv-packager` directory, adjust the `config.json` file according to the encoded dataset, and execute:

    ```bash
    # Create DVV Segments
    python dvv-encode.py
    ```

3. Generate DASH Manifest (MPD): Navigate to the `./dash-packager` directory, update `config.json` with dataset characteristics, particularly ensuring bitrate levels are correctly inserted based on your encoded dataset, and sorted from lowest to highest, then run:

    ```bash
    # Create DVV Segments
    python mpd-generator.py
    ```

### Storing the Dataset on the Server

By now, the dataset is prepared and all ready to be stored on the server. For doing so, place the directory containing your dataset (including your DVV video segments and your MPD file) into `./dash-server/videos`, and copy the entire directory from local into the container in the server's Dockerfile. For instance:

```dockerfile
COPY videos/longdress/* /usr/local/nginx/html/longdress/
```

### Add Bandwidth Scenario:
You can define any bandwidth scenario in the `ha-proxy/bw-config.json` file. Durations are specified in seconds, and bandwidths are in Kbps. 

Example:
```json
{
    "bandwidths": [
      400000, 200000, 550000
    ],
    "durations": 5
}
```
This configuration sets the bandwidth to:

- 400 Mbps from 0 to 5 seconds
- 200 Mbps from 5 to 10 seconds
- 550 Mbps for the remaining duration


### Adjusting DASH Parameters:
You can modify DASH streaming parameters such as buffer duration, minimum buffer level, smoothing factor, and initial startup bandwidth in the [config file](/istream-player/istream_player/config/config.py).

### Player Documentation:
To view player options, use the following entry point (commented out in the compose file):
```yml
entrypoint: ["iplay", "-h"]
```

### Build and Run:
Once all configurations are set according to your experiment, build the three modules with:

```bash
# Build all modules
docker compose build
```

Then run your experiment:

```bash
# run an experiment
bash run.sh
```

