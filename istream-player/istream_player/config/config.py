from dataclasses import dataclass, field
from typing import Optional


class StaticConfig(object):
    # Max initial bitrate (bps)
    max_initial_bitrate = 400000000

    # averageSpeed = SMOOTHING_FACTOR * lastSpeed + (1-SMOOTHING_FACTOR) * averageSpeed;
    smoothing_factor = 1

    # minimum frame chunk size ratio
    # The size ratio of a segment which is for I-, P-, and B-frames.
    min_frame_chunk_ratio = 0.6

    # VQ threshold
    vq_threshold = 0.8

    # [Not Used] VQ threshold for size ratio
    vq_threshold_size_ratio = min_frame_chunk_ratio * (min_frame_chunk_ratio + (1 - min_frame_chunk_ratio) * vq_threshold)

    # Update interval
    update_interval = 0.05

    # [Not Used] Chunk size
    chunk_size = 40960

    # [Not Used] Timeout max ratio
    timeout_max_ratio = 2

    # [Not Used] Min Duration for quality increase (ms)
    min_duration_for_quality_increase_ms = 6000

    # [Not Used] Max duration for quality decrease (ms)
    max_duration_for_quality_decrease_ms = 8000

    # [Not Used] Min duration to retrain after discard (ms)
    min_duration_to_retrain_after_discard_ms = 8000

    # [Not Used] Bandwidth fraction
    bandwidth_fraction = 0.75

    # If the packet arrives later than this it should not be consider in bw estimation
    max_packet_delay = 2

    # Continuous bw estimation window (s)
    cont_bw_window = 1


@dataclass
class PlayerConfig:
    # TODO: Move static configurations to dynamic
    static = StaticConfig

    log_level: str = "info"

    # Required config
    input: str = ""

    time_factor: float = 1
    stop_time: Optional[float] = None

    # Modules
    mod_mpd: str = "mpd"
    mod_downloader: str = "auto"
    mod_bw: str = "bw_meter"
    mod_abr: str = "dash"
    mod_scheduler: str = "scheduler"
    mod_buffer: str = "buffer_manager"
    mod_player: str = "dash_player"
    mod_analyzer: list[str] = field(default_factory=lambda: [])

    # Buffer Configuration
    buffer_duration: float = 4
    safe_buffer_level: float = 2
    panic_buffer_level: float = 1
    min_rebuffer_duration: float = 1.5
    min_start_duration: float = 2


    # Buffer Configuration
    # buffer_duration: float = 90
    # safe_buffer_level: float = 60
    # panic_buffer_level: float = 10
    # min_rebuffer_duration: float = 10
    # min_start_duration: float = 10

    run_dir: Optional[str] = None
    plots_dir: Optional[str] = None
    ssl_keylog_file: Optional[str] = None

    # Live event logs file path
    live_log: Optional[str] = None

    # Network Config
    bw_profile: Optional[str] = None

    ## For 360-sonali-video1-gaslamp-crf18
    # resolution_ladder: list[int] = field(default_factory=lambda: [80, 120, 192, 240, 360]) #tile resolutions[width](least to highest order)
    # bitrate_ladder: list[float] = field(default_factory=lambda: [27, 46.22, 86.66, 120.88, 215.17]) #bps in CRF=18 (not sure) 


    # 360 Video Config 
    fov_groups_number: int = 3
    tile_number: int = 9
    fov_prediction_mode: bool = False
    ## For akramURl-srd-360
    # resolution_ladder: list[int] = field(default_factory=lambda: [240, 360]) #tile resolutions(least to highest order) not changed in here
    # bitrate_ladder: list[float] = field(default_factory=lambda: [128000, 1500000]) #bps 
    ## For 360-sonali-video1-gaslamp-crf18
    resolution_ladder: list[int] = field(default_factory=lambda: [80, 120, 192, 240, 360]) #tile resolutions[height](least to highest order)
    bitrate_ladder: list[float] = field(default_factory=lambda: [143000000, 198000000, 256000000, 348000000, 453000000]) #bps in CRF=18 (not sure and is average) --- Max: [96000, 157000, 263000, 503000, 1024000] #total lowest quality = 442010 bps
    bw_penalty_factor: float = 1
    bw_limit: float = 400000000 #bps
    bw_error_tolerance: float = 1.25 #should be set based on the bw_limit and tc settings
    bw_error_offset: float = 1.1 #should be set based on the bw_limit and tc settings
    resolution_threshold: list[int] = field(default_factory=lambda: [0, 1, -1]) #index of resolution ladder sorted (from highest to lowest)
    priority_weight: list[float] = field(default_factory=lambda: [1, 1, 1]) #0.757, 0.198, 0.02 #priority weight for each tile zone group (it can be set using GA aglorithm)
    priority_weight_tbra: list[float] = field(default_factory=lambda: [0.757, 0.198, 0.02]) 
    weight_obj_func: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.4]) #weight for objective function TBRA Algorithm (has been set according to the Mosaic/TBRA paper)
    #prediction
    weight_prediction: list[float] = field(default_factory=lambda: [0.7, 0.8, 0.4]) #[alpha, beta, gamma] for accurcy less than 90 we used 0.5, 0.8, 0.4/ over than 90 we used 0.7, 0.8, 0.4
    prediction_model_parameters: list[float] = field(default_factory=lambda: [0.80, 1]) #[pred model accuracy, pred model window time]
    prediction_weight_rate_controller: list[float] = field(default_factory=lambda: [0.8, 20, 0.5]) #[pred_thereshold, pred_alpha, time_beta]

    solver_mode = "gurobi" # gurobi or pyomo
    # Gurobi Solver Config
    pool_search_mode = 2 # 0: no pool search, 1: no guarantess on optimality, 2: find the best n solutions
    pool_solutions = 20 # number of solutions to find
    pool_gap = 0.02 # gap to optimality
    output_flag = 0 # 0: no output details, 1: output details

    tmc2_decoder_path = "./tmc2-rs-main"


    def validate(self) -> None:
        """Assert if config properties are set properly"""
        assert bool(self.input), "A non-empty '--input' arg or 'input' config is required"

