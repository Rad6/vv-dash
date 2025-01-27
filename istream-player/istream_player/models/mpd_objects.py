from dataclasses import dataclass
from typing import Dict, Literal, Optional


class MPD(object):
    def __init__(
        self,
        content: str,
        url: str,
        type_: Literal["static", "dynamic"],
        media_presentation_duration: float,
        max_segment_duration: float,
        min_buffer_time: float,
        adaptation_sets: Dict[int, "AdaptationSet"],
        attrib: Dict[str, str],
    ):
        self.content = content
        """
        The raw content of the MPD file
        """

        self.url = url
        """
        The URL of the MPD file
        """

        self.type: Literal["static", "dynamic"] = type_
        """
        If this source is VOD ("static") or Live ("dynamic")
        """

        self.media_presentation_duration = media_presentation_duration
        """
        The media presentation duration in seconds
        """

        self.min_buffer_time = min_buffer_time
        """
        The recommended minimum buffer time in seconds
        """

        self.max_segment_duration = max_segment_duration
        """
        The maximum segment duration in seconds
        """

        self.adaptation_sets: Dict[int, AdaptationSet] = adaptation_sets
        """
        All the adaptation sets
        """

        self.attrib = attrib
        """
        All attributes from XML
        """


@dataclass
class AdaptationSet(object):
    # The adaptation set id
    id: int
    # The content type of the adaptation set. It could only be "video" or "audio"
    content_type: Literal["video", "audio"]
    # The frame rate string
    frame_rate: Optional[str]
    # The maximum width
    max_width: int
    # The maximum height
    max_height: int
    # The ratio of width / height
    par: str
    # All the representations under the adaptation set
    representations: Dict[int, "Representation"]
    # Spatial Relation Description (source, position_rw)
    # source_id, col, row, tile_cols, tile_rows, grid_cols, grid_rows
    srd: Optional[tuple[int, int, int, int, int, int, int]]
    # All attributes from XML
    attrib: Dict[str, str]


class Representation(object):
    def __init__(
        self,
        id_: int,
        mime_type: str,
        codecs: str,
        bandwidth: int,
        width: int,
        height: int,
        initialization: Optional[str],
        segments: Dict[int, "Segment"],
        attrib: Dict[str, str],
    ):
        self.id = id_
        """
        The id of the representation
        """

        self.mime_type = mime_type
        """
        The mime type of the representation
        """

        self.codecs: str = codecs
        """
        The codec string of the representation
        """

        self.bandwidth: int = bandwidth
        """
        Average bitrate of this stream in bps
        """

        self.width = width
        """
        Width of picture
        """

        self.height = height
        """
        Height of picture
        """

        self.initialization: Optional[str] = initialization
        """
        The initialization URL
        """

        self.segments: Dict[int, Segment] = segments
        """
        The video segments
        """

        self.attrib = attrib
        """
        All attributes from XML
        """


@dataclass
class Segment(object):
    # index
    index: int
    
    # Segment URL
    url: str

    # Stream Initilalization URL
    init_url: Optional[str]

    # Segment play duration in seconds
    duration: float

    # Segment play start time
    start_time: float

    # Adaptation Set ID
    as_id: int

    # Representation ID
    repr_id: int

    # Quality (0: best quality ...)
    quality: int

    # Number of frames
    num_frames: int

    # repr_bitrate
    bitrate: int

    # extra args
    attrib: Dict[str, str]
