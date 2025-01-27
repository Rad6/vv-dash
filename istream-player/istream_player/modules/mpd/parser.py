from copy import deepcopy
import logging
import os
import re
from abc import ABC, abstractmethod
from math import ceil
from typing import Dict, Optional
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

from istream_player.models.mpd_objects import MPD, AdaptationSet, Representation, Segment


class MPDParsingException(BaseException):
    pass


class MPDParser(ABC):
    @abstractmethod
    def parse(self, content: str, url: str) -> MPD:
        pass


class DefaultMPDParser(MPDParser):
    log = logging.getLogger("DefaultMPDParser")
    
    def __init__(self, stop_time: Optional[float]) -> None:
        super().__init__()
        self.stop_time = stop_time if stop_time is not None else float('inf')

    @staticmethod
    def parse_iso8601_time(duration: Optional[str]) -> float:
        """
        Parse the ISO8601 time string to the number of seconds
        """
        if duration is None or duration == "":
            return 0
        pattern = r"^PT(?:(\d+(?:.\d+)?)H)?(?:(\d+(?:.\d+)?)M)?(?:(\d+(?:.\d+)?)S)?$"
        results = re.match(pattern, duration)
        if results is not None:
            dur = [float(i) if i is not None else 0 for i in results.group(1, 2, 3)]
            dur = 3600 * dur[0] + 60 * dur[1] + dur[2]
            return dur
        else:
            return 0

    @staticmethod
    def remove_namespace_from_content(content):
        """
        Remove the namespace string from XML string
        """
        content = re.sub('xmlns="[^"]+"', "", content, count=1)
        return content

    def parse(self, content: str, url: str) -> MPD:
        content = self.remove_namespace_from_content(content)
        root = ElementTree.fromstring(content)

        type_ = root.attrib["type"]
        assert type_ == "static" or type_ == "dynamic"

        # media presentation duration
        media_presentation_duration = self.parse_iso8601_time(root.attrib.get("mediaPresentationDuration", ""))
        self.log.info(f"{media_presentation_duration=}")

        # min buffer duration
        min_buffer_time = self.parse_iso8601_time(root.attrib.get("minBufferTime", ""))
        self.log.info(f"{min_buffer_time=}")

        # max segment duration
        max_segment_duration = self.parse_iso8601_time(root.attrib.get("maxSegmentDuration", ""))
        self.log.info(f"{max_segment_duration=}")

        period = root.find("Period")

        if period is None:
            raise MPDParsingException('Cannot find "Period" tag')

        adaptation_sets: Dict[int, AdaptationSet] = {}

        base_url = os.path.dirname(url) + "/"

        for index, adaptation_set_xml in enumerate(period):
            adaptation_set: AdaptationSet = self.parse_adaptation_set(
                adaptation_set_xml, base_url, index, media_presentation_duration
            )
            adaptation_sets[adaptation_set.id] = adaptation_set

        return MPD(
            content, url, type_, media_presentation_duration, max_segment_duration, min_buffer_time, adaptation_sets, root.attrib
        )

    def parse_adaptation_set(
        self, tree: Element, base_url, index: Optional[int], media_presentation_duration: float
    ) -> AdaptationSet:
        id_ = int(tree.attrib.get("id", str(index)))
        content_type = tree.attrib.get("contentType") or tree.find("Representation").attrib.get("mimeType") or "unknown"
        content_type = content_type.lower().split("/")[0]
        assert (
            content_type == "video" or content_type == "audio"
        ), f"Only 'video' or 'audio' content_type is supported, Got {content_type}"
        
        frame_rate = tree.attrib.get("frameRate")
        max_width = int(tree.attrib.get("maxWidth", 0))
        max_height = int(tree.attrib.get("maxHeight", 0))
        par = tree.attrib.get("par")
        self.log.debug(f"{frame_rate=}, {max_width=}, {max_height=}, {par=}")

        adap_set = AdaptationSet(int(id_), content_type, frame_rate, max_width, max_height, par, {}, None, tree.attrib)
        representations = adap_set.representations
        # GPAC MPD has segment template inside adaptation set
        segment_template: Optional[Element] = tree.find("SegmentTemplate")
        
        for srd_suppl_prop in tree.findall("./SupplementalProperty[@schemeIdUri = 'urn:mpeg:dash:srd:2014']"):
            adap_set.srd = tuple(map(int, srd_suppl_prop.attrib["value"].split(",")))
            assert len(adap_set.srd) == 7, "SupplementalProperty 'urn:mpeg:dash:srd:2014' should have 7 comma separated integers"
        
        for representation_tree in tree.findall("Representation"):
            representation = self.parse_representation(
                representation_tree, adap_set, base_url, segment_template, media_presentation_duration
            )
            representations[representation.id] = representation

        reprs_list = sorted(representations.values(), key=lambda r: r.bandwidth)
        for quality, repr in enumerate(reprs_list):
            for seg in repr.segments.values():
                seg.quality = quality

        return adap_set

    def parse_representation(
        self,
        tree: Element,
        adap_set: AdaptationSet,
        base_url,
        segment_template: Optional[Element],
        media_presentation_duration: float,
    ) -> Representation:
        inner_segment_template = tree.find("SegmentTemplate")
        if inner_segment_template is not None:
            if segment_template is None:
                segment_template = inner_segment_template
            else:
                segment_template = deepcopy(segment_template)
                segment_template.attrib.update(inner_segment_template.attrib)
        if segment_template is None:
            raise MPDParsingException("The MPD support is not complete yet")
        return self.parse_representation_with_segment_template(
            tree, adap_set, base_url, segment_template, media_presentation_duration
        )

    def parse_representation_with_segment_template(
        self, tree: Element, adap_set: AdaptationSet, base_url, segment_template: Element, media_presentation_duration: float
    ) -> Representation:
        id_ = tree.attrib["id"]
        mime = tree.attrib["mimeType"]
        codec = tree.attrib["codecs"]
        bandwidth = int(tree.attrib["bandwidth"])
        is_video = mime.lower().startswith("video")
        if is_video:
            width, height = int(tree.attrib["width"]), int(tree.attrib["height"])
        else:
            width, height = 0, 0

        assert segment_template is not None, "Segment Template not found in representation"

        initialization = segment_template.attrib["initialization"]
        initialization = self.var_repl(initialization, {"RepresentationID": id_})
        initialization = base_url + initialization
        self.log.info(f"{id_=}, {segment_template.attrib['initialization']=}, {initialization=}")
        segments: Dict[int, Segment] = {}

        timescale = int(segment_template.attrib["timescale"])
        media = self.var_repl(segment_template.attrib["media"], {"RepresentationID": id_})
        start_number = int(segment_template.attrib["startNumber"])
        
        frame_rate = "1"
        if is_video:
            frame_rate = tree.attrib.get("frameRate") or adap_set.frame_rate
            assert frame_rate is not None, "Frame rate not found"

        segment_timeline = segment_template.find("SegmentTimeline")
        if segment_timeline is not None:
            num = start_number
            start_time = 0
            for segment in segment_timeline:
                duration = int(segment.attrib["d"]) / timescale
                url = base_url + self.var_repl(media, {"Number": num})
                if "t" in segment.attrib:
                    start_time = float(segment.attrib["t"]) / timescale
                
                fr_num, fr_den = self.parse_frame_rate(frame_rate)
                num_frames = int((int(segment.attrib["d"]) * fr_num) / (timescale * fr_den))
                if "st" in segment_timeline.attrib:
                    all_st = segment_timeline.attrib["st"]
                    # print(f"{num=}, {all_st=}, {(num-start_number) % len(all_st)=}")
                    if num == start_number:
                        segment.attrib["st"] = 'i'
                    else:
                        segment.attrib["st"] = all_st[(num-start_number) % len(all_st)]
                segments[num] = Segment(
                    num,
                    url,
                    initialization,
                    duration,
                    start_time,
                    adap_set.id,
                    int(id_),
                    0,
                    num_frames,
                    bandwidth,
                    segment.attrib,
                )
                num += 1
                start_time += duration

                if "r" in segment.attrib:  # repeat
                    for _ in range(int(segment.attrib["r"])):
                        if start_time >= self.stop_time:
                            break
                        url = base_url + self.var_repl(media, {"Number": num})
                        segattrib = {**segment.attrib}
                        if "st" in segment_timeline.attrib:
                            all_st = segment_timeline.attrib["st"]
                            segattrib["st"] = all_st[(num-start_number) % len(all_st)]
                        segments[num] = Segment(
                            num,
                            url,
                            initialization,
                            duration,
                            start_time,
                            adap_set.id,
                            int(id_),
                            0,
                            num_frames,
                            bandwidth,
                            segattrib,
                        )
                        num += 1
                        start_time += duration
                
                if start_time >= self.stop_time:
                    break
        else:
            # GPAC DASH format
            num = start_number
            start_time = 0
            num_segments = ceil((int(media_presentation_duration) * timescale) / int(segment_template.attrib["duration"]))
            duration = int(segment_template.attrib["duration"]) / timescale
            fr_num, fr_den = self.parse_frame_rate(frame_rate)
            num_frames = int((int(segment_template.attrib["duration"]) * fr_num) / (timescale * fr_den))
            self.log.debug(f"{num_segments=}, {duration=}")
            for _ in range(num_segments):
                url = base_url + self.var_repl(media, {"Number": num})
                # self.log.info(f"{media=}, {url=}", {"Number": num})
                segments[num] = Segment(
                    num,
                    url,
                    initialization,
                    duration,
                    start_time,
                    adap_set.id,
                    int(id_),
                    0,
                    num_frames,
                    bandwidth,
                    segment_template.attrib,
                )
                num += 1
                start_time += duration
                if start_time >= self.stop_time:
                    break
            # self.log.debug(segments)

        return Representation(int(id_), mime, codec, bandwidth, width, height, initialization, segments, tree.attrib)

    @staticmethod
    def var_repl(s: str, vars: Dict[str, int | str]):
        def _repl(m) -> str:
            m = m.group()[1:-1]
            if m in vars:
                return str(vars[m])
            elif "%" in m:
                v, p = m.split("%", 1)
                if v in vars:
                    return f"%{p}" % vars[v]
            return f"${m}$"

        return re.sub(r"\$[^\$]*\$", _repl, s)

    @staticmethod
    def parse_frame_rate(frame_rate: str):
        if "/" in frame_rate:
            num, den = frame_rate.split("/")
            return int(num), int(den)
        else:
            return int(frame_rate), 1
