import xml.etree.ElementTree as ET
import json

def generate_mpd(
    frameRate,
    mediaPresentationDuration, 
    maxSegmentDuration, 
    initialization_template, 
    media_template, 
    num_segments, 
    bitrates, 
    segment_timescale, 
    segment_duration,
    output_file="output.mpd"
):
    with open("mpd_template.xml", "r", encoding="utf-8") as template_file:
        template = template_file.read()
    
    representations = ""
    for i, bitrate in enumerate(bitrates, start=1):
        representations += f'''
        <Representation id="{i}" mimeType="video" codecs="DRCO" bandwidth="{bitrate}" width="512" height="512" sar="1:1">
            <SegmentTemplate timescale="{segment_timescale}" initialization="{initialization_template}" media="{media_template}" startNumber="1">
                <SegmentTimeline>
                    <S t="0" d="{segment_duration}" r="{num_segments - 1}" />
                </SegmentTimeline>
            </SegmentTemplate>
        </Representation>'''
    
    mpd_content = template.format(
        frameRate=frameRate,
        mediaPresentationDuration=mediaPresentationDuration,
        maxSegmentDuration=maxSegmentDuration,
        representations=representations
    )
    
    with open(output_file, "w", encoding="utf-8") as output:
        output.write(mpd_content)
    print(f"MPD file '{output_file}' generated successfully.")

with open("config.json", "r", encoding="utf-8") as file:
        config = json.load(file)

generate_mpd(
        frameRate=config["frameRate"],
        mediaPresentationDuration=config["mediaPresentationDuration"],
        maxSegmentDuration=config["maxSegmentDuration"],
        initialization_template=config["initialization_template"],
        media_template=config["media_template"],
        num_segments=config["num_segments"],
        bitrates=config["bitrates"],
        segment_timescale=config["segment_timescale"],
        segment_duration=config["segment_duration"],
        output_file=config["output_file"]
    )
