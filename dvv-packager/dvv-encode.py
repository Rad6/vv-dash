import json

def main():
    with open("config.json", "r", encoding="utf-8") as file:
        config = json.load(file)
    
    start = config["start_frame"]
    end = config["end_frame"]
    fps = config["fps"]
    frame_path_template = config["frame_path_template"]
    output_path_template = config["output_path_template"]
    
    frame_info = []
    payload = bytes()
    timescale = fps
    
    for rate in range(1, 6):
        for frame in range(start, end + 1):
            frame_path = frame_path_template.format(rate=rate, frame=frame)
            with open(frame_path, "rb") as f:
                frame_content = f.read()
                frame_size = len(frame_content)
                frame_info.append({
                    "offset": len(payload),
                    "size": frame_size,
                    "pts": frame
                })
                payload += frame_content
            
            if (frame % fps == 0):
                chunk_number = int(frame / fps)
                output_path = output_path_template.format(rate=rate, chunk_number=f"%05d" % chunk_number)
                with open(output_path, "wb") as f:
                    f.write(bytes([1, 0, 0]))
                    header = json.dumps({
                        "timescale": timescale,
                        "frames": frame_info
                    }).encode("ascii")
                    f.write("JSON".encode("ascii"))
                    f.write(len(header).to_bytes(4, "big"))
                    f.write(header)
                    f.write(payload)
                    f.flush()
                
                frame_info = []
                payload = bytes()

if __name__ == "__main__":
    main()
