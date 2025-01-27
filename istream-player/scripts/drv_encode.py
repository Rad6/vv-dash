import argparse
import json
import sys

def main():
    parser = argparse.ArgumentParser(
                    prog='DRV Codec',
                    description='Draco Video Encoder')
    parser.add_argument('frames', nargs='+')
    parser.add_argument('-f', '--fps', type=int)

    args = parser.parse_args()
    print(args)


    frame_info = []
    payload = bytes()
    timescale = args.fps
    for frame_idx, frame_path in enumerate(args.frames):
        with open(frame_path, "rb") as f:
            frame_content = f.read()
            frame_size = len(frame_content)
            frame_info.append({
                "offset": len(payload),
                "size": frame_size,
                "pts": frame_idx
            })
            
            print(f"Frame size: {frame_size.to_bytes(4, 'big')}")
            # payload += frame_size.to_bytes(4, 'big')
            payload += frame_content


    sys.stdout.buffer.write(bytes([1, 0, 0]))

    # write Header
    header = json.dumps({
        "timescale": timescale,
        "frames": frame_info
    }).encode("ascii")
    sys.stdout.buffer.write("JSON".encode("ascii"))
    sys.stdout.buffer.write(len(header).to_bytes(4, "big"))
    sys.stdout.buffer.write(header)



    # Write frames payload
    sys.stdout.buffer.write(payload)
    
    sys.stdout.buffer.flush()
    sys.stdout.flush()



if __name__ == "__main__":
    main()