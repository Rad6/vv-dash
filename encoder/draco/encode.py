import os
import subprocess
import json
from multiprocessing import Process

def run_command(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))  # Print the output directly

def create_command(rate_id, params, frame, start_frame, video_path, frame_naming_template, output_folder):
    cl, qp, qg = params
    input_file = os.path.join(video_path, frame_naming_template.format(frame_number=frame))
    output_file = os.path.join(output_folder, f"stream{rate_id}-frame{frame - start_frame + 1}.drc")
    
    cmd = ['draco_encoder', '-point_cloud', '-i', input_file,
           '-cl', str(cl), '-qp', str(qp), '-qt', str(qg), '-o', output_file]
    return cmd

def main():
    with open("config.json", "r", encoding="utf-8") as file:
        config = json.load(file)
    
    encoding_parameters = config["encoding_parameters"]
    start_frame = config["start_frame"]
    frame_count = config["frame_count"]
    video_path = config["video_path"]
    frame_naming_template = config["frame_naming_template"]
    output_folder = config["output_folder"]
    
    os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists
    
    processes = []

    # Prepare and start all the processes
    for rate_id, params in encoding_parameters.items():
        for frame in range(start_frame, start_frame + frame_count):
            cmd = create_command(rate_id, params, frame, start_frame, video_path, frame_naming_template, output_folder)
            p = Process(target=run_command, args=(cmd,))
            p.start()
            processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
