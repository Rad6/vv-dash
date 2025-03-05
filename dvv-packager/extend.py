import os
import shutil
import re

REPEATS = 1 #how many times to repeat the video with itself

def extend_video_chunks(directory):
    # Define the regex pattern to match the files
    pattern = re.compile(r'(chunk-stream\d+)-(\d{5})\.dvv')
    
    # List all files in the directory
    files = [f for f in os.listdir(directory) if pattern.match(f)]
    
    # Sort files to ensure proper ordering
    files.sort()
    
    for file in files:
        match = pattern.match(file)
        if match:
            base_name = match.group(1)
            chunk_num = int(match.group(2))
            
            # We need to create two more copies for each chunk
            for i in range(1, REPEATS + 1):  # Create chunk 11 to 20
                new_chunk_num = chunk_num + (i * 10)
                new_file_name = f"{base_name}-{new_chunk_num:05d}.dvv"
                
                # Copy the file
                shutil.copy(os.path.join(directory, file), os.path.join(directory, new_file_name))
                print(f"Copied {file} to {new_file_name}")

# Example usage
directory = "."  # Change this to your actual file directory
extend_video_chunks(directory)
