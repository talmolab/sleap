import subprocess
import os

def encode_video(input_file, output_file):
    """Encode a video using ffmpeg with specified parameters."""
    command = [
        'ffmpeg', '-y',
        '-i', input_file,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'superfast',
        '-crf', '23',
        output_file
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Encoding completed: {output_file}")
    except subprocess.CalledProcessError:
        print(f"An error occurred during encoding: {input_file}")

def encode_directory(directory):
    """Encodes all .mp4 files in the specified directory."""
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            input_file = os.path.join(directory, filename)
            output_file = os.path.join(directory, os.path.splitext(filename)[0] + '_encoded.mp4')
            print(f"Starting encoding: {input_file}")
            encode_video(input_file, output_file)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python video_encoder.py <directory>")
    else:
        directory = sys.argv[1]
        if os.path.isdir(directory):
            encode_directory(directory)
        else:
            print("The provided directory does not exist.")
