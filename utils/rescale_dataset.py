"""
    Rescaling video dataset in a directory to a given one using ffmpeg (must be installed).

    Usage: $python rescale_dataset.py source/dataset/directory/ target/dataset/directory/ width height
"""

import os
import sys
import subprocess
from time import sleep
from IPython.display import clear_output


def rescale_video(src_video: str,
                  dst_video: str,
                  target_width: int,
                  target_height: int) -> None:
    ffmpeg_cmd = (f"ffmpeg "
                  f"-i {src_video} "
                  f"-vf scale={target_width}:{target_height} "
                  f"-c:v libx264 "
                  f"{dst_video}")
    sleep(0.5)
    clear_output(wait=True)
    print(ffmpeg_cmd)
    subprocess.call(ffmpeg_cmd, shell=True)


def rescale_dataset(input_path: str,
                    output_path: str,
                    target_width: int,
                    target_height: int) -> None:
    for video_name in os.listdir(input_path):
        rescale_video(
            os.path.join(input_path, video_name),
            os.path.join(output_path, video_name),
            target_width,
            target_height
        )


if __name__ == "__main__":
    src_dataset_directory = sys.argv[1]
    dst_dataset_directory = sys.argv[2]
    width = int(sys.argv[3])
    height = int(sys.argv[4])

    rescale_dataset(src_dataset_directory, dst_dataset_directory, width, height)
