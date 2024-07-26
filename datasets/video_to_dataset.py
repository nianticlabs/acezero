#! /usr/bin/env python3

import os
import glob
import argparse
import shutil
import subprocess
from pathlib import Path


def find_mp4_files(directory):
    # Use os.path.join to ensure the path is constructed correctly for the OS
    search_path = os.path.join(directory, '*.mp4')

    # Use glob.glob to find all .mp4 files
    mp4_files = glob.glob(search_path)

    return mp4_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find all .mp4 files in a directory and convert them to a dataset.')
    parser.add_argument('directory', type=str, help='The directory to search')
    parser.add_argument('--min_side_length', type=int, default=540,
                        help='The minimum side length of the output frames.')

    args = parser.parse_args()

    # get ffmpeg path
    ffmpeg_path = shutil.which("ffmpeg")

    # get list of all .mp4 files in the directory
    mp4_files = find_mp4_files(args.directory)

    # iterate over all .mp4 files
    for mp4_file in mp4_files:

        # check whether a folder with the same name exists
        folder_name = Path(mp4_file).stem
        folder_path = Path(args.directory) / ("video_" + folder_name)

        if not folder_path.exists():
            print(f"Creating folder: {folder_path}")
            folder_path.mkdir(parents=True)

            print(f"Extracting frames from: {mp4_file}")
            subprocess.run(
                [
                    ffmpeg_path,
                    "-i", mp4_file,
                    "-vf",
                    f"scale=w='if(lte(iw,ih),{args.min_side_length},-1)':h='if(lte(iw,ih),-1,{args.min_side_length})'",
                    "-qmin", "1",
                    "-q:v", "1",
                    f"{folder_path}/%06d.jpg"
                ],
                check=True,
            )
        else:
            print(f"Folder already exists: {folder_path}. Skip.")
