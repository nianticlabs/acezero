#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2024.

import json
import logging
import argparse
import os
from pathlib import Path
import glob

_logger = logging.getLogger(__name__)


def read_split_file(split_file: Path):
    """
    Reads a split file and converts each line into a dataset sequence folder name.

    :param split_file: A Path object pointing to the split file to be read.
    :return: A list of formatted sequence folder names derived from the split file.
    """

    # read the split file
    with open(split_file, 'r') as f:
        data = f.readlines()

    # convert the file entry to a dataset sequence folder
    seq_folders = [f"seq-{int(seq_id[8:]):02d}" for seq_id in data]

    return seq_folders


def process_split(split_file: Path, scene_folder: Path):
    """
    Processes a given split file to find and list all color image files within the specified scene folder.

    :param split_file: A Path object pointing to the split file. This file contains lines that are used
                       to construct the names of sequence folders.
    :param scene_folder: A Path object pointing to the scene folder where the sequence folders are located.
    :return: A list of paths to color image files found within the sequence folders specified in the split file.
    """

    # Read the split file and get a list of sequence folder names
    seq_folders = read_split_file(scene_folder / split_file)

    # Initialize an empty list to store paths to the color image files
    seq_files = []
    # Iterate over each sequence folder
    for seq_folder in seq_folders:
        # Use glob to find all color image files in the sequence folder and add them to the list
        seq_files += glob.glob(f"{scene_folder / seq_folder}/*.color.png")

    # Return the list of color image file paths
    return seq_files


if __name__ == '__main__':

    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Creat benchmarking train/test split files for 7Scenes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dataset_root', type=Path, help="Root folder of the dataset.")

    parser.add_argument('output_folder', type=Path, help="Where to store the split files.")

    args = parser.parse_args()

    # Create the output folder if it does not exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Scenes of 7Scenes as sub folders of the dataset root
    scene_folders = [f for f in args.dataset_root.glob('*/') if f.is_dir()]

    # Process each scene
    for scene_folder in scene_folders:

        _logger.info(f"Processing scene {scene_folder.name}.")

        # Read the split file
        test_files = process_split(Path('TestSplit.txt'), scene_folder)
        train_files = process_split(Path('TrainSplit.txt'), scene_folder)

        _logger.info(f"Found {len(test_files)} test files and {len(train_files)} train files.")

        # Create the split info
        split_info = {
            'train_filenames': train_files,
            'test_filenames': test_files
        }

        # Store the split info in a JSON file
        _logger.info(f"Writing split info to {args.output_folder / f'{scene_folder.name}.json'}")

        with open(args.output_folder / f"7scenes_{scene_folder.name}.json", 'w') as f:
            json.dump(split_info, f)
