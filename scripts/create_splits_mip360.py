#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2024.

import json
import logging
import argparse
import os
from pathlib import Path
import glob

_logger = logging.getLogger(__name__)


if __name__ == '__main__':

    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Creat benchmarking train/test split files for MIP-NeRF 360. '
                    'Takes every 8th frames starting from 0.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dataset_root', type=Path, help="Root folder of the dataset.")

    parser.add_argument('--subfolder', type=str, default='images_4',
                        help="Subfolder containing the images for each scene.")

    parser.add_argument('--test_step', type=int, default=8, help="Select 1 test image every N images")

    parser.add_argument('output_folder', type=Path, help="Where to store the split files.")

    args = parser.parse_args()

    # Create the output folder if it does not exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Scenes of 7Scenes as sub folders of the dataset root
    scene_folders = [f for f in args.dataset_root.glob('*/') if f.is_dir()]

    # Process each scene
    for scene_folder in scene_folders:

        _logger.info(f"Processing scene {scene_folder.name}.")

        # Get the image files
        scene_image_folder = scene_folder / args.subfolder
        image_files = sorted(list(glob.glob(f"{scene_image_folder}/*.JPG")))

        # Split by taking every n th items for test, and everything else for train
        test_files = image_files[args.test_step-1::args.test_step]
        train_files = [f for f in image_files if f not in test_files]

        _logger.info(f"Found {len(test_files)} test files and {len(train_files)} train files.")

        # Create the split info
        split_info = {
            'train_filenames': train_files,
            'test_filenames': test_files
        }

        # Store the split info in a JSON file
        _logger.info(f"Writing split info to {args.output_folder / f'{scene_folder.name}.json'}")

        with open(args.output_folder / f"mip360_{scene_folder.name}.json", 'w') as f:
            json.dump(split_info, f)
