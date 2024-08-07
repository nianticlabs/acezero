#!/usr/bin/env python3

import argparse
import os
import subprocess
from typing import List, Optional
import numpy as np

def run_command(command: str) -> None:
    """
    Execute a shell command and raise an exception if it fails.

    Args:
        command (str): The shell command to execute.

    Raises:
        subprocess.CalledProcessError: If the command execution fails.
    """
    subprocess.run(command, shell=True, check=True)

def calculate_average_focal_length(root_dir: str) -> Optional[float]:
    """
    Calculate the average focal length from the cameras.txt file.

    Args:
        root_dir (str): The root directory containing the COLMAP sparse reconstruction.

    Returns:
        Optional[float]: The average focal length, or None if no valid focal lengths are found.
    """
    intrinsics_dir = os.path.join(root_dir, 'colmap', 'arkit', '0', 'cameras.txt')
    focal_lengths: List[float] = []
    
    try:
        with open(intrinsics_dir, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        focal_lengths.append(float(parts[4]))
    except FileNotFoundError:
        print(f"Warning: cameras.txt file not found at {intrinsics_dir}")
        return None
    except ValueError as e:
        print(f"Warning: Error parsing focal length: {e}")
        return None

    return np.mean(focal_lengths) if focal_lengths else None

def check_data_integrity(root_dir: str) -> None:
    """
    Check if necessary files and directories exist in the dataset.

    Args:
        root_dir (str): The root directory of the dataset.

    Raises:
        AssertionError: If any required file or directory is missing.
    """
    assert os.path.isdir(root_dir), f"Root directory {root_dir} does not exist"
    assert os.path.isdir(os.path.join(root_dir, 'colmap', 'arkit', '0')), f"COLMAP sparse reconstruction directory does not exist: {os.path.join(root_dir, 'colmap', 'arkit', '0')}"
    assert os.path.isfile(os.path.join(root_dir, 'colmap', 'arkit', '0', 'cameras.txt')), f"cameras.txt file does not exist"
    assert os.path.isfile(os.path.join(root_dir, 'colmap', 'arkit', '0', 'images.txt')), f"images.txt file does not exist"
    assert os.path.isdir(os.path.join(root_dir, 'images')), f"Images directory does not exist: {os.path.join(root_dir, 'images')}"
    
    try:
        with open(os.path.join(root_dir, 'colmap', 'arkit', '0', 'images.txt'), 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 10:
                        image_name = parts[9]
                        assert os.path.isfile(os.path.join(root_dir, 'images', image_name)), f"Image file does not exist: {image_name}"
    except FileNotFoundError:
        raise AssertionError(f"images.txt file not found or cannot be read")

def main() -> None:
    """
    Main function to run the ACE0 Pipeline for Home Environment Reconstruction.
    """
    parser = argparse.ArgumentParser(description="ACE0 Pipeline for Home Environment Reconstruction")
    parser.add_argument("--root_dir", default="data/tw_home", help="Original data root directory")
    parser.add_argument("--refine_dir", help="Refined data directory")
    args = parser.parse_args()

    root_dir = args.root_dir
    refine_dir = args.refine_dir or os.path.join(root_dir, "colmap", "ace_refine")
    intermediate_dir = f"{root_dir}_intermediate"

    try:
        check_data_integrity(root_dir)
    except AssertionError as e:
        print(f"Data integrity check failed: {e}")
        return

    focal_length = calculate_average_focal_length(root_dir)
    if focal_length is None:
        print("Unable to calculate average focal length, please check the dataset")
        return

    print(f"Calculated average focal length: {focal_length}")

    # Step 1: Set up COLMAP dataset
    run_command(f"python datasets/setup_colmap.py --root {root_dir} --output_dir {intermediate_dir} --images_folder images --all_train")

    # Create results directory in intermediate_dir
    os.makedirs(os.path.join(intermediate_dir, "results"), exist_ok=True)

    # Step 2: Train ACE network
    run_command(f"python train_ace.py \"{intermediate_dir}/train/rgb/*.jpg\" {intermediate_dir}/results/ace_network.pt "
                f"--pose_files \"{intermediate_dir}/train/poses/*.txt\" "
                f"--pose_refinement mlp "
                f"--pose_refinement_wait 5000 "
                f"--use_external_focal_length {focal_length} "
                f"--refine_calibration False")

    # Step 3: Register mapping
    run_command(f"python register_mapping.py \"{intermediate_dir}/train/rgb/*.jpg\" {intermediate_dir}/results/ace_network.pt "
                f"--use_external_focal_length {focal_length} "
                f"--session ace_network")

    # Step 4: Export point cloud
    run_command(f"python export_point_cloud.py {intermediate_dir}/results/point_cloud_out.txt "
                f"--network {intermediate_dir}/results/ace_network.pt "
                f"--pose_file {intermediate_dir}/results/poses_ace_network.txt")

    # Step 5: Convert to COLMAP format
    run_command(f"python datasets/convert_colmap.py --ace0_file {intermediate_dir}/results/poses_ace_network.txt --output_dir {refine_dir}")

if __name__ == "__main__":
    main()