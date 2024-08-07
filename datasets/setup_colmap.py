#!/usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path
import pycolmap
import numpy as np

def process_split(in_dir: Path, out_dir: Path, images_folder: str, split_step: int, is_train: bool, all_train: bool):
    """
    Process COLMAP dataset and convert to ACE0 format.

    Args:
        in_dir (Path): Input directory with COLMAP reconstruction files.
        out_dir (Path): Output directory for ACE0 formatted data.
        images_folder (str): Name of the RGB images folder.
        split_step (int): Interval for selecting test images.
        is_train (bool): True for training split, False for test split.
        all_train (bool): If True, use all images for training.

    This function:
    1. Creates ACE0 directory structure.
    2. Loads COLMAP reconstruction data.
    3. Processes images, filtering based on split.
    4. Saves intrinsics and extrinsics matrices.
    5. Creates symbolic links to RGB images.

    Assumes valid COLMAP sparse reconstruction in 'arkit/0' subdirectory.
    Applies downsampling to intrinsics if images_folder name contains factor.
    """
    
    in_dir, out_dir = in_dir.resolve(), out_dir.resolve()
    out_dir.mkdir(exist_ok=True, parents=True)

    try:
        downsampling_factor = int(images_folder.split("_")[-1])
    except ValueError:
        downsampling_factor = 1

    # Prepare ACE0 structure
    intrinsics_path = out_dir / "calibration"
    extrinsics_path = out_dir / "poses"
    rgb_path = out_dir / "rgb"
    for path in [intrinsics_path, extrinsics_path, rgb_path]:
        path.mkdir(exist_ok=True)
        print(f"Created directory: {path}")

    # Load reconstruction
    reconstruction_path = in_dir / "colmap" / "arkit" / "0"
    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    print(f"Loaded reconstruction from {reconstruction_path}: {len(reconstruction.cameras)} cameras, {len(reconstruction.images)} images")

    # Process images
    for out_img_idx, image_idx in enumerate(sorted(reconstruction.images.keys())):
        if not all_train and ((is_train and image_idx % split_step == 0) or (not is_train and image_idx % split_step != 0)):
            continue

        image = reconstruction.images[image_idx]
        out_image_prefix = f"{out_img_idx:06d}"

        # Copy RGB file
        in_image_file = in_dir / images_folder / image.name
        out_image_file = rgb_path / f"{out_image_prefix}.jpg"
        shutil.copy2(in_image_file, out_image_file)

        # Save intrinsics
        intrinsics_matrix = reconstruction.cameras[image.camera_id].calibration_matrix()
        intrinsics_matrix[:2] /= downsampling_factor
        np.savetxt(intrinsics_path / f"{out_image_prefix}.txt", intrinsics_matrix)

        # Save extrinsics
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = image.rotation_matrix()
        world_to_camera[:3, 3] = image.tvec
        camera_to_world = np.linalg.inv(world_to_camera)
        np.savetxt(extrinsics_path / f"{out_image_prefix}.txt", camera_to_world)

    print(f"Processed {out_img_idx + 1} images")


def ace0_to_colmap(ace0_file: Path, output_dir: Path):
    """
    Convert ACE0 format data to COLMAP dataset format.

    Args:
    ace0_file (Path): Path to the ACE0 format data file
    output_dir (Path): Output directory for the COLMAP dataset

    This function will:
    1. Create COLMAP dataset directory structure
    2. Parse and sort ACE0 format data
    3. Generate cameras.txt, images.txt, and points3D.txt files required by COLMAP
    4. Copy image files to the COLMAP dataset directory
    """
    
    # Create COLMAP dataset directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    
    # Prepare COLMAP files
    cameras_file = output_dir / "cameras.txt"
    images_file = output_dir / "images.txt"
    points3d_file = output_dir / "points3D.txt"
    
    # Read and sort ACE0 data
    with open(ace0_file, 'r') as f:
        ace0_data = f.readlines()
    
    # Sort by image name
    ace0_data.sort(key=lambda x: Path(x.split()[0]).name)
    
    # Parse ACE0 data and write to COLMAP files
    with open(cameras_file, 'w') as cf, open(images_file, 'w') as imf, open(points3d_file, 'w') as pf:
        # Write file headers
        cf.write("# Camera list with one line of data per camera:\n")
        cf.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        imf.write("# Image list with two lines of data per image:\n")
        imf.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        imf.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for i, line in enumerate(ace0_data):
            parts = line.strip().split()
            image_path = parts[0]
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            focal_length = float(parts[8])
            
            # Get actual image resolution
            with Image.open(image_path) as img:
                width, height = img.size
            
            # Calculate principal point coordinates
            cx, cy = width / 2, height / 2
            
            # Write to cameras.txt
            cf.write(f"{i+1} PINHOLE {width} {height} {focal_length} {cx} {cy} 0\n")
            
            # Write to images.txt
            image_name = Path(image_path).name
            imf.write(f"{i+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i+1} {image_name}\n")
            imf.write("\n")  # Second line is empty because we don't have 2D point information
            
            # Copy image file
            shutil.copy2(image_path, output_dir / "images" / image_name)
    
    print(f"COLMAP dataset has been generated at {output_dir}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert a COLMAP dataset to ACE0 format.")
    parser.add_argument("--root", type=Path, help="Root directory of the COLMAP dataset.")
    parser.add_argument("--output_dir", type=Path, help="Output directory where the ACE0 formatted dataset will be saved.")
    parser.add_argument("--images_folder", type=str, default="images_4", help="The folder containing the RGB images to use.")
    parser.add_argument("--test_step", type=int, default=8, help="Step interval for selecting test images.")
    parser.add_argument("--all_train", action="store_true", help="Use all images for training without splitting.")
    
    args = parser.parse_args()

    # Define input and output paths
    input_directory = args.root
    output_directory = args.output_dir
    images_folder_name = args.images_folder
    test_image_step = args.test_step
    all_train = args.all_train

    # Process train split
    print("Processing train split...")
    process_split(
        in_dir=input_directory,
        out_dir=output_directory / "train",
        images_folder=images_folder_name,
        split_step=test_image_step,
        is_train=True,
        all_train=all_train
    )

    # Process test split only if not using all images for training
    if not all_train:
        print("Processing test split...")
        process_split(
            in_dir=input_directory,
            out_dir=output_directory / "test",
            images_folder=images_folder_name,
            split_step=test_image_step,
            is_train=False,
            all_train=all_train
        )
    else:
        print("Skipping test split as all images are used for training")