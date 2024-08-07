#!/usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path
import pycolmap
import numpy as np
from PIL import Image


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
    
    def quaternion_to_rotation_matrix(q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])

    # Create COLMAP dataset directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)  # Create sparse/0 directory
    
    # Prepare COLMAP files
    cameras_file = output_dir / "sparse" / "0" / "cameras.txt"
    images_file = output_dir / "sparse" / "0" / "images.txt"
    points3d_file = output_dir / "sparse" / "0" / "points3D.txt"
    
    # Read and sort ACE0 data
    with open(ace0_file, 'r') as f:
        ace0_data = f.readlines()
    
    # Sort by image name
    ace0_data.sort(key=lambda x: Path(x.split()[0]).name)
    
    # Read point cloud data
    with open(ace0_file.parent / "point_cloud_out.txt", 'r') as f:
        point_cloud_data = f.readlines()

    # Parse ACE0 data and write to COLMAP files
    with open(cameras_file, 'w') as cf, open(images_file, 'w') as imf, open(points3d_file, 'w') as pf:
        # Write file headers
        cf.write("# Camera list with one line of data per camera:\n")
        cf.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, FX, FY, CX, CY\n")
        imf.write("# Image list with two lines of data per image:\n")
        imf.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        imf.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        

        # Write to points3D.txt
        for i, point in enumerate(point_cloud_data):
            x, y, z, r, g, b = map(float, point.strip().split())
            pf.write(f"{i} {x} {y} {z} 0 0 0 0\n")

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
            cf.write(f"{i+1} PINHOLE {width} {height} {focal_length} {focal_length} {cx} {cy}\n")
            
            # Write to images.txt
            image_name = Path(image_path).name
            imf.write(f"{i+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i+1} {image_name}\n")
            imf.write("\n")  # Second line is empty because we don't have 2D point information
            
            # Copy image file
            shutil.copy2(image_path, output_dir / "images" / image_name)
    
    print(f"COLMAP dataset has been generated at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ACE0 format data to COLMAP dataset format")
    parser.add_argument("--ace0_file", type=Path, help="Path to the ACE0 format data file")
    parser.add_argument("--output_dir", type=Path, help="Output directory for the COLMAP dataset")
    
    args = parser.parse_args()
    
    ace0_to_colmap(args.ace0_file, args.output_dir)