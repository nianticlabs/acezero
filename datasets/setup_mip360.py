#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import pycolmap
import numpy as np

source_url = "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip"


def download_and_extract(target_path: Path):
    if target_path.is_dir():
        print(f"Target path: {target_path} exists. Skipping download.")
        return

    # Download to a temporary folder then rename if no errors.
    tmp_dst_path = target_path.with_suffix(".tmp")
    if tmp_dst_path.is_dir():
        print(f"Temporary: {tmp_dst_path} exists. Cleaning it.")
        shutil.rmtree(tmp_dst_path)

    tmp_dst_path.mkdir(parents=True)

    tmp_dst = tmp_dst_path / source_url.split("/")[-1]
    print(f"Downloading {source_url} -> {tmp_dst}")
    subprocess.run(["wget", source_url, "-O", tmp_dst], check=True)

    print(f"Download complete. Unpacking archive.")
    subprocess.run(["unzip", "-d", tmp_dst_path, tmp_dst], check=True)

    print(f"Unpacking complete, renaming output folder to: {target_path}")
    tmp_dst_path.rename(target_path)


def process_split(in_dir: Path, out_dir: Path, images_folder: str, split_step: int, is_train: bool):
    # Resolve paths to create relative symlinks.
    in_dir = in_dir.resolve()
    out_dir = out_dir.resolve()

    # Create output dir.
    out_dir.mkdir(exist_ok=True, parents=True)

    try:
        # Used to scale intrinsics (folder name is images_N).
        downsampling_factor = int(images_folder.split("_")[-1])
    except ValueError:
        downsampling_factor = 1

    # Prepare ACE structure
    calibration_path = out_dir / "calibration"
    calibration_path.mkdir(exist_ok=True)
    print(f"Saving calibration files to: {calibration_path}")

    poses_path = out_dir / "poses"
    poses_path.mkdir(exist_ok=True)
    print(f"Saving poses into: {poses_path}")

    rgb_path = out_dir / "rgb"
    rgb_path.mkdir(exist_ok=True)
    print(f"Creating RGB symlinks into: {rgb_path}")

    # Load reconstruction.
    reconstruction_path = in_dir / "sparse" / "0"
    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    print(
        f"Loaded reconstruction from: {reconstruction_path} with {len(reconstruction.cameras)} cameras and {len(reconstruction.images)} images.")

    # Process each image.
    out_img_idx = 0
    for image_idx in sorted(reconstruction.images.keys()):
        # Check image split.
        if is_train and image_idx % split_step == 0:
            continue
        elif not is_train and image_idx % split_step != 0:
            continue

        image = reconstruction.images[image_idx]

        out_image_prefix = f"{out_img_idx:06d}"

        # Symlink RGB file.
        image_name = image.name
        in_image_file = in_dir / images_folder / image_name
        out_image_file = rgb_path / f"{out_image_prefix}.jpg"

        # Replace if it exists already.
        if out_image_file.exists():
            out_image_file.unlink()

        # Create symlink.
        out_image_file.symlink_to(os.path.relpath(in_image_file, start=rgb_path))

        # Create calibration file (save the full 3x3 calibration matrix).
        calibration_file = calibration_path / f"{out_image_prefix}.txt"

        # Scale intrinsics
        calibration_matrix = reconstruction.cameras[image.camera_id].calibration_matrix()
        calibration_matrix[0] /= downsampling_factor
        calibration_matrix[1] /= downsampling_factor

        with calibration_file.open("w") as f:
            np.savetxt(f, calibration_matrix)

        # Create pose file.
        pose_file = poses_path / f"{out_image_prefix}.txt"
        # Colmap stores world to camera'.
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = image.rotation_matrix()
        world_to_camera[:3, 3] = image.tvec
        # ACE uses camera to world.
        camera_to_world = np.linalg.inv(world_to_camera)
        with pose_file.open("w") as f:
            np.savetxt(f, camera_to_world)

        # Increment output counter.
        out_img_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download and setup the Mip-NeRF 360 dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--setup_ace_structure', action='store_true',
                        help='Create a copy of the dataset in the ACE format in 7scenes_ace. '
                             'Otherwise, the dataset is left in the original format in 7scenes.')
    parser.add_argument("--images_folder", type=str, default="images_4", help="Which version of the RGB images to use")
    parser.add_argument("--test_step", type=int, default=8, help="Select 1 test image every N images")
    args = parser.parse_args()

    print("\n############################################################################")
    print("# Please make sure to check this dataset's license before using it!        #")
    print("# https://jonbarron.info/mipnerf360/                                       #")
    print("############################################################################\n\n")

    license_response = input('Please confirm with "yes" or abort. ')
    if not (license_response == "yes" or license_response == "y"):
        print(f"Your response: {license_response}. Aborting.")
        exit()

    images_folder = args.images_folder
    test_step = args.test_step

    source_path = Path(__file__).parent / "mip360"
    print(f"Downloading raw dataset into: {source_path}")
    download_and_extract(source_path)

    if not args.setup_ace_structure:
        print("ACE dataset format not requested. Done.")
        exit()

    processed_path = Path(__file__).parent / "mip360_ace"
    print(f"Creating ACE dataset into: {processed_path}")
    processed_path.mkdir(exist_ok=True)

    for scene_source_dir in source_path.iterdir():
        if not scene_source_dir.is_dir():
            continue

        scene_name = scene_source_dir.name
        scene_target_dir = processed_path / scene_name
        print(f"Processing scene: {scene_name} into {scene_target_dir}")
        scene_target_dir.mkdir(exist_ok=True, parents=True)

        # Process train.
        print(f"Processing train split.")
        process_split(scene_source_dir, scene_target_dir / "train", images_folder, test_step, True)

        print(f"Processing test split.")
        process_split(scene_source_dir, scene_target_dir / "test", images_folder, test_step, False)
