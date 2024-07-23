#!/usr/bin/env python3

import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
# Important, install a specific version.
# pip install pycolmap==0.4.0
import pycolmap

splits = [
    "training",
    "intermediate",
    "advanced",
    "training_videos",
    "intermediate_videos",
    "advanced_videos",
]
colmap_splits = ["training", "intermediate", "advanced"]
output_height = 540

colmap_archive_url = "https://storage.googleapis.com/niantic-lon-static/research/acezero/colmap_raw.tar.gz"


def extract_split_frames(split_path: Path):
    video_path = split_path / "videos"

    for video_file in sorted(video_path.glob("*.mp4")):
        frames_folder = split_path / video_file.stem
        frames_folder.mkdir(exist_ok=True)

        print(f"Extracting frames from: {video_file} into: {frames_folder}")
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_file,
                "-vf",
                f"scale=-1:{output_height}",
                "-qmin",
                "1",
                "-q:v",
                "1",
                f"{frames_folder}/%06d.jpg",
            ],
            check=True,
        )


def unpack_split(split_archive: Path, split_dir: Path):
    # Unpack to a temporary folder first.
    temp_split_dir = split_dir.with_suffix(".tmp")
    temp_split_dir.mkdir(exist_ok=True)
    print(f"Unpacking {split_archive} into: {temp_split_dir}")

    subprocess.run(["unzip", "-d", temp_split_dir, split_archive], check=True)
    print(f"Unpacking complete.")

    # If the split contains videos, extract frames.
    if "videos" in split_dir.name:
        print(f"Extracting frames from videos in: {temp_split_dir}")
        extract_split_frames(temp_split_dir)
        print(f"Frames extraction complete.")

    # If we got here there were no errors. Rename to the final folder.
    print(f"Renaming {temp_split_dir} to {split_dir}")
    temp_split_dir.rename(split_dir)


def process_scene(in_dir: Path, out_dir: Path):
    # Create symlinks from the raw folder to the output folder.
    for in_image_file in sorted(in_dir.glob("*.jpg")):
        out_image_file = out_dir / in_image_file.name

        if out_image_file.exists():
            out_image_file.unlink()

        # Create symlink.
        out_image_file.symlink_to(os.path.relpath(in_image_file, start=out_dir))


def save_calibration(reconstruction: pycolmap.Reconstruction, out_file: Path):
    # We use the first (and only) camera.

    camera_idx = 1
    if len(reconstruction.cameras) != 1 or camera_idx not in reconstruction.cameras:
        raise ValueError("Expected only one camera in the reconstruction.")

    # We save the average of the focal length.
    calibration = reconstruction.cameras[camera_idx].calibration_matrix()
    focal_length = (calibration[0, 0] + calibration[1, 1]) / 2

    with out_file.open("w") as f:
        f.write(f"{focal_length}\n")
    print(f"Saved focal length ({focal_length:.3f}) to: {out_file}")


def process_colmap_scene(in_dir: Path, colmap_dir: Path, out_dir: Path):
    # We work in a temporary folder then rename at the end.
    temp_out_path = out_dir.with_suffix(".tmp")
    temp_out_path.mkdir(exist_ok=True)

    # Load reconstruction
    reconstruction = pycolmap.Reconstruction(colmap_dir)
    print(
        f"Loaded reconstruction from: {colmap_dir} with"
        f" {len(reconstruction.cameras)} cameras and {len(reconstruction.images)} images."
    )

    # Not all images have a pose, so we create a map: image_name -> image_id to check if a pose exists.
    image_name_to_id = {Path(v.name).name: k for k, v in reconstruction.images.items()}

    # Export focal length.
    out_calibration_path = temp_out_path / "focal_length.txt"
    save_calibration(reconstruction, out_calibration_path)

    # Process each image.
    for image_path in sorted(in_dir.glob("*.jpg")):
        image_name = image_path.name

        out_image_file = temp_out_path / image_name
        out_pose_file = temp_out_path / image_name.replace(".jpg", "_pose.txt")

        if image_name in image_name_to_id:
            # Extract metadata from reconstruction.
            image_id = image_name_to_id[image_name]
            image = reconstruction.images[image_id]

            # Colmap stores world to camera'.
            world_to_camera = np.eye(4)
            world_to_camera[:3, :3] = image.rotation_matrix()
            world_to_camera[:3, 3] = image.tvec
            # ACE uses camera to world.
            camera_to_world = np.linalg.inv(world_to_camera)
        else:
            # Image wasn't reconstructed, create dummy pose matrix.
            camera_to_world = np.full((4, 4), np.inf)

        # Save pose.
        with out_pose_file.open("w") as f:
            np.savetxt(f, camera_to_world)

        # Create symlink.
        if out_image_file.exists():
            out_image_file.unlink()
        out_image_file.symlink_to(os.path.relpath(image_path, start=temp_out_path))

    # If we got here there were no errors. Rename to the final folder.
    print(f"Renaming {temp_out_path} to {out_dir}")
    temp_out_path.rename(out_dir)


def process_colmap_split(
    raw_split_path: Path, raw_colmap_path: Path, ace_split_path: Path
):
    split_name = raw_split_path.name

    # We work in a temporary folder then rename at the end.
    temp_ace_path = ace_split_path.with_suffix(".tmp")
    temp_ace_path.mkdir(exist_ok=True)
    print(f"Processing split (COLMAP): {raw_split_path} -> {temp_ace_path}")

    for raw_scene in raw_split_path.iterdir():
        if not raw_scene.is_dir():
            continue

        if raw_scene.name == "videos":
            continue

        scene_name = raw_scene.name
        raw_scene_colmap = raw_colmap_path / f"{split_name}__{scene_name}" / "0"
        if not raw_scene_colmap.is_dir():
            print(
                f"Colmap data for scene {split_name}/{scene_name} not found. Skipping."
            )
            continue

        temp_scene = temp_ace_path / scene_name
        temp_scene.mkdir(exist_ok=True)

        print(
            f"Processing colmap scene {scene_name} ({raw_scene_colmap}) -> {temp_scene}"
        )
        process_colmap_scene(raw_scene, raw_scene_colmap, temp_scene)
        print(f"Colmap scene {scene_name} processed successfully.")

    # If we got here there were no errors. Rename to the final folder.
    print(f"Renaming {temp_ace_path} to {ace_split_path}")
    temp_ace_path.rename(ace_split_path)


def process_colmap(raw_data_path: Path, ace_data_path: Path):
    colmap_raw_path = ace_data_path / "colmap_raw"
    colmap_raw_archive = colmap_raw_path.with_suffix(".tar.gz")

    if not colmap_raw_archive.is_file():
        print(f"Downloading colmap archive from: {colmap_archive_url}")
        subprocess.run(["wget", colmap_archive_url, "-P", ace_data_path], check=True)

    print(f"Colmap raw data archive found: {colmap_raw_archive}. Processing it.")
    if not colmap_raw_path.is_dir():
        # Extract to a temporary folder.
        colmap_tmp_path = colmap_raw_path.with_suffix(".tmp")
        colmap_tmp_path.mkdir(exist_ok=True)

        print(f"Extracting colmap raw data to: {colmap_tmp_path}")
        subprocess.run(["tar", "-xzf", colmap_raw_archive, "-C", colmap_tmp_path], check=True)

        # If we got here there were no errors. Rename to the final folder.
        print(f"Renaming {colmap_tmp_path} to {colmap_raw_path}")
        colmap_tmp_path.rename(colmap_raw_path)

    # Process the colmap data.
    ace_data_path.mkdir(exist_ok=True)
    print(f"Creating colmap dataset into: {ace_data_path}")

    for split in colmap_splits:
        print(f"Processing split: {split}")
        raw_split_path = raw_path / split
        ace_split_path = ace_data_path / split

        if not raw_split_path.is_dir():
            print(f"Raw split folder: {raw_split_path} doesn't exist. Skipping split.")
            continue

        if ace_split_path.is_dir():
            print(f"ACE split folder: {ace_split_path} already exists. Nothing to do.")
            continue

        process_colmap_split(raw_split_path, colmap_raw_path, ace_split_path)
        print(f"Split {split} processed successfully.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Setup the Tanks and Temples dataset.")
    parser.add_argument(
        "--with-colmap",
        action="store_true",
        help="Download the colmap data and create a variant of the datasets with SFM poses to use as initialisation.",
    )
    args = parser.parse_args()

    print("\n############################################################################")
    print("# Please make sure to check this dataset's license before using it!        #")
    print("# https://www.tanksandtemples.org/license/                                 #")
    print("############################################################################\n\n")

    license_response = input('Please confirm with "yes" or abort. ')
    if not (license_response == "yes" or license_response == "y"):
        print(f"Your response: {license_response}. Aborting.")
        exit()

    if args.with_colmap:

        print("\n#############################################################################")
        print("# To enable reproducing our paper results, we make available COLMAP         #")
        print("# reconstructions for the Tanks&Temples dataset. Please make sure you agree #")
        print("# to the COLMAP license, including licenses of its third party components   #")
        print("# and the Tanks and Temples data license before using them.                 #")
        print("# https://github.com/colmap/colmap?tab=License-1-ov-file#readme             #")
        print("# https://github.com/colmap/colmap/blob/main/src/thirdparty/SiftGPU/LICENSE #")
        print("# https://www.tanksandtemples.org/license/                                  #")
        print("#############################################################################\n\n")

        license_response = input('Please confirm with "yes" or abort. ')
        if not (license_response == "yes" or license_response == "y"):
            print(f"Your response: {license_response}. Aborting.")
            exit()

    datasets_path = Path(__file__).parent

    raw_path = datasets_path / "t2"
    raw_path.mkdir(exist_ok=True, parents=True)
    print(f"Preparing raw dataset into: {raw_path}")

    # Unpack datasets.
    print("Unpacking datasets.")
    available_splits = []

    for split in splits:
        print(f"Processing split: {split}")
        split_dir = raw_path / split
        split_archive = split_dir.with_suffix(".zip")

        if split_dir.is_dir():
            print(f"Split folder: {split_dir} already exists. Nothing to do.")
            available_splits.append(split)
            continue

        print(
            f"Split folder: {split_dir} doesn't exist. Checking for archive: {split_archive}"
        )
        if not split_archive.is_file():
            print(
                f"Archive {split_archive} doesn't exist. Skipping split.\n"
                f"\tNOTE: If you want to test on this data, please download it from:"
                f"\n\t'https://www.tanksandtemples.org/download' or 'https://github.com/isl-org/TanksAndTemples/issues/35'"
                f"\n\tand place it in "
                f"{raw_path}, without unpacking it, then rerun the script."
            )
            continue

        unpack_split(split_archive, split_dir)
        available_splits.append(split)
        print(f"Split {split} unpacked successfully.")

    print(
        f"Finished processing the dataset.\nTanks and Temples ACE data is available in: {raw_path}. Available splits: {available_splits}"
    )

    # Optionally, process the colmap data and create a variant of the datasets with SFM poses.
    if args.with_colmap:
        ace_colmap_path = datasets_path / "t2_colmap"

        print(
            f"Processing the colmap data to create a variant of the datasets with SFM poses in: {ace_colmap_path}"
        )
        process_colmap(raw_path, ace_colmap_path)
        print(
            f"Finished processing the colmap data.\nTanks and Temples ACE COLMAP data is available in: {ace_colmap_path}"
        )
