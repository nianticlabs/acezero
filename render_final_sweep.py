#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2023.

import os
from pathlib import Path
import logging
import argparse
import torch

import dataset_io
from ace_visualizer import ACEVisualizer
from distutils.util import strtobool

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def  get_pose_iteration_dict(last_pose_file, max_iteration, confidence_threshold):

        with open(last_pose_file, 'r') as f:
            poses = f.readlines()

        # dictionary contains the first iteration where an image was registered
        pose_dict = {}
        # initialise each image with the last iteration
        for pose in poses:
            img_file = pose.split()[0]
            pose_dict[img_file] = max_iteration

        # loop through all pose files backwards and overwrite the image iteration

        for iteration in reversed(range(max_iteration)):

            pose_file = last_pose_file.stem.split("_")
            pose_file[-1] = f"iteration{iteration}"
            pose_file = "_".join(pose_file)

            if iteration == 0:
                pose_files = list(last_pose_file.parent.glob(f"{pose_file}_seed[0-9].txt"))

                if len(pose_files) > 0:
                    pose_file = pose_files[0]
                else:
                    pose_file = f"{pose_file[6:-1]}1_refined_poses.txt"
                    pose_file = opt.pose_file.parent / pose_file
            else:
                pose_file += ".txt"
                pose_file = last_pose_file.parent / pose_file

            with open(pose_file, 'r') as f:
                current_poses = f.readlines()

            for pose in current_poses:
                img_file = pose.split()[0]
                confidence = float(pose.split()[-1])
                if confidence > confidence_threshold:
                    pose_dict[img_file] = iteration

        return pose_dict

def _strtobool(x):
    return bool(strtobool(x))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Renders additional frames at the end of a reconstruction visualisation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('render_folder', type=Path)

    parser.add_argument('--render_camera_z_offset', type=int, default=4,
                        help='zoom out of the scene by moving render camera backwards, in meters')

    parser.add_argument('--render_marker_size', type=float, default=0.03)

    opt = parser.parse_args()
    device = torch.device("cuda")

    pose_file_conf_threshold = 1000

    # find render state of last iteration
    max_iteration = 100
    state_file_found = False

    for iteration in reversed(range(max_iteration)):
        state_file = opt.render_folder / f"iteration{iteration}_register.pkl"

        if os.path.isfile(state_file):
            state_file_found = True
            break

    if not state_file_found:
        _logger.error(f"Could not find a state file. Last tried: {state_file}")
        exit()

    pose_file = opt.render_folder.parent / f"poses_iteration{iteration}.txt"

    if not os.path.isfile(pose_file):
        _logger.error(f"Could not find a pose file: {pose_file} does not exist.")
        exit()

    # get information when which image was registered
    pose_dict = get_pose_iteration_dict(pose_file, iteration, confidence_threshold=pose_file_conf_threshold)

    # load poses
    rgb_files, poses, _ = dataset_io.load_dataset_ace(pose_file, pose_file_conf_threshold)
    poses = [pose.numpy() for pose in poses]
    pose_iterations = [pose_dict[rgb_file] for rgb_file in rgb_files]

    # setup visualiser
    visualiser = ACEVisualizer(opt.render_folder, flipped_portait=False, map_depth_filter=100,
                               mapping_state_file_name=state_file.name, marker_size=opt.render_marker_size)

    visualiser.render_final_sweep(
        frame_count=150,
        camera_z_offset=opt.render_camera_z_offset,
        poses=poses,
        pose_iterations=pose_iterations,
        total_poses=len(pose_dict))
