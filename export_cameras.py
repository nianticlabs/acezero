#! /usr/bin/env python3

from distutils.util import strtobool

import ace_vis_util as vutil
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np
from ace_visualizer import ACEVisualizer
import trimesh
import logging
import torch
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Poses to PLY file with camera meshes. Cameras are color-coded by confidence',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('pose_file', type=Path, help="Path to an ACE pose file.")

    parser.add_argument('output_file', type=Path, help="Path to the output PLY file.")

    parser.add_argument('--frustum_scale', type=float, default=0.1, help="Scale factor for camera frustums.")

    parser.add_argument('--frustum_markers', type=_strtobool, default=False,
                        help="Export camera positions as markers only.")

    parser.add_argument('--draw_non_confident', type=_strtobool, default=True,
                        help="Draw poses of non-confident cameras. They will be colored distinctively.")

    parser.add_argument('--confidence_threshold', type=int, default=1000,
                        help="Confidence threshold for color coding of camera frustums or filtering.")

    opt = parser.parse_args()
    device = torch.device("cuda")

    # setup confidence color map
    confidence_max = 5000

    # use two color maps, below and above a user-specific threshold
    confidence_threshold = opt.confidence_threshold
    conf_neg_steps = int(confidence_threshold / confidence_max * 256)
    conf_pos_steps = 256 - conf_neg_steps

    # color map for confident poses
    conf_pos_map = plt.cm.get_cmap("summer")(np.linspace(1, 0, conf_pos_steps))[:, :3]
    # color map for non-confident poses
    conf_neg_map = plt.cm.get_cmap("cool")(np.linspace(1, 0, conf_neg_steps))[:, :3]
    # concatenate both color maps
    conf_color_map = np.concatenate((conf_neg_map, conf_pos_map))

    # read pose file data
    with open(opt.pose_file, 'r') as f:
        poses = f.readlines()

    _logger.info(f"Read {len(poses)} poses from: {opt.pose_file}")

    _logger.info("Writing poses to mesh.")

    # object to store camera meshes
    vis_cameras = vutil.CameraTrajectoryBuffer(frustum_skip=0, frustum_scale=opt.frustum_scale)

    # parse pose file data
    for pose in poses:
        # image info as: file_name, q_w, q_x, q_y, q_z, t_x, t_y, t_z, focal_length, confidence
        pose_tokens = pose.split()

        # read file name and confidence
        file_name = pose_tokens[0]
        confidence = float(pose_tokens[-1])

        # clamp confidence for visualization
        confidence = min(confidence, confidence_max)

        # read pose
        q_wxyz = [float(t) for t in pose_tokens[1:5]]
        t_xyz = [float(t) for t in pose_tokens[5:8]]

        # quaternion to rotation matrix
        R = Rotation.from_quat(q_wxyz[1:] + [q_wxyz[0]]).as_matrix()

        # construct full pose matrix
        T_world2cam = np.eye(4)
        T_world2cam[:3, :3] = R
        T_world2cam[:3, 3] = t_xyz

        # pose files contain world-to-cam but we need cam-to-world
        T_cam2world = np.linalg.inv(T_world2cam)
        T_cam2world_opengl = ACEVisualizer._convert_cv_to_gl(T_cam2world)

        if confidence > confidence_threshold or opt.draw_non_confident:

            if len(poses) == 1:
                # special handling of seed poses
                current_color = (100, 100, 100)
                marker_scale = opt.frustum_scale
            else:
                # map confidence to color
                conf_color_idx = min(int(confidence / confidence_max * 255), 255)
                current_color = conf_color_map[conf_color_idx] * 255
                marker_scale = opt.frustum_scale

            if opt.frustum_markers:
                # export camera as position marker only
                vis_cameras.add_position_marker(T_cam2world_opengl, marker_color=current_color, frustum_maker=True,
                                                marker_extent=marker_scale)
            else:
                # export camera as full frustum object
                vis_cameras.add_camera_frustum(T_cam2world_opengl, frustum_color=current_color)

    # trajectory object to trimesh
    trajectory_mesh = vis_cameras.trajectory + vis_cameras.frustums
    trajectory_mesh = trimesh.util.concatenate(trajectory_mesh)

    # export ply file
    with open(opt.output_file, 'wb') as f:
        f.write(trimesh.exchange.ply.export_ply(trajectory_mesh))

    _logger.info(f"Done. Stored as: {opt.output_file}")
