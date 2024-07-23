#! /usr/bin/env python3

import os.path
import pickle
from distutils.util import strtobool

import ace_vis_util as vutil
import argparse
from pathlib import Path
import logging
import torch
from ace_network import Regressor
from dataset import CamLocDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Extract point cloud from network (slow) or visualization buffer file (fast).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('output_file', type=Path)

    parser.add_argument('--network', type=Path, help='network to extract point cloud from.')

    parser.add_argument('--pose_file', type=Path, help='pose file of images that trained the network')

    parser.add_argument('--visualization_buffer', type=Path,
                        help='Vis buffer files that contains a pre-calculated point cloud.')

    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--image_resolution', type=int, default=480, help='base image resolution')

    parser.add_argument('--confidence_threshold', type=int, default=500)

    opt = parser.parse_args()

    if opt.visualization_buffer is None and (opt.network is None or opt.pose_file is None):
        parser.error("You must provide either a visualization buffer or network and pose file.")

    device = torch.device("cuda")

    if opt.visualization_buffer is None:

        _logger.info("Extracting point cloud from network.")

        # Load network weights.
        encoder_state_dict = torch.load(opt.encoder_path, map_location="cpu")
        _logger.info(f"Loaded encoder from: {opt.encoder_path}")
        head_state_dict = torch.load(opt.network, map_location="cpu")
        _logger.info(f"Loaded head weights from: {opt.network}")

        # Create regressor.
        network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)

        # Setup for evaluation.
        network = network.to(device)
        network.eval()

        # Setup dataset.
        dataset = CamLocDataset(
            rgb_files=None,
            image_short_size=opt.image_resolution,
            ace_pose_file=opt.pose_file,
            ace_pose_file_conf_threshold=opt.confidence_threshold
        )
        _logger.info(f'Images found: {len(dataset)}')

        # Setup dataloader. Batch size 1 by default.
        data_loader = DataLoader(dataset, shuffle=False, num_workers=6)

        pc_xyz, pc_clr = vutil.get_point_cloud_from_network(network, data_loader, filter_depth=100)

    else:
        _logger.info("Extracting point cloud from visualization buffer.")

        with open(opt.visualization_buffer, "rb") as file:
            state_dict = pickle.load(file)

        pc_xyz = state_dict['map_xyz']
        pc_clr = state_dict['map_clr']

    with open(opt.output_file.with_suffix(".txt"), 'w') as f:
        for pt in range(pc_xyz.shape[0]):
            f.write(f"{pc_xyz[pt, 0]} {pc_xyz[pt, 1]} {pc_xyz[pt, 2]} "
                    f"{pc_clr[pt, 0]:.0f} {pc_clr[pt, 1]:.0f} {pc_clr[pt, 2]:.0f}\n")

    _logger.info(f"Done. Wrote point cloud to: {opt.output_file}")
