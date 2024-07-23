#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.

import os

os.environ["MKL_NUM_THREADS"] = "1"  # noqa: E402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa: E402

import dataset_io
import argparse
import logging
import pickle
import time
from distutils.util import strtobool
from pathlib import Path
import random

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import dsacstar
from ace_network import Regressor
from dataset import CamLocDataset

import eval_poses_util as tutil
from ace_visualizer import ACEVisualizer

_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':
    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Test a trained network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('rgb_files', type=str, help="Glob pattern for RGB files, e.g. 'datasets/scene/*.jpg'")

    parser.add_argument('network', type=Path, help='path to a network trained for the scene (just the head weights)')

    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--session', '-sid', default='',
                        help='custom session name appended to output files, '
                             'useful to separate different runs of a script')

    parser.add_argument('--image_resolution', type=int, default=480, help='base image resolution')

    parser.add_argument('--num_data_workers', type=int, default=12,
                        help='number of data loading workers, set according to the number of available CPU cores')

    # DSACStar RANSAC parameters. ACE Keeps them at default.
    parser.add_argument('--hypotheses', '-hyps', type=int, default=64,
                        help='number of hypotheses, i.e. number of RANSAC iterations')

    parser.add_argument('--threshold', '-t', type=float, default=10,
                        help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

    parser.add_argument('--inlieralpha', '-ia', type=float, default=100,
                        help='alpha parameter of the soft inlier count; controls the softness of the '
                             'hypotheses score distribution; lower means softer')

    parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100,
                        help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking '
                             'pose consistency towards all measurements; error is clamped to this value for stability')

    # Params for the visualization. If enabled, it will slow down relocalisation considerably. But you get a nice video :)
    parser.add_argument('--render_visualization', type=_strtobool, default=False,
                        help='create a video of the mapping process')

    parser.add_argument('--render_target_path', type=Path, default='renderings',
                        help='target folder for renderings, visualizer will create a subfolder with the map name')

    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False,
                        help='flag for wayspots dataset where images are sideways portrait')

    parser.add_argument('--render_pose_conf_threshold', type=int, default=5000,
                        help='max confidence value expected')

    parser.add_argument('--render_map_depth_filter', type=int, default=10,
                        help='to clean up the ACE point cloud remove points too far away')

    parser.add_argument('--render_camera_z_offset', type=int, default=4,
                        help='zoom out of the scene by moving render camera backwards, in meters')

    parser.add_argument('--base_seed', type=int, default=1305,
                        help='seed to control randomness')

    parser.add_argument('--confidence_threshold', type=float, default=1000,
                        help='Consider an image successfully registered if its confidence is above this threshold.')

    parser.add_argument('--max_estimates', type=int, default=-1,
                        help='max number of images to consider')

    parser.add_argument('--use_external_focal_length', type=float, default=-1,
                        help="Provide the focal length of images. Can be refined. "
                             "-1: Use 70% of image diagonal as guess.")

    parser.add_argument('--render_marker_size', type=float, default=0.03,
                        help='size of the camera frustums in the visualization')


    opt = parser.parse_args()

    device = torch.device("cuda")
    num_workers = opt.num_data_workers

    #set random seeds
    torch.manual_seed(opt.base_seed)
    np.random.seed(opt.base_seed)
    random.seed(opt.base_seed)

    head_network_path = Path(opt.network)
    encoder_path = Path(opt.encoder_path)
    session = opt.session

    use_heuristic_focal_length = opt.use_external_focal_length < 0

    # Setup dataset.
    testset = CamLocDataset(
        rgb_files=opt.rgb_files,
        image_short_size=opt.image_resolution,
        use_heuristic_focal_length=use_heuristic_focal_length,
    )
    _logger.info(f'Test images found: {len(testset)}')

    # Overwrite dataset heuristic focal length with external value if provided.
    if opt.use_external_focal_length > 0:
        testset.set_external_focal_length(opt.use_external_focal_length)
        _logger.info(f"Using external focal length: {opt.use_external_focal_length}")

    # Setup dataloader. Batch size 1 by default.
    testset_loader = DataLoader(testset, shuffle=True, num_workers=6, timeout=60)

    # Load network weights.
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    _logger.info(f"Loaded encoder from: {encoder_path}")
    head_state_dict = torch.load(head_network_path, map_location="cpu")
    _logger.info(f"Loaded head weights from: {head_network_path}")

    # Create regressor.
    network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)

    # Setup for evaluation.
    network = network.to(device)
    network.eval()

    # Save the outputs in the same folder as the network being evaluated.
    output_dir = head_network_path.parent

    # This will contain each frame's pose (stored as quaternion + translation) and errors.
    pose_log_file = output_dir / f'poses_{opt.session}.txt'
    _logger.info(f"Saving per-frame poses and errors to: {pose_log_file}")

    # Setup output files.
    pose_log = open(pose_log_file, 'w', 1)

    # Metrics of interest.
    avg_batch_time = 0
    num_batches = 0

    # Generate video of training process
    if opt.render_visualization:
        # infer rendering folder from map file name
        target_path = opt.render_target_path
        os.makedirs(target_path, exist_ok=True)
        state_file = opt.network.stem + "_mapping.pkl"

        ace_visualizer = ACEVisualizer(target_path,
                                       opt.render_flipped_portrait,
                                       opt.render_map_depth_filter,
                                       reloc_vis_conf_threshold=opt.render_pose_conf_threshold,
                                       confidence_threshold=opt.confidence_threshold,
                                       mapping_state_file_name=state_file,
                                       marker_size=opt.render_marker_size)

        ace_visualizer.setup_reloc_visualisation(
            frame_count=len(testset),
            camera_z_offset=opt.render_camera_z_offset)
    else:
        ace_visualizer = None

    ace_estimates = []

    # Testing loop.
    testing_start_time = time.time()
    with torch.no_grad():
        for image_B1HW, _, _, _, intrinsics_B33, _, _, filenames, indices in testset_loader:
            batch_start_time = time.time()
            batch_size = image_B1HW.shape[0]

            image_B1HW = image_B1HW.to(device, non_blocking=True)

            # Predict scene coordinates.
            with autocast(enabled=True):
                scene_coordinates_B3HW = network(image_B1HW)

            # We need them on the CPU to run RANSAC.
            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

            # Each frame is processed independently.
            for scene_coordinates_3HW, intrinsics_33, frame_path, index in zip(scene_coordinates_B3HW, intrinsics_B33, filenames, indices):

                # We support a single focal length.
                assert torch.allclose(intrinsics_33[0, 0], intrinsics_33[1, 1])
                # Extract focal length and principal point from the intrinsics matrix.
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()

                # Allocate output variable.
                out_pose = torch.zeros((4, 4))

                # Compute the pose via RANSAC.
                inlier_count = dsacstar.forward_rgb(
                    scene_coordinates_3HW.unsqueeze(0),
                    out_pose,
                    opt.hypotheses,
                    opt.threshold,
                    focal_length,
                    ppX,
                    ppY,
                    opt.inlieralpha,
                    opt.maxpixelerror,
                    network.OUTPUT_SUBSAMPLE,
                    opt.base_seed
                )

                # Store estimates.
                ace_estimates.append(tutil.TestEstimate(
                    pose_est=out_pose.numpy().copy(),
                    pose_gt=None,
                    focal_length=testset.get_focal_length(index),
                    confidence=inlier_count,
                    image_file=frame_path
                ))

            avg_batch_time += time.time() - batch_start_time
            num_batches += 1

            if 0 < opt.max_estimates <= len(ace_estimates):
                _logger.info(f"Stopping at {len(ace_estimates)} estimates.")
                break

    # Process estimates and write them to file.
    for estimate in ace_estimates:

        pose_est = estimate.pose_est

        _logger.info(f"Frame: {estimate.image_file}, Confidence: {estimate.confidence}")

        if ace_visualizer is not None:
            ace_visualizer.render_reloc_frame(
                query_file=estimate.image_file,
                est_pose=pose_est,
                confidence=estimate.confidence)

        # Write estimated pose to pose file (inverse).
        out_pose = np.linalg.inv(pose_est)
        dataset_io.write_pose_to_pose_file(pose_log, rgb_file=estimate.image_file, pose=out_pose,
                                           confidence=estimate.confidence, focal_length=estimate.focal_length)

    # Compute average time.
    avg_time = avg_batch_time / num_batches
    _logger.info(f"Avg. processing time: {avg_time * 1000:4.1f}ms")

    pose_log.close()

    if ace_visualizer is not None:
        # update visualisation buffer
        state_in_file = os.path.join(opt.render_target_path, opt.network.stem + "_mapping.pkl")
        state_out_file = os.path.join(opt.render_target_path, opt.network.stem + "_register.pkl")

        with open(state_in_file, "rb") as file:
            state_dict = pickle.load(file)

        # save state for smooth transition when rendering the localisation phase
        state_dict['frame_idx'] = ace_visualizer.frame_idx
        state_dict['camera_buffer'] = ace_visualizer.scene_camera.get_camera_buffer()

        with open(state_out_file, "wb") as file:
            pickle.dump(state_dict, file)
        _logger.info(f"Stored rendering buffer to {state_out_file}.")
