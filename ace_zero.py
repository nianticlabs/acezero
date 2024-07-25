#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2023.

import logging
import shutil
from pathlib import Path
import os
import numpy as np
import argparse
from distutils.util import strtobool
import time
import ace_zero_util as zutil
from joblib import Parallel, delayed

import dataset_io

_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':

    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Run ACE0 for a dataset or a scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('rgb_files', type=str, help="Glob pattern for RGB files, e.g. 'datasets/scene/*.jpg'")

    parser.add_argument('results_folder', type=Path, help='path to output folder for result files')

    parser.add_argument('--depth_files', type=str, default=None,
                        help="Which depth to use for the seed image, Glob pattern. "
                             "Correspondence to rgb files via alphateical ordering. "
                             "None: estimate depth using ZoeDepth")

    # === Main reconstruction loop =====================================================================================

    parser.add_argument('--iterations_max', type=int, default=100,
                        help="Maximum number of ACE0 iterations, ie mapping and relocalisation rounds.")
    parser.add_argument('--registration_threshold', type=float, default=0.99,
                        help="Stop reconstruction when this ratio of images has been registered.")
    parser.add_argument('--relative_registration_threshold', type=float, default=0.01,
                        help="Stop reconstruction when less percent of images was registered wrt the last iteration.")
    parser.add_argument('--final_refine', type=_strtobool, default=True,
                        help="One more round of mapping when the stopping criteria have been met.")
    parser.add_argument('--final_refit', type=_strtobool, default=True,
                        help="Refit new (uninitialised) network in last iteration without early stopping")
    parser.add_argument('--final_refit_posewait', type=int, default=5000,
                        help="Fix poses for the first n training iterations of the final refit.")
    parser.add_argument('--refit_iterations', type=int, default=25000,
                        help='Number of training iterations for the final refit.')
    parser.add_argument('--registration_confidence', type=int, default=500,
                        help="Consider an image registered if it has this many inlier scene coordinates.")

    parser.add_argument('--try_seeds', type=int, default=5,
                        help="Number of random images to try when starting the reconstruction.")
    parser.add_argument('--seed_parallel_workers', type=int, default=3,
                        help="Number of parallel workers for seed mapping. "
                             "ZoeDepth might be a limiting factor in terms of GPU memory. "
                             "-1 -> all available cores, 1 -> no parallelism.")
    parser.add_argument('--seed_iterations', type=int, default=10000,
                        help="Maximum number of ACE training iterations for seed images.")

    parser.add_argument('--seed_network', type=Path, default=None,
                        help="Path to a pre-trained network to start the reconstruction.")

    parser.add_argument('--warmstart', type=_strtobool, default=True,
                        help="For each ACE0 mapping round, load the ACE weights of the last iteration.")

    # === Pose refinement ==================================================================================================

    parser.add_argument('--refinement', type=str, default="mlp", choices=['mlp', 'none', 'naive'],
                        help="How to refine poses. MLP: refinement network. Naive: Backprop to poses.")
    parser.add_argument('--refinement_ortho', type=str, default="gram-schmidt", choices=['gram-schmidt', 'procrustes'],
                        help="How to orthonormalise rotations when refining poses.")
    parser.add_argument('--pose_refinement_wait', type=int, default=0,
                        help="Keep poses frozen for the first n training iterations of ACE.")
    parser.add_argument('--pose_refinement_lr', type=float, default=0.001,
                        help="Learning rate for pose refinement.")

    # === Calibration refinement ===========================================================================================

    parser.add_argument('--refine_calibration', type=_strtobool, default=True,
                        help="Optimize focal length during mapping.")
    parser.add_argument('--use_external_focal_length', type=float, default=-1,
                        help="Provide the focal length of images. Can be refined. "
                             "-1: Use 70% of image diagonal as guess.")

    # === ACE Early Stopping ===============================================================================================

    parser.add_argument('--learning_rate_schedule', type=str, default="1cyclepoly",
                        choices=["circle", "constant", "1cyclepoly"],
                        help='circle: move from min to max to min, constant: stay at min, '
                             '1cyclepoly: linear approximation of 1cycle to support early stopping')
    parser.add_argument('--learning_rate_max', type=float, default=0.003, help="max learning rate of the lr schedule")
    parser.add_argument('--cooldown_iterations', type=int, default=5000,
                        help="train to min learning rate for this many iterations after early stopping criterion has been met")
    parser.add_argument('--cooldown_threshold', type=float, default=0.7,
                        help="Start cooldown after this percent of batch pixels are below the inlier reprojection error.")

    # === General ACE parameters ===========================================================================================

    parser.add_argument('--image_resolution', type=int, default=480,
                        help='base image resolution')
    parser.add_argument('--num_head_blocks', type=int, default=1,
                        help='depth of the regression head, defines the map size')
    parser.add_argument('--max_dataset_passes', type=int, default=10,
                        help='max number of repetition of mapping images (with different augmentations)')
    parser.add_argument('--repro_loss_type', type=str, default="tanh",
                        choices=["l1", "l1+sqrt", "l1+log", "tanh", "dyntanh"],
                        help='Loss function on the reprojection error. Dyn varies the soft clamping threshold')
    parser.add_argument('--repro_loss_hard_clamp', type=int, default=1000,
                        help='hard clamping threshold for the reprojection losses')
    parser.add_argument('--repro_loss_soft_clamp', type=int, default=50,
                        help='soft clamping threshold for the reprojection losses')
    parser.add_argument('--aug_rotation', type=int, default=15,
                        help='max inplane rotation angle')
    parser.add_argument('--num_data_workers', type=int, default=12,
                        help='number of data loading workers, set according to the number of available CPU cores')
    parser.add_argument('--training_buffer_cpu', type=_strtobool, default=False, 
                        help='store training buffer on CPU memory instead of GPU, '
                        'this allows running ACE0 on smaller GPUs, but is slower')

    # === Registration parameters ==========================================================================================

    parser.add_argument('--ransac_iterations', type=int, default=32,
                        help="Number of RANSAC hypothesis when registering mapping frames.")
    parser.add_argument('--ransac_threshold', type=float, default=10,
                        help='RANSAC inlier threshold in pixels')

    # === Visualisation parameters =========================================================================================

    parser.add_argument('--render_visualization', type=_strtobool, default=False,
                        help="Render visualisation frames of the whole reconstruction process.")
    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False,
                        help="Dataset images are 90deg flipped (like Wayspots).")
    parser.add_argument('--render_marker_size', type=float, default=0.03,
                        help="Size of the camera marker when rendering scenes.")
    parser.add_argument('--iterations_output', type=int, default=500,
                        help='how often to print the loss and render a frame')

    parser.add_argument('--random_seed', type=int, default=1305,
                        help='random seed, predominately used to select seed images')

    opt = parser.parse_args()

    # create output directory
    os.makedirs(opt.results_folder, exist_ok=True)

    _logger.info(f"Starting reconstruction of files matching {opt.rgb_files}.")
    reconstruction_start_time = time.time()

    # We warm up the torch.hub cache and make sure the depth estimation model is available 
    # before we start the main ACE0 loop (ACE0 uses multiple processes for the initial seed
    # stage and the download should run only once).
    _logger.info(f"Downloading ZoeDepth model from the main process.")
    model = dataset_io.get_depth_model()
    del model
    _logger.info(f"Depth estimation model ready to use.")

    if opt.seed_network is not None:
        _logger.info(f"Using pre-trained network as seed: {opt.seed_network}")
        iteration_id = opt.seed_network.stem
    else:
        # use individual images as seeds, try multiple and choose the one that registers the most images
        np.random.seed(opt.random_seed)
        seeds = np.random.uniform(size=opt.try_seeds)
        _logger.info(f"Trying seeds: {seeds}")

        # process seeds in parallel
        arg_instances = []
        for seed_idx, seed in enumerate(seeds):
            # show progress only for the first seed or if we are not using parallel workers
            verbose = (seed_idx == 0) or (opt.seed_parallel_workers == 1)
            arg_instances.append((seed_idx, seed, opt.rgb_files, opt.results_folder, opt, verbose, False, False))

        if opt.seed_parallel_workers != 1:
            _logger.info(f"Processing {len(arg_instances)} seeds in parallel.")

        # as we process initial seeds, keep track of their registration rates
        seed_reg_rates = Parallel(n_jobs=opt.seed_parallel_workers, verbose=1)(
            map(delayed(zutil.map_seed), arg_instances))

        for seed_idx, seed in enumerate(seeds):
            _logger.info(f"Seed {seed_idx}: {seed} -> {seed_reg_rates[seed_idx] * 100:.1f}%")

        # select the best seed
        best_seed = np.argmax(seed_reg_rates)
        iteration_id = zutil.get_seed_id(best_seed)

        _logger.info(f"Selected best seed {iteration_id} "
                     f"with registration rate: {seed_reg_rates[best_seed] * 100:.1f}%")

        # if a visualisation is requested, we need to re-map the best seed with visualisation enabled
        if opt.render_visualization:
            _logger.info(f"Re-mapping best seed {iteration_id} with visualisation enabled.")
            zutil.map_seed((best_seed, seeds[best_seed], opt.rgb_files, opt.results_folder, opt, True, True, True))

    _logger.info(f"Registering all images to best seed {iteration_id}.")

    # Register all images to the best seed. Also render visualisation if requested.
    # In some cases, this is redundant - when the dataset is small and the seed scoring already registered all images
    # AND no visualisation was requested. However, for small datasets, this is fast anyway.
    reg_cmd = [
        zutil.REGISTER_EXE,
        opt.rgb_files,
        opt.results_folder / f"{iteration_id}.pt",
        "--render_visualization", opt.render_visualization,
        "--render_target_path", zutil.get_render_path(opt.results_folder),
        "--render_marker_size", opt.render_marker_size,
        "--render_flipped_portrait", opt.render_flipped_portrait,
        "--session", f"{iteration_id}",
        "--confidence_threshold", opt.registration_confidence,
        "--use_external_focal_length", opt.use_external_focal_length,
        "--hypotheses", opt.ransac_iterations,
        "--threshold", opt.ransac_threshold,
        "--image_resolution", opt.image_resolution,
        "--num_data_workers", opt.num_data_workers
    ]
    zutil.run_cmd(reg_cmd)

    scheduled_to_stop_early = False
    prev_iteration_id = iteration_id

    # check the number of registered mapping images
    max_registration_rate = zutil.get_registration_rates(
        pose_file=opt.results_folder / f"poses_{iteration_id}.txt",
        thresholds=[opt.registration_confidence])[0]
    _logger.info(f"Best seed successfully registered {max_registration_rate * 100:.1f}% of mapping images.")

    # iterate mapping and registration starting from the best seed iteration
    for iteration in range(1, opt.iterations_max):

        iteration_id = f"iteration{iteration}"

        if scheduled_to_stop_early and opt.final_refit:
            # get full refitting mapping call
            mapping_cmd = zutil.get_refit_mapping_cmd(opt.rgb_files, iteration_id, opt.results_folder, opt)
        else:
            # get base mapping call
            mapping_cmd = zutil.get_base_mapping_cmd(opt.rgb_files, iteration_id, opt.results_folder, opt)

        # setting parameters for mapping after initial seed
        mapping_cmd += [
            "--render_visualization", opt.render_visualization,
            "--use_ace_pose_file", f"{opt.results_folder}/poses_{prev_iteration_id}.txt",
            "--pose_refinement", opt.refinement,
            "--use_existing_vis_buffer", f"{prev_iteration_id}_register.pkl",
            "--refine_calibration", opt.refine_calibration,
            "--num_data_workers", opt.num_data_workers,
        ]

        # load previous network weights starting from iteration 2, or if we started from a seed network
        if (opt.warmstart and iteration > 1) or (opt.warmstart and opt.seed_network is not None):
            # skip warmstart in final refit iteration
            if not (opt.final_refit and scheduled_to_stop_early):
                mapping_cmd += ["--load_weights", f"{opt.results_folder}/{prev_iteration_id}.pt"]

        zutil.run_cmd(mapping_cmd)

        # register all images to the current map
        _logger.info(f"Registering all images to map {iteration_id}.")

        reg_cmd = [
            zutil.REGISTER_EXE,
            opt.rgb_files,
            opt.results_folder / f"{iteration_id}.pt",
            "--render_visualization", opt.render_visualization,
            "--render_target_path", zutil.get_render_path(opt.results_folder),
            "--render_marker_size", opt.render_marker_size,
            "--session", iteration_id,
            "--confidence_threshold", opt.registration_confidence,
            "--render_flipped_portrait", opt.render_flipped_portrait,
            "--image_resolution", opt.image_resolution,
            "--hypotheses", opt.ransac_iterations,
            "--threshold", opt.ransac_threshold,
            "--num_data_workers", opt.num_data_workers
        ]

        # Get current focal length estimate from the pose file of the previous mapping iteration
        _, _, focal_lengths = dataset_io.load_dataset_ace(
            pose_file=opt.results_folder / f"poses_{iteration_id}_preliminary.txt",
            confidence_threshold=opt.registration_confidence)

        # We support a single focal length.
        assert np.allclose(focal_lengths, focal_lengths[0])

        _logger.info("Passing previous focal length estimate to registration.")
        reg_cmd += ["--use_external_focal_length", focal_lengths[0]]

        zutil.run_cmd(reg_cmd)

        # check the number of registered mapping images
        registration_rate = zutil.get_registration_rates(
            pose_file=opt.results_folder / f"poses_{iteration_id}.txt",
            thresholds=[opt.registration_confidence])[0]

        _logger.info(f"Successfully registered {registration_rate*100:.1f}% of mapping images.")

        prev_iteration_id = iteration_id

        if scheduled_to_stop_early:
            # we are in the final refinement iteration and stop here
            break

        # check stopping criteria
        if (registration_rate >= opt.registration_threshold) or ((registration_rate-max_registration_rate) < opt.relative_registration_threshold):
            if opt.final_refine:
                # stopping criteria have been met, but we want to do one more round of mapping
                _logger.info(f"Stopping training loop in next iteration. Enough mapping images registered. "
                             f"(Threshold={opt.registration_threshold * 100:.1f}%")
                scheduled_to_stop_early = True
            else:
                # stopping criteria have been met, and we do not want to do one more round of mapping
                _logger.info(f"Stopping training loop. Enough mapping images registered. "
                             f"(Threshold={opt.registration_threshold * 100:.1f}%")
                break

        # stop in any case if we are approaching the maximum number of iterations
        if iteration >= (opt.iterations_max - 2):
            scheduled_to_stop_early = True

        max_registration_rate = max(registration_rate, max_registration_rate)

    if opt.render_visualization:
        _logger.info("Rendering final sweep.")

        zutil.run_cmd(["./render_final_sweep.py",
                       zutil.get_render_path(opt.results_folder),
                       "--render_marker_size", opt.render_marker_size
                       ])

        _logger.info("Converting to video.")
        zutil.run_cmd(["/usr/bin/ffmpeg",
                       "-y",
                       "-framerate", 30,
                       "-pattern_type", "glob",
                       "-i", f"{zutil.get_render_path(opt.results_folder)}/*.png",
                       "-c:v", "libx264",
                       "-pix_fmt", "yuv420p",
                       opt.results_folder / "reconstruction.mp4"
                       ])

    reconstruction_end_time = time.time()
    reconstruction_time = reconstruction_end_time - reconstruction_start_time
    _logger.info(f"Reconstructed in {reconstruction_time/60:.1f} Minutes.")

    # check the number of registered mapping images
    registration_rates = zutil.get_registration_rates(
        pose_file=opt.results_folder / f"poses_{iteration_id}.txt",
        thresholds=[500, 1000, 2000, 4000])

    # copy pose estimates of the final iteration to output file
    final_pose_file = opt.results_folder / f"poses_{iteration_id}.txt"
    shutil.copy(final_pose_file, final_pose_file.parent / f"poses_final.txt")

    stats_report = "Time (min) | Iterations | Reg. Rate @500 | @1000 | @2000 | @4000\n"
    stats_report += f"{reconstruction_time / 60:.1f} " \
                    f"{iteration} " \
                    f"{registration_rates[0] * 100:.1f}% " \
                    f"{registration_rates[1] * 100:.1f}% " \
                    f"{registration_rates[2] * 100:.1f}% " \
                    f"{registration_rates[3] * 100:.1f}%\n"

    _logger.info(stats_report)
