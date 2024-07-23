#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.

import argparse
import logging
from distutils.util import strtobool
from pathlib import Path

from ace_trainer import TrainerACE


def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':

    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Fast training of a scene coordinate regression network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('rgb_files', type=str, help="Glob pattern for RGB files, e.g. 'datasets/scene/*.jpg'")

    parser.add_argument('output_map_file', type=Path,
                        help='target file for the trained network')

    parser.add_argument('--base_seed', type=int, default=2089,
                        help='seed to control randomness')

    # === Data definition ==============================================================================================

    parser.add_argument('--pose_files', type=str, default=None,
                        help="Glob pattern for pose files, e.g. 'datasets/scene/*.txt', each file is assumed to "
                             "contain a 4x4 pose matrix, cam2world, correspondence with rgb files is assumed by "
                             "alphabetical order; None: provide poses via use_ace_pose_file or use_pose_seed")

    parser.add_argument('--use_ace_pose_file',  type=Path, default=None,
                        help='ACE pose file containing mapping images to use, their poses and focal lengths;'
                             'None: provide poses via pose_files or use_pose_seed')

    parser.add_argument('--ace_pose_file_conf_threshold', type=float, default=1000,
                        help='Consider only files with larger confidence in ACE pose file')

    parser.add_argument('--use_pose_seed', type=float, default=-1,
                        help='use a single image with identity pose as seed, '
                             'float value [0-1] represents image ID relative to dataset size, -1: do not use seed')

    parser.add_argument('--depth_files', type=str, default=None,
                        help="Glob pattern for depth files, e.g. 'datasets/scene/*.png', each file is assumed to "
                             "contain depth in millimeters, correspondence with rgb files is assumed by "
                             "alphabetical order, None: Don't use depth.")

    parser.add_argument('--refine_calibration', type=_strtobool, default=False,
                        help='Refine calibration parameters.')

    parser.add_argument('--refine_calibration_lr', type=float, default=0.001,
                        help='Learning rate for refining calibration parameters.')

    parser.add_argument('--use_heuristic_focal_length', type=_strtobool, default=False,
                        help="Focal length set to 70% of image diagonal. Recommended to activate refine_calibration."
                             "If False, use_external_focal_length or use_ace_pose_file must be set.")

    parser.add_argument('--use_external_focal_length', type=float, default=None,
                        help="Set external focal length value. Can be combined with refine_calibration."
                             "If None, use_heuristic_focal_length or use_ace_pose_file must be set.")

    parser.add_argument('--image_resolution', type=int, default=480,
                        help='base image resolution (px length of shortest side), will rescale images to this')

    parser.add_argument('--num_data_workers', type=int, default=12,
                        help='number of data loading workers, set according to the number of available CPU cores')

    # === Network definition ===========================================================================================

    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--load_weights', type=Path, help='path to initialised network weights', default=None)

    parser.add_argument('--num_head_blocks', type=int, default=1,
                        help='depth of the regression head, defines the map size')

    parser.add_argument('--use_half', type=_strtobool, default=True,
                        help='train with half precision')

    parser.add_argument('--use_homogeneous', type=_strtobool, default=True,
                        help='predict homogeneous scene coordinates')

    # === Learning rate schedule =======================================================================================

    parser.add_argument('--learning_rate_min', type=float, default=0.0005,
                        help='lowest learning rate of 1 cycle scheduler')

    parser.add_argument('--learning_rate_max', type=float, default=0.005,
                        help='highest learning rate of 1 cycle scheduler')

    parser.add_argument('--learning_rate_schedule', type=str, default="circle",
                        choices=["circle", "constant", "1cyclepoly"],
                        help='circle: move from min to max to min, constant: stay at min, '
                             '1cyclepoly: linear approximation of 1cycle')

    parser.add_argument('--learning_rate_warmup_iterations', type=int, default=1000,
                        help='length of the warmup period')

    parser.add_argument('--learning_rate_warmup_learning_rate', type=float, default=0.0005,
                        help='start learning rate of 1cycle poly')

    parser.add_argument('--learning_rate_cooldown_iterations', type=int, default=5000,
                        help='length of the cooldown period')

    parser.add_argument('--learning_rate_cooldown_trigger_px_threshold', type=int, default=10,
                        help='inlier threshold for early cool down criterium')

    parser.add_argument('--learning_rate_cooldown_trigger_percent_threshold', type=float, default=0.7,
                        help='min percentage of inliers for early cool down')

    # === ACE training buffer ==========================================================================================

    parser.add_argument('--max_training_buffer_size', type=int, default=8000000,
                        help='number of patches in the training buffer')

    parser.add_argument('--max_dataset_passes', type=int, default=10,
                        help='max number of repetition of mapping images (with different augmentations)')

    parser.add_argument('--samples_per_image', type=int, default=1024,
                        help='number of patches drawn from each image when creating the buffer')
    
    parser.add_argument('--training_buffer_cpu', type=_strtobool, default=False, 
                        help='store training buffer on CPU memory instead of GPU, '
                        'this allows running ACE0 on smaller GPUs, but is slower')

    # === Optimization parameters ======================================================================================

    parser.add_argument('--batch_size', type=int, default=5120,
                        help='number of patches for each parameter update (has to be a multiple of 512)')

    parser.add_argument('--iterations', type=int, default=25000,
                        help='number of runs through the training buffer')

    parser.add_argument('--iterations_output', type=int, default=300,
                        help='print training statistics every n iterations, also render_visualization frame frequency')

    # === Loss Definition ==============================================================================================

    parser.add_argument('--repro_loss_hard_clamp', type=int, default=1000,
                        help='hard clamping threshold for the reprojection losses')

    parser.add_argument('--repro_loss_soft_clamp', type=int, default=50,
                        help='soft clamping threshold for the reprojection losses')

    parser.add_argument('--repro_loss_soft_clamp_min', type=int, default=1,
                        help='minimum value of the soft clamping threshold when using a schedule')

    parser.add_argument('--repro_loss_type', type=str, default="dyntanh",
                        choices=["l1", "l1+sqrt", "l1+log", "tanh", "dyntanh"],
                        help='Loss function on the reprojection error. Dyn varies the soft clamping threshold')

    parser.add_argument('--repro_loss_schedule', type=str, default="circle", choices=['circle', 'linear'],
                        help='How to decrease the softclamp threshold during training, circle is slower first')

    parser.add_argument('--depth_min', type=float, default=0.1,
                        help='enforce minimum depth of network predictions')

    parser.add_argument('--depth_target', type=float, default=10,
                        help='default depth to regularize training')

    parser.add_argument('--depth_max', type=float, default=1000,
                        help='enforce maximum depth of network predictions')

    # === Data augmentation ============================================================================================

    parser.add_argument('--use_aug', type=_strtobool, default=True,
                        help='Use any augmentation.')

    parser.add_argument('--aug_rotation', type=int, default=15,
                        help='max inplane rotation angle')

    parser.add_argument('--aug_scale', type=float, default=1.5,
                        help='max scale factor')

    # === Visualisation parameters =====================================================================================

    # Params for the visualization. If enabled, it will slow down training considerably. But you get a nice video :)
    parser.add_argument('--render_visualization', type=_strtobool, default=False,
                        help='create a video of the mapping process')

    parser.add_argument('--render_target_path', type=Path, default='renderings',
                        help='target folder for renderings, visualizer will create a subfolder with the map name')

    parser.add_argument('--use_existing_vis_buffer', type=Path, default=None,
                        help='continue from existing visualization state')

    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False,
                        help='flag for wayspots dataset where images are sideways portrait')

    parser.add_argument('--render_map_error_threshold', type=int, default=10,
                        help='reprojection error threshold for the visualisation in px')

    parser.add_argument('--render_map_depth_filter', type=int, default=100,
                        help='to clean up the ACE point cloud remove points too far away')

    parser.add_argument('--render_camera_z_offset', type=int, default=4,
                        help='zoom out of the scene by moving render camera backwards, in meters')

    parser.add_argument('--render_marker_size', type=float, default=0.03,
                        help='size of the camera frustums in the visualization')

    # === Pose refinement parameters ===================================================================================

    parser.add_argument('--pose_refinement', type=str, default='none', choices=['none', 'naive', 'mlp'],
                        help='refine poses with a neural network (mlp) or by back-propagation to poses (naive)')

    parser.add_argument('--pose_refinement_weight', type=float, default=0.1,
                        help='weight to scale the refiner pose updates, '
                             'mainly to reduce the impact of random updates in the beginning of the optimization')

    parser.add_argument('--pose_refinement_wait', type=int, default=0,
                        help='start pose refinement after n iterations, can increase stability')

    parser.add_argument('--pose_refinement_lr', type=float, default=0.001,
                        help='learning rate for the pose refinement')

    parser.add_argument('--refinement_ortho', type=str, default="gram-schmidt", choices=['gram-schmidt', 'procrustes'],
                        help='orthogonalization method for pose rotations after pose updates')


    options = parser.parse_args()

    if options.use_pose_seed < 0 and options.use_ace_pose_file is None and options.pose_files is None:
        raise ValueError("Either use_pose_seed or use_ace_pose_file or pose_files has to be set.")

    if not options.use_heuristic_focal_length and options.use_external_focal_length is None \
            and options.use_ace_pose_file is None:
        raise ValueError("Either use_heuristic_focal_length or use_external_focal_length "
                         "or use_ace_pose_file has to be set.")

    trainer = TrainerACE(options)
    trainer.train()
