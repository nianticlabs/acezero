# Copyright © Niantic, Inc. 2022.

import os

os.environ["MKL_NUM_THREADS"] = "1"  # noqa: E402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa: E402

import dataset_io
import logging

import random
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from ace_util import get_pixel_grid, to_homogeneous
from ace_loss import ReproLoss
from ace_network import Regressor
from ace_schedule import ScheduleACE
from refine_calibration import CalibrationRefiner
from refine_poses import PoseRefiner
from dataset import CamLocDataset

from ace_visualizer import ACEVisualizer

_logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    Seed all sources of randomness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class TrainerACE:
    def __init__(self, options):
        self.log_file = None
        self.options = options

        self.device = torch.device('cuda')
        _logger.info(f"Using device for training: {self.device}")
        self.training_buffer_device = torch.device('cpu') if self.options.training_buffer_cpu else self.device
        _logger.info(f"ACE feature buffer device: {self.training_buffer_device}")

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
        # torch.backends.cuda.matmul.allow_tf32 = False

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        # torch.backends.cudnn.allow_tf32 = False

        # Setup randomness for reproducibility.
        self.base_seed = self.options.base_seed
        _logger.info(f"Setting random seed to {self.base_seed}")
        set_seed(self.base_seed)

        # Used to generate batch indices.
        self.batch_generator = torch.Generator()
        self.batch_generator.manual_seed(self.base_seed + 1023)

        # Dataloader generator, used to seed individual workers by the dataloader.
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(self.base_seed + 511)

        # Generator used to sample random features (runs on the GPU).
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(self.base_seed + 4095)

        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = torch.Generator()
        self.training_generator.manual_seed(self.base_seed + 8191)

        # Initialize iteration and epoch counters.
        self.iteration = 0
        self.epoch = 0
        # Keep track of the training time.
        self.training_start = None
        # Number of workers for data loading.
        self.num_data_loader_workers = self.options.num_data_workers

        # Determine whether we will generate ground truth scene coordinate from depth maps.
        self.use_depth = (self.options.use_pose_seed >= 0) or (self.options.depth_files is not None)

        # Disable multi-threaded data loading in case we have to predict depth maps in the dataset class.
        if self.use_depth and self.options.depth_files is None:
            _logger.info("Disabling multi-threaded data loading because we cannot run multiple depth inference passes "
                         "simultaneously.")
            self.num_data_loader_workers = 0

        # Create dataset.
        self.dataset = CamLocDataset(
            rgb_files=self.options.rgb_files,
            pose_files=self.options.pose_files,
            ace_pose_file=self.options.use_ace_pose_file,
            ace_pose_file_conf_threshold=self.options.ace_pose_file_conf_threshold,
            pose_seed=self.options.use_pose_seed,
            depth_files=self.options.depth_files,
            use_depth=self.use_depth,
            augment=self.options.use_aug,
            aug_rotation=self.options.aug_rotation,
            aug_scale_max=self.options.aug_scale,
            aug_scale_min=1 / self.options.aug_scale,
            image_short_size=self.options.image_resolution,
            use_half=self.options.use_half,
            use_heuristic_focal_length=self.options.use_heuristic_focal_length,
        )

        if self.options.use_external_focal_length is not None:
            self.dataset.set_external_focal_length(self.options.use_external_focal_length)

        _logger.info("Loaded training scan from: {} -- {} images, mean: {:.2f} {:.2f} {:.2f}".format(
            self.options.rgb_files,
            len(self.dataset),
            self.dataset.mean_cam_center[0],
            self.dataset.mean_cam_center[1],
            self.dataset.mean_cam_center[2]))

        # Create network using the state dict of the pretrained encoder.
        encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")
        _logger.info(f"Loaded pretrained encoder from: {self.options.encoder_path}")

        if self.options.load_weights is None:

            self.regressor = Regressor.create_from_encoder(
                encoder_state_dict,
                mean=self.dataset.mean_cam_center,
                num_head_blocks=self.options.num_head_blocks,
                use_homogeneous=self.options.use_homogeneous
            )

        else:
            head_state_dict = torch.load(self.options.load_weights, map_location="cpu")
            _logger.info(f"Loaded head weights from: {self.options.load_weights}")

            # Create regressor.
            self.regressor = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)

        self.regressor = self.regressor.to(self.device)
        self.regressor.train()

        # Setup optimization parameters and learning rate scheduler.
        self.training_scheduler = ScheduleACE(self.regressor, self.options)

        # Generate grid of target reprojection pixel positions.
        pixel_grid_2HW = get_pixel_grid(self.regressor.OUTPUT_SUBSAMPLE)
        self.pixel_grid_2HW = pixel_grid_2HW.to(self.device)

        # print loss every n iterations, and (optionally) write a visualisation frame
        self.iterations_output = self.options.iterations_output

        # Setup reprojection loss function.
        self.repro_loss = ReproLoss(
            total_iterations=self.options.iterations,
            soft_clamp=self.options.repro_loss_soft_clamp,
            soft_clamp_min=self.options.repro_loss_soft_clamp_min,
            type=self.options.repro_loss_type,
            circle_schedule=(self.options.repro_loss_schedule == 'circle')
        )

        # Will be filled at the beginning of the training process.
        self.training_buffer = None
        self.training_buffer_size = self.options.max_training_buffer_size

        # Setup pose refinement.
        self.pose_refiner = PoseRefiner(dataset=self.dataset, device=self.device, options=self.options)

        # Setup calibration refinement.
        if self.options.refine_calibration:
            self.K_optimizer = CalibrationRefiner(
                dataset=self.dataset,
                learning_rate=self.options.refine_calibration_lr,
                device=self.device
            )
        else:
            self.K_optimizer = None

        # Generate video of training process
        if self.options.render_visualization:
            target_path = self.options.render_target_path
            os.makedirs(target_path, exist_ok=True)
            state_file = self.options.output_map_file.stem + "_mapping.pkl"

            self.ace_visualizer = ACEVisualizer(
                target_path,
                self.options.render_flipped_portrait,
                self.options.render_map_depth_filter,
                mapping_vis_error_threshold=self.options.render_map_error_threshold,
                mapping_state_file_name=state_file,
                marker_size=self.options.render_marker_size)
        else:
            self.ace_visualizer = None

    def train(self):
        """
        Main training method.

        Fills a feature buffer using the pretrained encoder and subsequently trains a scene coordinate regression head.
        """

        if self.ace_visualizer is not None:
            # Setup the ACE render pipeline.
            self.ace_visualizer.setup_mapping_visualisation(
                poses=self.dataset.poses,
                frame_count=100,
                camera_z_offset=self.options.render_camera_z_offset,
                existing_vis_buffer=self.options.use_existing_vis_buffer
            )

        creating_buffer_time = 0.
        training_time = 0.

        self.training_start = time.time()

        # Create training buffer.
        buffer_start_time = time.time()
        self.create_training_buffer()
        buffer_end_time = time.time()
        creating_buffer_time += buffer_end_time - buffer_start_time
        _logger.info(f"Filled training buffer in {buffer_end_time - buffer_start_time:.1f}s.")

        # Create pose buffer for camera pose refinement.
        self.pose_refiner.create_pose_buffer()
        _logger.info(f"Created pose buffer for camera pose refinement.")

        # Setup a training log - first derive a log file from the map file
        base_file_name, _ = os.path.splitext(self.options.output_map_file)
        self.log_file = open(base_file_name + '.txt', 'w')

        # Train the regression head.
        while True:

            self.epoch += 1

            epoch_start_time = time.time()
            continue_training = self.run_epoch()
            training_time += time.time() - epoch_start_time

            if not continue_training:
                break

        # Save trained model.
        self.save_model()
        self.save_poses()
        self.log_file.close()

        end_time = time.time()
        _logger.info(f'Done without errors. '
                     f'Creating buffer time: {creating_buffer_time:.1f} seconds. '
                     f'Training time: {training_time:.1f} seconds. '
                     f'Total time: {end_time - self.training_start:.1f} seconds.')

        if self.ace_visualizer is not None:
            # Finalize the rendering by animating the fully trained map.
            vis_dataset = CamLocDataset(
                rgb_files=self.options.rgb_files,
                pose_files=self.options.pose_files,
                ace_pose_file=self.options.use_ace_pose_file,
                ace_pose_file_conf_threshold=self.options.ace_pose_file_conf_threshold,
                pose_seed=self.options.use_pose_seed,
                augment=False,
                image_short_size=self.options.image_resolution,
                use_half=self.options.use_half,
                use_heuristic_focal_length=self.options.use_heuristic_focal_length,
            )
            # Set focal length, either from optimizer or use external value, otherwise dataset value will be used
            if self.K_optimizer is not None:
                vis_dataset.set_external_focal_length(float(self.K_optimizer.get_focal_length()))
            elif self.options.use_external_focal_length is not None:
                vis_dataset.set_external_focal_length(self.options.use_external_focal_length)

            vis_dataset_loader = torch.utils.data.DataLoader(
                vis_dataset,
                shuffle=False,  # Process data in order for a growing effect later when rendering
                num_workers=self.num_data_loader_workers,
                timeout=60 if self.num_data_loader_workers > 0 else 0)

            self.ace_visualizer.finalize_mapping(
                self.regressor,
                vis_dataset_loader,
                self.pose_refiner.get_all_current_poses(),
                self.pose_refiner.get_all_original_poses()
            )

    def create_training_buffer(self):
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        # Sampler.
        batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator),
                                             batch_size=1,
                                             drop_last=False)

        # Used to seed workers in a reproducible manner.
        def seed_worker(worker_id):
            # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
            # the dataloader generator.
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Batching is handled at the dataset level (the dataset __getitem__ receives a list of indices, because we
        # need to rescale all images in the batch to the same size).
        training_dataloader = DataLoader(dataset=self.dataset,
                                         sampler=batch_sampler,
                                         batch_size=None,
                                         worker_init_fn=seed_worker,
                                         generator=self.loader_generator,
                                         pin_memory=True,
                                         num_workers=self.num_data_loader_workers,
                                         persistent_workers=self.num_data_loader_workers > 0,
                                         timeout=60 if self.num_data_loader_workers > 0 else 0,
                                         )

        _logger.info("Starting creation of the training buffer.")

        # Calculate size of the dataset buffer
        training_buffer_size = self.options.max_dataset_passes * len(self.dataset) * self.options.samples_per_image
        training_buffer_size = min(training_buffer_size, self.options.max_training_buffer_size)

        # Create a training buffer that lives either on the GPU or in main memory.
        self.training_buffer = {
            'features': torch.empty((training_buffer_size, self.regressor.feature_dim),
                                    dtype=(torch.float32, torch.float16)[self.options.use_half], device=self.training_buffer_device),
            'target_px': torch.empty((training_buffer_size, 2), dtype=torch.float32, device=self.training_buffer_device),
            'aug_poses_inv': torch.empty((training_buffer_size, 3, 4), dtype=torch.float32, device=self.training_buffer_device),
            'poses_inv': torch.empty((training_buffer_size, 4, 4), dtype=torch.float32, device=self.training_buffer_device),
            'intrinsics': torch.empty((training_buffer_size, 3, 3), dtype=torch.float32, device=self.training_buffer_device),
            'intrinsics_inv': torch.empty((training_buffer_size, 3, 3), dtype=torch.float32, device=self.training_buffer_device),
            'target_crds': torch.empty((training_buffer_size, 3), dtype=torch.float32, device=self.training_buffer_device),
            'pose_idx': torch.empty((training_buffer_size, 1), dtype=torch.int16, device=self.training_buffer_device),
        }

        # Features are computed in evaluation mode.
        self.regressor.eval()

        # The encoder is pretrained, so we don't compute any gradient.
        with torch.no_grad():
            # Iterate until the training buffer is full.
            buffer_idx = 0
            dataset_passes = 0

            while buffer_idx < self.options.max_training_buffer_size and dataset_passes < self.options.max_dataset_passes:
                dataset_passes += 1
                for image_B1HW, image_mask_B1HW, pose_inv_B44, aug_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, target_crds_B3HW, _, idx_B in training_dataloader:

                    # Copy to device.
                    image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                    image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                    aug_pose_inv_B44 = aug_pose_inv_B44.to(self.device, non_blocking=True)
                    pose_inv_B44 = pose_inv_B44.to(self.device, non_blocking=True)
                    intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
                    intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)
                    target_crds_B3HW = target_crds_B3HW.to(self.device, non_blocking=True)
                    idx_B = idx_B.to(self.device, non_blocking=True)

                    # Compute image features.
                    with autocast(enabled=self.options.use_half):
                        features_BCHW = self.regressor.get_features(image_B1HW)

                    # Dimensions after the network's downsampling.
                    B, C, H, W = features_BCHW.shape

                    # The image_mask needs to be downsampled to the actual output resolution and cast to bool.
                    image_mask_B1HW = TF.resize(image_mask_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)
                    image_mask_B1HW = image_mask_B1HW.bool()

                    # If the current mask has no valid pixels, continue.
                    if image_mask_B1HW.sum() == 0:
                        continue

                    # Create a tensor with the pixel coordinates of every feature vector.
                    pixel_positions_B2HW = self.pixel_grid_2HW[:, :H, :W].clone()  # It's 2xHxW (actual H and W) now.
                    pixel_positions_B2HW = pixel_positions_B2HW[None]  # 1x2xHxW
                    pixel_positions_B2HW = pixel_positions_B2HW.expand(B, 2, H, W)  # Bx2xHxW

                    # Bx4x4 -> Nx3x4 (for each image, repeat pose per feature)
                    aug_pose_inv = aug_pose_inv_B44[:, :3]
                    aug_pose_inv = aug_pose_inv.unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)

                    # Bx4x4 -> Nx4x4 (for each image, repeat pose per feature)
                    pose_inv = pose_inv_B44.unsqueeze(1).expand(B, H * W, 4, 4).reshape(-1, 4, 4)

                    # B -> N (for each image, repeat pose index)
                    idx_B = idx_B.expand(B, H * W).reshape(-1, 1)

                    # Bx3x3 -> Nx3x3 (for each image, repeat intrinsics per feature)
                    intrinsics = intrinsics_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)
                    intrinsics_inv = intrinsics_inv_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)

                    def normalize_shape(tensor_in):
                        """Bring tensor from shape BxCxHxW to NxC"""
                        return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

                    batch_data = {
                        'features': normalize_shape(features_BCHW),
                        'target_px': normalize_shape(pixel_positions_B2HW),
                        'aug_poses_inv': aug_pose_inv,
                        'poses_inv': pose_inv,
                        'intrinsics': intrinsics,
                        'intrinsics_inv': intrinsics_inv,
                        'target_crds': normalize_shape(target_crds_B3HW),
                        'pose_idx': idx_B
                    }

                    # Turn image mask into sampling weights (all equal).
                    image_mask_B1HW = image_mask_B1HW.float()
                    image_mask_N1 = normalize_shape(image_mask_B1HW)

                    # Over-sample according to image mask.
                    features_to_select = self.options.samples_per_image * B
                    features_to_select = min(features_to_select, self.options.max_training_buffer_size - buffer_idx)

                    # Sample indices uniformly, with replacement.
                    sample_idxs = torch.multinomial(image_mask_N1.view(-1),
                                                    features_to_select,
                                                    replacement=True,
                                                    generator=self.sampling_generator)

                    # Select the data to put in the buffer.
                    for k in batch_data:
                        batch_data[k] = batch_data[k][sample_idxs]

                    # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
                    buffer_offset = buffer_idx + features_to_select
                    for k in batch_data:
                        # Copy to buffer, if the buffer lives on the GPU the .to() is a noop.
                        self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[k].to(self.training_buffer_device)

                    buffer_idx = buffer_offset
                    if buffer_idx >= self.options.max_training_buffer_size:
                        break

        # clipping training buffer to actual size
        self.training_buffer_size = min(buffer_idx, self.options.max_training_buffer_size)

        for k in self.training_buffer:
            self.training_buffer[k] = self.training_buffer[k][:self.training_buffer_size]

        buffer_memory = sum([v.element_size() * v.nelement() for k, v in self.training_buffer.items()])
        buffer_memory /= 1024 * 1024 * 1024

        _logger.info(f"Created buffer of {buffer_memory:.2f}GB with {dataset_passes} passes over the training data.")
        self.regressor.train()

    def run_epoch(self):
        """
        Run one epoch of training, shuffling the feature buffer and iterating over it.
        """
        # check whether training has finished, number of total iterations might have been reduced
        if self.iteration >= self.training_scheduler.max_iterations:
            return False

        # Enable benchmarking since all operations work on the same tensor size.
        torch.backends.cudnn.benchmark = True

        # Shuffle indices.
        random_indices = torch.randperm(self.training_buffer_size, generator=self.training_generator)

        # Iterate with mini batches.
        for batch_start in range(0, self.training_buffer_size, self.options.batch_size):
            batch_end = batch_start + self.options.batch_size

            # Drop last batch if not full.
            if batch_end > self.training_buffer_size:
                continue

            # Sample indices.
            random_batch_indices = random_indices[batch_start:batch_end]

            # Call the training step with the sampled features and relevant metadata.
            # If the buffer lives in main memory we also move it to the GPU, otherwise the 
            # .to() is a no-op.
            # .contiguous() is needed to for a faster forward pass through the head later. 
            # If the buffer was in main memory, .to() already makes the GPU copy contiguous, 
            # so the .contiguous() is actually a no-op.
            self.training_step(
                self.training_buffer['features'][random_batch_indices].to(self.device).contiguous(),
                self.training_buffer['target_px'][random_batch_indices].to(self.device).contiguous(),
                self.training_buffer['aug_poses_inv'][random_batch_indices].to(self.device).contiguous(),
                self.training_buffer['poses_inv'][random_batch_indices].to(self.device).contiguous(),
                self.training_buffer['intrinsics'][random_batch_indices].to(self.device).contiguous(),
                self.training_buffer['intrinsics_inv'][random_batch_indices].to(self.device).contiguous(),
                self.training_buffer['target_crds'][random_batch_indices].to(self.device).contiguous(),
                self.training_buffer['pose_idx'][random_batch_indices].to(self.device).contiguous()
            )
            self.iteration += 1

        return True

    def training_step(self, features_bC, target_px_b2, inv_aug_poses_b34, inv_poses_b44, Ks_b33, invKs_b33,
                      target_crds_b3, pose_idx):
        """
        Run one iteration of training, computing the reprojection error and minimising it.
        """

        # check whether to start cooldown
        self.training_scheduler.check_and_set_cooldown(self.iteration)

        # check whether training has finished, number of total iterations might have been reduced
        if self.iteration >= self.training_scheduler.max_iterations:
            return

        batch_size = features_bC.shape[0]
        channels = features_bC.shape[1]

        # Reshape to a "fake" BCHW shape, since it's faster to run through the network compared to the original shape.
        features_bCHW = features_bC[None, None, ...].view(-1, 16, 32, channels).permute(0, 3, 1, 2)
        with autocast(enabled=self.options.use_half):
            pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(features_bCHW)

        # Back to the original shape. Convert to float32 as well.
        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()

        # Make 3D points homogeneous so that we can easily matrix-multiply them.
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)

        # Get current estimate of poses.
        gt_poses_orig_b44 = self.pose_refiner.get_current_poses(inv_poses_b44, pose_idx.int())

        # combine augmentation poses and original camera poses
        gt_inv_poses_b34 = torch.bmm(inv_aug_poses_b34, gt_poses_orig_b44)

        # Scene coordinates to camera coordinates.
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)

        # Project scene coordinates, either using refined or original intrinsics.
        if self.K_optimizer is not None:
            refined_Ks_scaled_b33 = self.K_optimizer.get_refined_calibration_matrices(Ks_b33)
            pred_px_b31 = torch.bmm(refined_Ks_scaled_b33, pred_cam_coords_b31)
        else:
            pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)

        # Avoid division by zero.
        # Note: negative values are also clamped at +self.options.depth_min. The predicted pixel would be wrong,
        # but that's fine since we mask them out later.
        pred_px_b31[:, 2].clamp_(min=self.options.depth_min)

        # Dehomogenise.
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        # Measure reprojection error.
        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)

        #
        # Compute masks used to ignore invalid pixels.
        #
        # Predicted coordinates behind or close to camera plane.
        invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.options.depth_min
        # Very large reprojection errors.
        invalid_repro_b1 = reprojection_error_b1 > self.options.repro_loss_hard_clamp
        # Predicted coordinates beyond max distance.
        invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.options.depth_max

        # Invalid mask is the union of all these. Valid mask is the opposite.
        invalid_mask_b1 = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)

        if self.use_depth:
            # if GT scene coordinates are available, also check whether predictions are already sufficiently close
            invalid_target_crds_b1 = (
                        torch.linalg.norm(target_crds_b3 - pred_scene_coords_b31.squeeze(), dim=1) > 0.1).unsqueeze(1)
            # in the previous mask, ignore pixels without GT scene coordinates (all zeros)
            target_crds_available_b1 = (target_crds_b3.abs().sum(dim=1) > 0.00001).unsqueeze(1)

            invalid_mask_b1 = invalid_mask_b1 | (invalid_target_crds_b1 & target_crds_available_b1)

        valid_mask_b1 = ~invalid_mask_b1

        if valid_mask_b1.sum() > 0:
            # Reprojection error for all valid scene coordinates.
            valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]

            # Compute the loss for valid predictions.
            loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, self.iteration)

            batch_inliers = valid_reprojection_error_b1 < self.options.learning_rate_cooldown_trigger_px_threshold
            batch_inliers = float(batch_inliers.sum()) / batch_size
        else:
            loss_valid = 0
            batch_inliers = 0

        # Handle the invalid predictions
        if not self.use_depth:
            # generate proxy coordinate targets with constant depth assumption.
            pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
            target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)

            # Compute the distance to target camera coordinates.
            invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
            loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(
                invalid_mask_b11).sum()
        else:
            if invalid_mask_b1.sum() > 0:
                # ignore pixels without GT scene coordinates (all zeros)
                invalid_mask_b1 = invalid_mask_b1 & target_crds_available_b1

                loss_invalid = torch.linalg.norm(target_crds_b3 - pred_scene_coords_b31.squeeze(), dim=1)
                loss_invalid = loss_invalid[invalid_mask_b1.squeeze()].sum()
            else:
                loss_invalid = 0

        # Final loss is the sum of all 2.
        loss = loss_valid + loss_invalid
        loss /= batch_size

        if torch.any(torch.isnan(loss)):
            _logger.error("Aborting because of NaN loss")
            exit()

        # Set gradient buffers to zero.
        self.training_scheduler.zero_grad(set_to_none=True)
        self.pose_refiner.zero_grad(set_to_none=True)

        if self.K_optimizer is not None:
            self.K_optimizer.zero_grad()

        # Calculate gradients of loss
        self.training_scheduler.backward(loss)

        # Parameter steps.

        # Update the ACE network weights. Pass batch inliers to potentially go into cooldown.
        self.training_scheduler.step(batch_inliers)

        if self.iteration > self.options.pose_refinement_wait:
            # Only update poses after initial wait period (if set)
            self.pose_refiner.step()

        if self.K_optimizer is not None:
            # Only update calibration if it is being refined
            self.K_optimizer.step()

        if self.iteration % self.iterations_output == 0:
            # Print status.
            time_since_start = time.time() - self.training_start

            _logger.info(f'Iteration: {self.iteration:6d}|{self.training_scheduler.max_iterations:6d} / '
                         f'Epoch {self.epoch:03d}, '
                         f'Loss: {loss:.1f}, '
                         f'Batch inliers ({self.options.learning_rate_cooldown_trigger_px_threshold}px): '
                         f'{batch_inliers * 100:.1f}%, '
                         f'Time: {time_since_start:.0f}s')

            # Print statistics about the pose refinement
            poses_original = self.pose_refiner.get_all_original_poses()
            poses_updated = self.pose_refiner.get_all_current_poses()

            pose_pos_distances = torch.linalg.norm(poses_updated[:, :, 3].cpu() - poses_original[:, :, 3].cpu(), dim=1)

            _logger.info(f'Poses moved by: '
                         f'Avg={pose_pos_distances.mean() * 100:.1f}cm, '
                         f'Min={pose_pos_distances.min() * 100:.1f}cm, '
                         f'Max={pose_pos_distances.max() * 100:.1f}cm')

            # write the main information to the log file
            log_str = f"{self.iteration} {time_since_start} {loss} {batch_inliers} " \
                      f"{pose_pos_distances.mean()} {pose_pos_distances.min()} {pose_pos_distances.max()}"

            if self.K_optimizer is not None:
                focal_length = float(self.K_optimizer.get_focal_length())
                _logger.info(f"Current Focal Length: {focal_length:.1f}")
                log_str += f" {focal_length}"

            self.log_file.write(log_str + "\n")

            if self.ace_visualizer is not None:
                vis_scene_coords = pred_scene_coords_b31.detach().cpu().squeeze().numpy()
                vis_errors = reprojection_error_b1.detach().cpu().squeeze().numpy()
                self.ace_visualizer.render_mapping_frame(vis_scene_coords, vis_errors, poses_updated, poses_original,
                                                         self.iteration)

    def save_model(self):
        """
        Save the trained model to disk.
        """
        # NOTE: This would save the whole regressor (encoder weights included) in full precision floats (~30MB).
        # torch.save(self.regressor.state_dict(), self.options.output_map_file)

        # This saves just the head weights as half-precision floating point numbers for a total of ~4MB, as mentioned
        # in the paper. The scene-agnostic encoder weights can then be loaded from the pretrained encoder file.
        head_state_dict = self.regressor.heads.state_dict()
        for k, v in head_state_dict.items():
            head_state_dict[k] = head_state_dict[k].half()
        torch.save(head_state_dict, self.options.output_map_file)
        _logger.info(f"Saved trained head weights to: {self.options.output_map_file}")

    def save_poses(self):
        """
        Save the refined poses and focal lengths to disk.
        """

        # get current estimate of all poses
        output_poses = self.pose_refiner.get_all_current_poses()

        # output file
        pose_file = self.options.output_map_file.parent / f"poses_{self.options.output_map_file.stem}_preliminary.txt"

        with open(pose_file, 'w') as f:

            for pose_idx in range(output_poses.shape[0]):

                pose_34 = output_poses[pose_idx].cpu().detach().numpy()

                # We do not know the confidence of the pose after refinement, so we set it to infinity
                confidence = float('inf')

                # Get the current focal length estimate for this image
                if self.K_optimizer is not None:
                    focal_length = float(self.K_optimizer.get_focal_length())
                else:
                    # if focal length is not being refined, the dataset contains the correct value for each image
                    # either from the calibration files or from the heuristic
                    focal_length = self.dataset.get_focal_length(pose_idx)

                # Write to file
                dataset_io.write_pose_to_pose_file(f, rgb_file=self.dataset.rgb_files[pose_idx], pose=pose_34,
                                                   confidence=confidence, focal_length=focal_length)

        _logger.info(f"Saved refined poses to: {pose_file}")
