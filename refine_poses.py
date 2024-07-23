# Copyright © Niantic, Inc. 2022.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import roma

_logger = logging.getLogger(__name__)


class PoseNetwork(nn.Module):
    """
    MLP network predicting a pose update.
    It takes 12 inputs (3x4 pose) and predicts 12 values, e.g. used as additive offsets.
    """

    def __init__(self, num_head_blocks, channels=512):
        super(PoseNetwork, self).__init__()

        self.in_channels = 12
        self.head_channels = channels  # Hardcoded.

        # We may need a skip layer if the number of features output by the encoder is different.
        self.head_skip = nn.Identity() if self.in_channels == self.head_channels else nn.Conv2d(self.in_channels,
                                                                                                self.head_channels, 1,
                                                                                                1, 0)

        self.conv1 = nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.conv3 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        self.res_blocks = []

        for block in range(num_head_blocks):
            self.res_blocks.append((
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
            ))

            super(PoseNetwork, self).add_module(str(block) + 'c0', self.res_blocks[block][0])
            super(PoseNetwork, self).add_module(str(block) + 'c1', self.res_blocks[block][1])
            super(PoseNetwork, self).add_module(str(block) + 'c2', self.res_blocks[block][2])

        self.fc1 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.fc2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.fc3 = nn.Conv2d(self.head_channels, 12, 1, 1, 0)

    def forward(self, res):

        x = F.relu(self.conv1(res))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        res = self.head_skip(res) + x

        for res_block in self.res_blocks:
            x = F.relu(res_block[0](res))
            x = F.relu(res_block[1](x))
            x = F.relu(res_block[2](x))

            res = res + x

        pose_update = F.relu(self.fc1(res))
        pose_update = F.relu(self.fc2(pose_update))
        pose_update = self.fc3(pose_update)

        return pose_update


class PoseRefiner:
    """
    Handles refinement of per-image pose information during ACE training.

    Support three variants.
    1. 'none': no pose refinement
    2. 'naive': back-prop to poses directly
    3. 'mlp': use a network to predict pose updates
    """

    def __init__(self, dataset, device, options):

        self.dataset = dataset
        self.device = device

        # set refinement strategy
        if options.pose_refinement not in ['none', 'naive', 'mlp']:
            raise ValueError(f"Pose refinement strategy {options.pose_refinement} not supported")
        self.refinement_strategy = options.pose_refinement

        # set options
        self.learning_rate = options.pose_refinement_lr
        self.update_weight = options.pose_refinement_weight
        self.orthonormalization = options.refinement_ortho

        # pose buffer for current estimate of refined poses
        self.pose_buffer = None
        # pose buffer for original poses
        self.pose_buffer_orig = None
        # network predicting pose updates (depending on the optimization strategy)
        self.pose_network = None
        # optimizer for pose updates
        self.pose_optimizer = None

    def create_pose_buffer(self):
        """
        Populate internal pose buffers and set up the pose optimization strategy.
        """
        self.pose_buffer_orig = torch.zeros(len(self.dataset), 3, 4)

        # fill pose buffer with inverse poses (camera to world)
        for pose_idx, pose in enumerate(self.dataset.poses):
            self.pose_buffer_orig[pose_idx] = pose.inverse().clone()[:3]
        self.pose_buffer = self.pose_buffer_orig.contiguous().to(self.device, non_blocking=True)

        # set the pose optimization strategy
        if self.refinement_strategy == 'none':
            # will keep original poses
            pass
        elif self.refinement_strategy == 'naive':
            # back-prop to poses directly
            self.pose_buffer = self.pose_buffer.detach().requires_grad_()
            self.pose_optimizer = optim.AdamW([self.pose_buffer], lr=self.learning_rate)
        else:
            # use small network to predict pose updates
            self.pose_network = PoseNetwork(0, 128)
            self.pose_network = self.pose_network.to(self.device)
            self.pose_network.train()
            self.pose_optimizer = optim.AdamW(self.pose_network.parameters(), lr=self.learning_rate)

    def _orthonormalize_poses(self, poses_b33):
        """
        Orthonormalize the rotation part of the poses.

        @param poses_bxx: poses to orthonormalize, shape (b, 3, 3) where x is 3 or 4
        """
        B, H, W = poses_b33.shape
        if H != 3 or W != 3:
            raise ValueError("Can only orthonormalize 3x3 rotation matrices")

        if self.orthonormalization == 'none':
            return poses_b33
        elif self.orthonormalization == 'gram-schmidt':
            return roma.special_gramschmidt(poses_b33)
        else:
            return roma.special_procrustes(poses_b33)

    def _predict_pose_updates(self, poses_b34):
        """
        Predict pose updates with the current state of the network.
        Returns rotations and translations separately to not break the PyTorch autograd graph.

        @param poses_b34: poses to predict updates for, shape (b, 3, 4)

        @return tuple: updated rotation matrices bx3x3 and translation vectors bx3x1
        """

        if self.pose_network is None:
            raise ValueError("Pose network not initialized")

        # get deltas from network
        poses_b1211 = poses_b34.view(-1, 12, 1, 1)
        pose_update_b1211 = self.pose_network(poses_b1211)

        # combine deltas with original poses
        updated_poses_b34 = (poses_b1211 + self.update_weight * pose_update_b1211).view(-1, 3, 4)

        # orthonormalize rotation part
        updated_rots_b33 = self._orthonormalize_poses(updated_poses_b34[:, :3, :3])
        updated_trans_b31 = updated_poses_b34[:, :3, 3]

        return updated_rots_b33, updated_trans_b31

    def get_all_original_poses(self):
        """
        Get all original poses.
        """
        return self.pose_buffer_orig.clone()

    def get_all_current_poses(self):
        """
        Get all current estimates of refined poses.
        """
        if self.refinement_strategy == 'none':
            # just return original poses
            return self.pose_buffer_orig.clone()
        elif self.refinement_strategy == 'naive':
            # return current state of the pose buffer
            current_poses = self.pose_buffer.clone()
            # orthonormalize rotation part
            current_poses[:, :3, :3] = self._orthonormalize_poses(current_poses[:, :3, :3])
            return current_poses
        else:
            # predict pose updates with current state of the network
            with torch.no_grad():
                # return current state of the pose buffer
                output_poses = self.pose_buffer.clone()

                # predict current poses
                current_rots_b33, current_trans_b31 = self._predict_pose_updates(output_poses)

                # put back together
                output_poses[:, :3, :3] = current_rots_b33
                output_poses[:, :3, 3] = current_trans_b31

                return output_poses

    def get_current_poses(self, original_poses_b44, original_poses_indices):
        """
        Get current estimates of refined poses for a subset of the original poses.

        @param original_poses_b44: original poses, shape (b, 4, 4)
        @param original_poses_indices: indices of the original poses in the dataset
        """
        output_poses_b44 = original_poses_b44.clone()

        if self.refinement_strategy == 'none':
            # just return original poses
            return output_poses_b44
        elif self.refinement_strategy == 'naive':
            # get current state of the poses from buffer
            current_poses_b34 = self.pose_buffer[original_poses_indices].squeeze()
            # orthonormalize rotation part
            current_rots_b33 = self._orthonormalize_poses(current_poses_b34[:, :3, :3])

            # put back together
            output_poses_b44[:, :3, :3] = current_rots_b33
            output_poses_b44[:, :3, 3] = current_poses_b34[:, :3, 3]

            return output_poses_b44

        else:
            # predict pose updates with current state of the network
            predicted_rots_b33, predicted_trans_b31 = self._predict_pose_updates(original_poses_b44[:, :3])

            # make current poses 4x4 by writing them back to the input poses
            output_poses_b44[:, :3, :3] = predicted_rots_b33
            output_poses_b44[:, :3, 3] = predicted_trans_b31

            return output_poses_b44

    def zero_grad(self, set_to_none=False):
        if self.pose_optimizer is not None:
            self.pose_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        if self.pose_optimizer is not None:
            self.pose_optimizer.step()
