import torch
from torch import optim
import numpy as np


class CalibrationRefiner:
    """
    Handles refinement of per-image calibration information during ACE training.
    """

    def __init__(self, dataset, learning_rate, device):
        # check whether focal length are all identical, we only support one focal length for all images
        focal_lengths = [dataset.get_focal_length(i) for i in range(len(dataset))]
        if not np.allclose(focal_lengths, focal_lengths[0]):
            raise ValueError("All images must have the same focal length for calibration refinement")

        # initialise intrinsics
        self.focal_length_init = focal_lengths[0]

        # this is the main learnable parameter, it is a relative scale factor to the focal length
        self.global_f = torch.zeros(1)
        self.global_f = self.global_f.to(device)
        self.global_f = self.global_f.detach().requires_grad_()

        # initialise optimizer
        self.optimizer = optim.AdamW([self.global_f], lr=learning_rate)

    def get_focal_length(self):
        """
        Get the current estimate of the focal length.
        """
        return (1 + self.global_f) * self.focal_length_init

    def get_refined_calibration_matrices(self, Ks_b33):
        """
        Get the refined calibration matrices, based on the initial calibration matrices and the refined focal length.

        @param Ks_b33: initial calibration matrices, shape (B, 3, 3)
        """

        # set current estimate of focal length in the original image scale
        refined_Ks_22 = torch.eye(2, 2).cuda() * self.get_focal_length()
        refined_Ks_b22 = refined_Ks_22.unsqueeze(0).expand(Ks_b33.shape[0], -1, -1)

        # scale the refined intrinsics by the augmentation scale factor, inferred from the initial calibration matrices
        aug_scales = Ks_b33[:, 0, 0] / self.focal_length_init
        refined_Ks_scaled_b22 = refined_Ks_b22 * aug_scales.detach()[:, None, None]

        # overwrite the focal length in the original calibration matrices with the refined focal length
        refined_Ks_scaled_b33 = Ks_b33.clone().detach()
        refined_Ks_scaled_b33[:, :2, :2] = refined_Ks_scaled_b22

        return refined_Ks_scaled_b33

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

