# Copyright © Niantic, Inc. 2022.

import numpy as np
import torch


def weighted_tanh(repro_errs, weight):
    return weight * torch.tanh(repro_errs / weight).sum()


class ReproLoss:
    """
    Compute per-pixel reprojection loss using different configurable approaches.

    - tanh:     tanh loss with a constant scale factor given by the `soft_clamp` parameter (when a pixel's reprojection
                error is equal to `soft_clamp`, its loss is equal to `soft_clamp * tanh(1)`).
    - dyntanh:  Used in the paper, similar to the tanh loss above, but the scaling factor decreases during the course of
                the training from `soft_clamp` to `soft_clamp_min`. The decrease is linear, unless `circle_schedule`
                is True (default), in which case it applies a circular scheduling. See paper for details.
    - l1:       Standard L1 loss, computed only on those pixels having an error lower than `soft_clamp`
    - l1+sqrt:  L1 loss for pixels with reprojection error smaller than `soft_clamp` and
                `sqrt(soft_clamp * reprojection_error)` for pixels with a higher error.
    - l1+logl1: Similar to the above, but using log L1 for pixels with high reprojection error.
    """

    def __init__(self,
                 total_iterations,
                 soft_clamp,
                 soft_clamp_min,
                 type='dyntanh',
                 circle_schedule=True):

        self.total_iterations = total_iterations
        self.soft_clamp = soft_clamp
        self.soft_clamp_min = soft_clamp_min
        self.type = type
        self.circle_schedule = circle_schedule

    def compute(self, repro_errs_b1N, iteration):
        """
        Compute the reprojection loss based on the type of loss function specified during the initialization of the class.
        The types of loss function available are: 'tanh', 'dyntanh', 'l1', 'l1+sqrt', and 'l1+logl1'.

        :param repro_errs_b1N: A tensor containing the reprojection errors.
        :param iteration: The current iteration of the training process.
        :return: The computed loss as a scalar.
        """
        # If there are no elements in the reprojection errors tensor, return 0
        if repro_errs_b1N.nelement() == 0:
            return 0

        # Compute the simple tanh loss
        if self.type == "tanh":
            return weighted_tanh(repro_errs_b1N, self.soft_clamp)

        # Compute the dynamic tanh loss
        elif self.type == "dyntanh":
            # Compute the progress over the training process.
            schedule_weight = iteration / self.total_iterations

            # Optionally scale it using the circular schedule.
            if self.circle_schedule:
                schedule_weight = 1 - np.sqrt(1 - schedule_weight ** 2)

            # Compute the weight to use in the tanh loss.
            loss_weight = (1 - schedule_weight) * self.soft_clamp + self.soft_clamp_min

            # Compute actual loss.
            return weighted_tanh(repro_errs_b1N, loss_weight)

        # Compute the L1 loss
        elif self.type == "l1":
            # L1 loss on all pixels with small-enough error.
            softclamp_mask_b1 = repro_errs_b1N > self.soft_clamp
            return repro_errs_b1N[~softclamp_mask_b1].sum()

        # Compute the L1 loss for small errors and sqrt loss for larger errors
        elif self.type == "l1+sqrt":
            softclamp_mask_b1 = repro_errs_b1N > self.soft_clamp
            loss_l1 = repro_errs_b1N[~softclamp_mask_b1].sum()
            loss_sqrt = torch.sqrt(self.soft_clamp * repro_errs_b1N[softclamp_mask_b1]).sum()

            return loss_l1 + loss_sqrt

        # Compute the L1 loss for small errors and log L1 loss for larger errors
        else:
            softclamp_mask_b1 = repro_errs_b1N > self.soft_clamp
            loss_l1 = repro_errs_b1N[~softclamp_mask_b1].sum()
            loss_logl1 = torch.log(1 + (self.soft_clamp * repro_errs_b1N[softclamp_mask_b1])).sum()

            return loss_l1 + loss_logl1