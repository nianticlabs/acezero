import math
import random
from collections import namedtuple
from scipy.spatial.transform import Rotation
import numpy as np
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

TestEstimate = namedtuple("TestEstimate", [
    "pose_est",
    "pose_gt",
    "focal_length",
    "confidence",
    "image_file"
])


def kabsch(pts1, pts2, estimate_scale=False):
    c_pts1 = pts1 - pts1.mean(axis=0)
    c_pts2 = pts2 - pts2.mean(axis=0)

    covariance = np.matmul(c_pts1.T, c_pts2) / c_pts1.shape[0]

    U, S, VT = np.linalg.svd(covariance)

    d = np.sign(np.linalg.det(np.matmul(VT.T, U.T)))
    correction = np.eye(3)
    correction[2, 2] = d

    if estimate_scale:
        pts_var = np.mean(np.linalg.norm(c_pts2, axis=1) ** 2)
        scale_factor = pts_var / np.trace(S * correction)
    else:
        scale_factor = 1.

    R = scale_factor * np.matmul(np.matmul(VT.T, correction), U.T)
    t = pts2.mean(axis=0) - np.matmul(R, pts1.mean(axis=0))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T, scale_factor


def print_hyp(hypothesis, hyp_name):
    h_translation = np.linalg.norm(hypothesis['transformation'][:3, 3])
    h_angle = np.linalg.norm(Rotation.from_matrix(hypothesis['transformation'][:3, :3]).as_rotvec()) * 180 / math.pi
    _logger.debug(f"{hyp_name}: score={hypothesis['score']}, translation={h_translation:.2f}m, "
                 f"rotation={h_angle:.1f}deg.")


def get_inliers(h_T, poses_gt, poses_est, inlier_threshold_t, inlier_threshold_r):

    # h_T aligns ground truth poses with estimates poses
    poses_gt_transformed = h_T @ poses_gt

    # calculate differences in position and rotations
    translations_delta = poses_gt_transformed[:, :3, 3] - poses_est[:, :3, 3]
    rotations_delta = poses_gt_transformed[:, :3, :3] @ poses_est[:, :3, :3].transpose([0, 2, 1])

    # translation inliers
    inliers_t = np.linalg.norm(translations_delta, axis=1) < inlier_threshold_t
    # rotation inliers
    inliers_r = Rotation.from_matrix(rotations_delta).magnitude() < (inlier_threshold_r / 180 * math.pi)
    # intersection of both
    return np.logical_and(inliers_r, inliers_t)

def estimate_alignment(estimates,
                       confidence_threshold,
                       min_cofident_estimates=10,
                       inlier_threshold_t=0.05,
                       inlier_threshold_r=5,
                       ransac_iterations=10000,
                       refinement_max_hyp=12,
                       refinement_max_it=8,
                       estimate_scale=False
                       ):
    _logger.debug("Estimate transformation between pose estimates and ground truth.")

    # Filter estimates using confidence threshold
    valid_estimates = [estimate for estimate in estimates if ((not np.any(np.isinf(estimate.pose_gt))) and (not np.any(np.isnan(estimate.pose_gt))))]
    confident_estimates = [estimate for estimate in valid_estimates if estimate.confidence > confidence_threshold]
    num_confident_estimates = len(confident_estimates)

    _logger.debug(f"{num_confident_estimates} estimates considered confident.")

    if num_confident_estimates < min_cofident_estimates:
        _logger.debug(f"Too few confident estimates. Aborting alignment.")
        return None, 1

    # gather estimated and ground truth poses
    poses_est = np.ndarray((num_confident_estimates, 4, 4))
    poses_gt = np.ndarray((num_confident_estimates, 4, 4))
    for i, estimate in enumerate(confident_estimates):
        poses_est[i] = estimate.pose_est
        poses_gt[i] = estimate.pose_gt

    # start robust RANSAC loop
    ransac_hypotheses = []

    for hyp_idx in range(ransac_iterations):

        # sample hypothesis
        min_sample_size = 3
        samples = random.sample(range(num_confident_estimates), min_sample_size)
        h_pts1 = poses_gt[samples, :3, 3]
        h_pts2 = poses_est[samples, :3, 3]

        h_T, h_scale = kabsch(h_pts1, h_pts2, estimate_scale)

        # calculate inliers
        inliers = get_inliers(h_T, poses_gt, poses_est, inlier_threshold_t, inlier_threshold_r)

        if inliers[samples].sum() >= 3:
            # only keep hypotheses if minimal sample is all inliers
            ransac_hypotheses.append({
                "transformation": h_T,
                "inliers": inliers,
                "score": inliers.sum(),
                "scale": h_scale
            })

    if len(ransac_hypotheses) == 0:
        _logger.debug(f"Did not fine a single valid RANSAC hypothesis, abort alignment estimation.")
        return None, 1

    # sort according to score
    ransac_hypotheses = sorted(ransac_hypotheses, key=lambda x: x['score'], reverse=True)

    for hyp_idx, hyp in enumerate(ransac_hypotheses):
        print_hyp(hyp, f"Hypothesis {hyp_idx}")

    # create shortlist of best hypotheses for refinement
    _logger.debug(f"Starting refinement of {refinement_max_hyp} best hypotheses.")
    ransac_hypotheses = ransac_hypotheses[:refinement_max_hyp]

    # refine all hypotheses in the short list
    for ref_hyp in ransac_hypotheses:

        print_hyp(ref_hyp, "Pre-Refinement")

        # refinement loop
        for ref_it in range(refinement_max_it):

            # re-solve alignment on all inliers
            h_pts1 = poses_gt[ref_hyp['inliers'], :3, 3]
            h_pts2 = poses_est[ref_hyp['inliers'], :3, 3]

            h_T, h_scale = kabsch(h_pts1, h_pts2, estimate_scale)

            # calculate new inliers
            inliers = get_inliers(h_T, poses_gt, poses_est, inlier_threshold_t, inlier_threshold_r)

            # check whether hypothesis score improved
            refined_score = inliers.sum()

            if refined_score > ref_hyp['score']:

                ref_hyp['transformation'] = h_T
                ref_hyp['inliers'] = inliers
                ref_hyp['score'] = refined_score
                ref_hyp['scale'] = h_scale

                print_hyp(ref_hyp, f"Refinement interation {ref_it}")

            else:
                _logger.debug(f"Stopping refinement. Score did not improve: New score={refined_score}, "
                             f"Old score={ref_hyp['score']}")
                break

    # re-sort refined hyotheses
    ransac_hypotheses = sorted(ransac_hypotheses, key=lambda x: x['score'], reverse=True)

    for hyp_idx, hyp in enumerate(ransac_hypotheses):
        print_hyp(hyp, f"Hypothesis {hyp_idx}")

    return ransac_hypotheses[0]['transformation'], ransac_hypotheses[0]['scale']
