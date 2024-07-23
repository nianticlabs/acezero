import glob
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import logging
import torchvision.transforms.functional as TF

_logger = logging.getLogger(__name__)


def load_pose(pose_file):
    """
    Load a pose from a file. The pose file should contain a 4x4 matrix.
    The pose is loaded using numpy's loadtxt function, converted to a torch tensor, and returned.

    :param pose_file: The path to the pose file.
    :return: The pose as a 4x4 torch tensor.
    """
    # Stored as a 4x4 matrix.
    pose = np.loadtxt(pose_file)
    pose = torch.from_numpy(pose).float()

    return pose


def load_focal_length(calibration_file):
    """
    Load the focal length from a calibration file. The calibration file can either contain the focal length directly
    or a calibration matrix. If the calibration file contains more than one value, it is assumed to be a calibration matrix
    and the focal length is extracted from the first element of the first row of the matrix. If the calibration file
    contains only one value, it is assumed to be the focal length.

    :param calibration_file: The path to the calibration file.
    :return: The focal length as a float.
    """
    # load data from calibration file
    calibration_data = np.loadtxt(calibration_file)

    if calibration_data.size > 1:
        # assume calibration file contains calibration matrix
        return float(np.loadtxt(calibration_file)[0, 0])
    else:
        # assume calibration file contains focal length only
        return float(np.loadtxt(calibration_file))


def get_files_from_glob(glob_pattern):
    """
    Get a list of files from a glob pattern, sorted alphabetically.
    """
    files = sorted(glob.glob(glob_pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No files found for glob pattern: {glob_pattern}")

    return files


def load_pose_files(glob_pattern):
    """
    Load pose files by resolving the glob pattern (sorted alphabetically), return as a list of 4x4 torch tensors.
    """
    pose_files = sorted(glob.glob(glob_pattern))
    return [load_pose(pose_file) for pose_file in pose_files]


def check_pose(pose):
    """
    Check if a pose is valid. A pose is considered valid if it does not contain NaN or inf values.

    :param pose: The pose as a 4x4 torch tensor.
    :return: True if the pose is valid, False otherwise.
    """
    return not torch.isnan(pose).any() and not torch.isinf(pose).any()


def remove_invalid_poses(rgb_files, poses):
    """
    Remove each invalid pose from poses and the corresponding RGB file from rgb_files.
    An invalid pose is a pose that contains NaN or inf values.
    """

    valid_rgb_files = []
    valid_poses = []

    for rgb_file, pose in zip(rgb_files, poses):
        if not check_pose(pose):
            _logger.warning(f"Pose for {rgb_file} contains NaN or inf values, skipping.")
        else:
            valid_rgb_files.append(rgb_file)
            valid_poses.append(pose)

    return valid_rgb_files, valid_poses


def load_dataset_ace(pose_file, confidence_threshold):
    """
    Load a dataset from a pose file. The pose file should contain lines with 10 tokens each.
    Poses are assumed to be world-to-cam.
    The tokens represent the following information:
    - mapping file
    - quaternion rotation (w, x, y, z)
    - translation (x, y, z)
    - focal length
    - confidence value

    Only entries with a confidence value above the provided threshold are included in the output.

    :param pose_file: The path to the pose file.
    :param confidence_threshold: The minimum confidence value for an entry to be included in the output.
    :return: A tuple containing three lists:
        - rgb_files: The paths to the RGB files.
        - poses: The poses as 4x4 torch tensors, cam-to-world.
        - focal_lengths: The focal lengths.
    """

    with open(pose_file, 'r') as f:
        pose_file_data = f.readlines()

        rgb_files = []
        poses = []
        focal_lengths = []

        for pose_file_entry in pose_file_data:

            pose_file_tokens = pose_file_entry.split()

            assert len(pose_file_tokens) == 10, f"Expected 10 tokens per line in pose file, got {len(pose_file_tokens)}"

            # read confidence values and compare to threshold
            confidence = float(pose_file_tokens[-1])
            if confidence < confidence_threshold:
                continue

            mapping_file = pose_file_tokens[0]

            # convert quaternion to rotation matrix
            mapping_pose_q_wxyz = [float(t) for t in pose_file_tokens[1:5]]
            mapping_pose_q_xyzw = mapping_pose_q_wxyz[1:] + [mapping_pose_q_wxyz[0]]
            mapping_pose_R = Rotation.from_quat(mapping_pose_q_xyzw).as_matrix()
            # read translation
            mapping_pose_t = [float(t) for t in pose_file_tokens[5:8]]
            # construct full pose matrix
            mapping_pose_4x4 = np.eye(4)
            mapping_pose_4x4[:3, :3] = mapping_pose_R
            mapping_pose_4x4[:3, 3] = mapping_pose_t

            # pose files contain world-to-cam but we need cam-to-world
            mapping_pose_4x4 = np.linalg.inv(mapping_pose_4x4)
            mapping_pose_4x4 = torch.from_numpy(mapping_pose_4x4).float()

            rgb_files.append(mapping_file)
            focal_lengths.append(float(pose_file_tokens[-2]))
            poses.append(mapping_pose_4x4)

    return rgb_files, poses, focal_lengths


def write_pose_to_pose_file(out_pose_file, rgb_file, pose, confidence, focal_length):
    """
    Write a pose to a pose file. The pose is converted from a numpy matrix to a quaternion and translation.
    The pose file line format is as follows:
    - RGB file path
    - Quaternion rotation (w, x, y, z)
    - Translation (x, y, z)
    - Focal length
    - Confidence value

    :param out_pose_file: The output file to write the pose to.
    :param rgb_file: The path to the RGB file.
    :param pose: The pose as a numpy matrix, 4x4 or 3x4, world-to-cam.
    :param confidence: The confidence value.
    :param focal_length: The focal length.
    """

    # convert Numpy pose matrix to quaternion and translation
    R_33 = pose[:3, :3]
    q_xyzw = Rotation.from_matrix(R_33).as_quat()
    t_xyz = pose[:3, 3]

    # write to pose file
    pose_str = f"{rgb_file} " \
               f"{q_xyzw[3]} {q_xyzw[0]} {q_xyzw[1]} {q_xyzw[2]} " \
               f"{t_xyz[0]} {t_xyz[1]} {t_xyz[2]} {focal_length} {confidence}\n"

    out_pose_file.write(pose_str)


def get_depth_model(init=False):
    """
    Load the pretrained ZoeDepth model from the isl-org/ZoeDepth repository.
    Use torch.hub.load to load the model directly from GitHub.

    :param init: Force reload the model from the repository.
    """

    # Warm up dependency in the torch hub cache.
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=init)
    repo = "isl-org/ZoeDepth"

    # # Zoe_N
    # model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True, force_reload=init)

    # # Zoe_K
    # model_zoe_k = torch.hub.load(repo, "ZoeD_K", pretrained=True, force_reload=init)

    # Zoe_NK (best performing model).
    model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True, force_reload=init)
    model_zoe_nk.eval().cuda()
    _logger.info(f"Loaded pretrained ZoeDepth model.")

    return model_zoe_nk


def estimate_depth(model: torch.nn.Module, image_rgb: np.ndarray) -> np.ndarray:
    """
    Estimate depth from an RGB image using the ZoeDepth model.

    :param model: The ZoeDepth model.
    :param image_rgb: The RGB image as a numpy array (HxWx3).

    :return: The estimated depth as a numpy array (in m, HxW).
    """
    # Convert to tensor.
    image_BCHW = TF.to_tensor(image_rgb).unsqueeze(0).cuda()

    # Run forward pass (on CPU)
    with torch.no_grad():
        depth_B1HW = model.infer(image_BCHW)

    # Convert to numpy.
    depth_HW = depth_B1HW.squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)

    return depth_HW
