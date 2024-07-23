import os
import numpy as np
from scipy.spatial.transform import Rotation as Rotation


def mkdir(directory):
    """Checks whether the directory exists and creates it if necessacy."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def dlheidata(doi, filename):
    """Download and unpack data from heiData host."""
    os.system(
        f"wget https://heidata.uni-heidelberg.de/api/access/datafile/:persistentId?persistentId=doi:{doi} -O {filename}")
    os.system(f"tar -xvzf {filename}")
    os.system(f"rm {filename}")


def get_base_file_name(file_name):
    """
    Extracts the base file name by removing file extension and modality string.
    eg. frame-000000.color.jpg -> frame-000000
    """

    base_file = os.path.splitext(file_name)[0]
    base_file = os.path.splitext(base_file)[0]

    return base_file


def read_pose_data(file_name):
    """
    Expects path to file with one pose per line.
    Input pose is expected to map world to camera coordinates.
    Output pose maps camera to world coordinates.
    Pose format: file qw qx qy qz tx ty tz (f)
    Return dictionary that maps a file name to a tuple of (4x4 pose, focal_length)
    Sets focal_length to None if not contained in file.
    """

    with open(file_name, "r") as f:
        pose_data = f.readlines()

    # create a dict from the poses with file name as key
    pose_dict = {}
    for pose_string in pose_data:

        pose_string = pose_string.split()
        file_name = pose_string[0]

        pose_q = np.array(pose_string[1:5])
        pose_q = np.array([pose_q[1], pose_q[2], pose_q[3], pose_q[0]])
        pose_t = np.array(pose_string[5:8])
        pose_R = Rotation.from_quat(pose_q).as_matrix()

        pose_4x4 = np.identity(4)
        pose_4x4[0:3, 0:3] = pose_R
        pose_4x4[0:3, 3] = pose_t

        # convert world->cam to cam->world for evaluation
        pose_4x4 = np.linalg.inv(pose_4x4)

        if len(pose_string) > 8:
            focal_length = float(pose_string[8])
        else:
            focal_length = None

        pose_dict[get_base_file_name(file_name)] = (pose_4x4, focal_length)

    return pose_dict


def write_cam_pose(file_path, cam_pose):
    """
    Writes 4x4 camera pose to a human readable text file.
    """
    with open(file_path, 'w') as f:
        f.write(str(float(cam_pose[0, 0])) + ' ' + str(float(cam_pose[0, 1])) + ' ' + str(
            float(cam_pose[0, 2])) + ' ' + str(float(cam_pose[0, 3])) + '\n')
        f.write(str(float(cam_pose[1, 0])) + ' ' + str(float(cam_pose[1, 1])) + ' ' + str(
            float(cam_pose[1, 2])) + ' ' + str(float(cam_pose[1, 3])) + '\n')
        f.write(str(float(cam_pose[2, 0])) + ' ' + str(float(cam_pose[2, 1])) + ' ' + str(
            float(cam_pose[2, 2])) + ' ' + str(float(cam_pose[2, 3])) + '\n')
        f.write(str(float(cam_pose[3, 0])) + ' ' + str(float(cam_pose[3, 1])) + ' ' + str(
            float(cam_pose[3, 2])) + ' ' + str(float(cam_pose[3, 3])) + '\n')


def write_focal_length(file_path, focal_length):
    """
    Write the focal length to a human readable text file.
    """
    with open(file_path, 'w') as f:
        f.write(str(focal_length))


def clone_external_pose_files():
    '''
    Clone repository containing SfM pose files for 7Scenes and 12Scenes.

    From paper:
    On the Limits of Pseudo Ground Truth in Visual Camera Re-localisation
    Eric Brachmann, Martin Humenberger, Carsten Rother, Torsten Sattler
    ICCV 2021

    @return folder name of the repo
    '''
    repo_url = "https://github.com/tsattler/visloc_pseudo_gt_limitations.git"
    repo_folder = os.path.splitext(os.path.basename(repo_url))[0]

    if not os.path.exists(repo_folder):
        os.system('git clone https://github.com/tsattler/visloc_pseudo_gt_limitations.git')

    return os.path.join(repo_folder, 'pgt', 'sfm')
