#!/usr/bin/env python3

import argparse
import os
import warnings

import dataset_util as dutil
import numpy as np
import torch
from joblib import Parallel, delayed
from skimage import io

# name of the folder where we download the original 7scenes dataset to
# we restructure the dataset by creating symbolic links to that folder
src_folder = '7scenes'
focal_length = 525.0

# focal length of the depth sensor (source: https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
d_focal_length = 585.0

# RGB image dimensions
img_w = 640
img_h = 480

# sub sampling factor of eye coordinate tensor
nn_subsampling = 8

# transformation from depth sensor to RGB sensor
# calibration according to https://projet.liris.cnrs.fr/voir/activities-dataset/kinect-calibration.html
d_to_rgb = np.array([
    [9.9996518012567637e-01, 2.6765126468950343e-03, -7.9041012313000904e-03, -2.5558943178152542e-02],
    [-2.7409311281316700e-03, 9.9996302803027592e-01, -8.1504520778013286e-03, 1.0109636268061706e-04],
    [7.8819942130445332e-03, 8.1718328771890631e-03, 9.9993554558014031e-01, 2.0318321729487039e-03],
    [0, 0, 0, 1]
])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download and setup the 7Scenes dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--setup_ace_structure', action='store_true',
                        help='Create a copy of the dataset in the ACE format in 7scenes_ace. '
                             'Otherwise, the dataset is left in the original format in 7scenes.')

    parser.add_argument('--depth', type=str, choices=['none', 'rendered', 'calibrated'], default='none',
                        help='For ACE dataset format only. '
                             'none: ignore depth maps; '
                             'rendered: download depth rendered using 3D scene model (28GB); '
                             'calibrated: register original depth maps to RGB')

    parser.add_argument('--eye', type=str, choices=['none', 'calibrated'], default='none',
                        help='For ACE dataset format only. '
                             'none: ignore eye coordinates; '
                             'original: calibrate original depth maps and precompute eye coordinates')

    parser.add_argument('--poses', type=str, choices=['original', 'calibrated', 'pgt'], default='calibrated',
                        help='For ACE dataset format only. '
                             'original: use raw poses of depth sensor; '
                             'calibrated: register poses to RGB sensor; '
                             'pgt: get SfM poses from external repository (Brachmann et al., ICCV21)')

    opt = parser.parse_args()

    print("\n############################################################################")
    print("# Please make sure to check this dataset's license before using it!        #")
    print("# https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/ #")
    print("############################################################################\n\n")

    license_response = input('Please confirm with "yes" or abort. ')
    if not (license_response == "yes" or license_response == "y"):
        print(f"Your response: {license_response}. Aborting.")
        exit()

    if opt.poses == 'pgt':

        print("\n###################################################################")
        print("# You requested external pose files. Please check the license at: #")
        print("# https://github.com/tsattler/visloc_pseudo_gt_limitations        #")
        print("###################################################################\n\n")

        license_response = input('Please confirm with "yes" or abort. ')
        if not (license_response == "yes" or license_response == "y"):
            print(f"Your response: {license_response}. Aborting.")
            exit()

        print("Getting external pose files...")
        external_pgt_folder = dutil.clone_external_pose_files()

    # download the original 7 scenes dataset for poses and images
    dutil.mkdir(src_folder)
    os.chdir(src_folder)

    for ds in ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']:

        if not os.path.exists(ds):

            print("=== Downloading 7scenes Data:", ds, "===============================")

            os.system(
                'wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/' + ds + '.zip')
            os.system('unzip ' + ds + '.zip')
            os.system('rm ' + ds + '.zip')

            sequences = os.listdir(ds)

            for file in sequences:
                if file.endswith('.zip'):
                    print("Unpacking", file)
                    os.system('unzip ' + ds + '/' + file + ' -d ' + ds)
                    os.system('rm ' + ds + '/' + file)
        else:
            print(f"Found data of scene {ds} already. Assuming its complete and skipping download.")

    if not opt.setup_ace_structure:
        print("ACE dataset format not requested. Done.")
        exit()

    print("Processing frames...")

    def process_scene(ds):

        if opt.poses == 'pgt':
            target_folder = '../7scenes_ace/pgt_7scenes_' + ds + '/'
        else:
            target_folder = '../7scenes_ace/kf_7scenes_' + ds + '/'

        def link_frames(split_file, variant):

            # create subfolders
            dutil.mkdir(target_folder + variant + '/rgb/')
            dutil.mkdir(target_folder + variant + '/poses/')
            dutil.mkdir(target_folder + variant + '/calibration/')
            if opt.depth == 'calibrated':
                dutil.mkdir(target_folder + variant + '/depth/')
            if opt.eye == 'calibrated':
                dutil.mkdir(target_folder + variant + '/eye/')

            # read the split file
            with open(ds + '/' + split_file, 'r') as f:
                split = f.readlines()
            # map sequences to folder names
            split = ['seq-' + s.strip()[8:].zfill(2) for s in split]

            # read external poses and calibration if requested
            if opt.poses == 'pgt':
                pgt_file = os.path.join('..', external_pgt_folder, '7scenes', f'{ds}_{variant}.txt')
                pgt_poses = dutil.read_pose_data(pgt_file)

            for seq in split:
                files = os.listdir(ds + '/' + seq)

                # link images
                images = [f for f in files if f.endswith('color.png')]
                for img in images:
                    os.system(
                        'ln -s ../../../' + src_folder + '/' + ds + '/' + seq + '/' + img + ' ' + target_folder + variant + '/rgb/' + seq + '-' + img)

                pose_files = [f for f in files if f.endswith('pose.txt')]

                # create pose files
                if opt.poses == 'original':

                    # link original poses
                    for p_file in pose_files:
                        os.system(
                            'ln -s ../../../' + src_folder + '/' + ds + '/' + seq + '/' + p_file + ' ' + target_folder + variant + '/poses/' + seq + '-' + p_file)

                elif opt.poses == 'pgt':

                    # use poses as externally provided
                    for p_file in pose_files:
                        cam_pose, _ = pgt_poses[os.path.join(seq, dutil.get_base_file_name(p_file))]
                        dutil.write_cam_pose(target_folder + variant + '/poses/' + seq + '-' + p_file, cam_pose)

                else:

                    # adjust original camera pose files by mapping to RGB sensor
                    for p_file in pose_files:
                        # load original pose (aligned to the depth sensor)
                        cam_pose = np.loadtxt(ds + '/' + seq + '/' + p_file)

                        # apply relative transform from depth to RGB sensor
                        cam_pose = np.matmul(cam_pose, np.linalg.inv(d_to_rgb))

                        # write adjusted pose (aligned to the RGB sensor)
                        dutil.write_cam_pose(target_folder + variant + '/poses/' + seq + '-' + p_file, cam_pose)

                # create calibration files
                if opt.poses == 'pgt':

                    for p_file in pose_files:
                        base_file = dutil.get_base_file_name(p_file)
                        calib_file = f'{base_file}.calibration.txt'

                        _, rgb_f = pgt_poses[os.path.join(seq, base_file)]
                        dutil.write_focal_length(target_folder + variant + '/calibration/' + seq + '-' + calib_file,
                                                 rgb_f)
                else:
                    for i in range(len(images)):
                        dutil.write_focal_length(
                            target_folder + variant + '/calibration/%s-frame-%s.calibration.txt' % (
                            seq, str(i).zfill(6)),
                            focal_length)

                if opt.depth != 'calibrated' and opt.eye != 'calibrated':
                    # no calibration requested, done
                    continue

                # adjust depth files by mapping to RGB sensor
                depth_files = [f for f in files if f.endswith('depth.png')]

                for d_file in depth_files:

                    if opt.poses == 'pgt':
                        # use correct per-frame focal length if provided
                        _, rgb_f = pgt_poses[os.path.join(seq, dutil.get_base_file_name(d_file))]
                    else:
                        # use default focal length as fall back
                        rgb_f = focal_length

                    depth = io.imread(ds + '/' + seq + '/' + d_file)
                    depth = depth.astype(np.float64)
                    depth /= 1000  # from millimeters to meters

                    d_h = depth.shape[0]
                    d_w = depth.shape[1]

                    # reproject depth map to 3D eye coordinates
                    eye_coords = np.zeros((4, d_h, d_w))
                    # set x and y coordinates
                    eye_coords[0] = np.dstack([np.arange(0, d_w)] * d_h)[0].T
                    eye_coords[1] = np.dstack([np.arange(0, d_h)] * d_w)[0]

                    eye_coords = eye_coords.reshape(4, -1)
                    depth = depth.reshape(-1)

                    # filter pixels with invalid depth
                    depth_mask = (depth > 0) & (depth < 100)
                    eye_coords = eye_coords[:, depth_mask]
                    depth = depth[depth_mask]

                    # substract depth principal point (assume image center)
                    eye_coords[0] -= d_w / 2
                    eye_coords[1] -= d_h / 2
                    # reproject
                    eye_coords[0:2] /= d_focal_length
                    eye_coords[0] *= depth
                    eye_coords[1] *= depth
                    eye_coords[2] = depth
                    eye_coords[3] = 1

                    # transform from depth sensor to RGB sensor
                    eye_coords = np.matmul(d_to_rgb, eye_coords)

                    # project
                    depth = eye_coords[2]

                    eye_coords[0] /= depth
                    eye_coords[1] /= depth
                    eye_coords[0:2] *= rgb_f

                    # add RGB principal point (assume image center)
                    eye_coords[0] += img_w / 2
                    eye_coords[1] += img_h / 2

                    registered_depth = np.zeros((img_h, img_w), dtype='uint16')

                    for pt in range(eye_coords.shape[1]):
                        x = round(eye_coords[0, pt])
                        y = round(eye_coords[1, pt])
                        d = eye_coords[2, pt]

                        registered_depth[y, x] = d * 1000

                    if opt.depth == 'calibrated':
                        # store calibrated depth
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            io.imsave(target_folder + variant + '/depth/' + seq + '-' + d_file, registered_depth)

                    if opt.eye != 'calibrated':
                        # done
                        continue

                    # generate sub-sampled eye coordinate tensor from calibrated depth
                    out_h = int(img_h / nn_subsampling)
                    out_w = int(img_w / nn_subsampling)
                    nn_offset = int(nn_subsampling / 2)

                    eye_tensor = np.zeros((3, out_h, out_w))

                    # generate pixel coordinates
                    eye_tensor[0] = np.dstack([np.arange(0, out_w)] * out_h)[0].T * nn_subsampling + nn_offset
                    eye_tensor[1] = np.dstack([np.arange(0, out_h)] * out_w)[0] * nn_subsampling + nn_offset

                    # substract principal point
                    eye_tensor[0] -= img_w / 2
                    eye_tensor[1] -= img_h / 2

                    # prepare depth (convert to meters and subsample)
                    registered_depth = registered_depth.astype(float)
                    registered_depth /= 1000
                    registered_depth = registered_depth[nn_offset::nn_subsampling, nn_offset::nn_subsampling]

                    # project
                    eye_tensor[0:2] /= rgb_f
                    eye_tensor[2, 0:registered_depth.shape[0], 0:registered_depth.shape[1]] = registered_depth
                    eye_tensor[0] *= eye_tensor[2]
                    eye_tensor[1] *= eye_tensor[2]

                    eye_tensor = torch.from_numpy(eye_tensor).float()

                    torch.save(eye_tensor,
                               target_folder + variant + '/eye/' + seq + '-' + d_file[:-10] + '.eye.dat')

        link_frames('TrainSplit.txt', 'train')
        link_frames('TestSplit.txt', 'test')


    Parallel(n_jobs=7, verbose=0)(
        map(delayed(process_scene), ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']))

    if opt.depth == 'rendered':
        os.chdir('..')
        dutil.dlheidata("10.11588/data/N07HKC/4PLEEJ", "7scenes_depth.tar.gz")
