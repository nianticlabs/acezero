import shutil
from dataclasses import dataclass
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class Resolution:
    height: int
    width: int

    def __post_init__(self):
        assert self.height > 0
        assert self.width > 0

    def flip(self):
        return Resolution(height=self.width, width=self.height)

    def __hash__(self) -> int:
        return hash((self.height, self.width))


@dataclass
class Frame:
    # Represents a single RGB frame
    rgb_path: Path

    def get_image_resolution(self) -> Resolution:
        im = Image.open(self.rgb_path)
        return Resolution(height=im.height, width=im.width)

    def load_image(self) -> Image.Image:
        return Image.open(self.rgb_path)


def preprocess_json_frames(ace_poses_path: Path, dataset_frames: List[Frame]) -> List[Dict]:
    # Takes a path to a file containing some ACE-format poses, and a dataset as represented by a list of Frame objects,
    # and processes them into some 'transforms.json'-style frames.

    poses = parse_ace_poses_file(ace_poses_path)
    json_frames = convert_ace_poses_to_nerf_blender_frames_json(poses)
    resolution = get_resolution_from_frames(frames=dataset_frames)
    json_frames_lookup: dict[str, dict] = {frame['file_path']: frame for frame in json_frames}

    # We iterate over all frames in the dataset, rather than the frames we got from the poses file, because some frames
    # might not have poses; in such cases, the frame still needs to go into the transforms.json file, but with an
    # identity pose.
    json_frames_preprocessed = []
    for dataset_frame in dataset_frames:
        print('Processing frame', dataset_frame.rgb_path)

        # Match this dataset frame with the corresponding pose from the poses file (if there is one there)
        json_frame = json_frames_lookup.get(str(dataset_frame.rgb_path))
        if json_frame is None:
            # If no pose was provided for this frame, we will use an identity pose
            print(f'WARNING: No pose found for frame {dataset_frame.rgb_path}; using identity pose instead!')
            json_frame = make_json_frame_for_missing_pose(dataset_frame, resolution)

        # We expect to have focal length info, and we also assume elsewhere that they are equal
        # so we should verify this explicitly:
        assert json_frame['fl_x'] == json_frame['fl_y'], 'Expected focal lengths to be equal'

        # Now make an intrinsics dict from the focal length for this frame:
        intrinsics = make_intrinsics_dict_from_focal(resolution=resolution, focal_length=json_frame['fl_x'])

        print('Intrinsics for frame are', intrinsics)
        json_frame.update(intrinsics)
        json_frames_preprocessed.append(json_frame)
    return json_frames_preprocessed


def convert_ace_zero_to_nerf_blender_format(poses_path: Path, images_glob_pattern: str, output_path: Path,
                                            split_file_path: Optional[Path] = None) -> None:
    """
    Converts a dataset in the Ace Zero format to a format suitable for ingestion by Nerfstudio.
    """
    dataset_frames = glob_for_frames(images_glob_pattern=images_glob_pattern)
    json_frames = preprocess_json_frames(ace_poses_path=poses_path, dataset_frames=dataset_frames)

    # Filter out low-confidence poses: any training pose with <1000 confidence gets dropped from the nerf train set
    print('json frames before confidence filtering', json_frames)
    for frame in json_frames:
        print(frame)

    # Impose a split if one is specified
    if split_file_path is not None:
        print('Using split file', split_file_path)
        split_frames = apply_precomputed_split(
            frames_json=json_frames,
            split_file_path=split_file_path,
        )
    else:
        print('No split file given; taking every 8th frame as test set')
        split_frames = split_frames_json(frames_json=json_frames)

    # If frames are low-confidence, we exclude them from NeRF fitting, because NeRFs are highly sensitive to even a few
    # bad poses in the training set.
    split_frames['train'] = [frame for frame in split_frames['train'] if frame['confidence_score'] >= 1000]
    print('train split size after confidence filtering', len(split_frames['train']))

    transforms_json = {
        "frames": json_frames,
        **make_filenames_json(split_frames=split_frames),
    }
    assert len(transforms_json['train_filenames']) > 0, 'No train filenames! Must have at least one'

    # Check whether there is a ACE point cloud file, and if so, copy it to the output directory
    point_cloud_file = poses_path.parent / 'pc_final.ply'
    if point_cloud_file.exists():
        print('Copying point cloud file', point_cloud_file, 'to', output_path)
        shutil.copy(point_cloud_file, output_path / 'pc_final.ply')

        # Add point cloud file to the transforms.json file
        transforms_json['ply_file_path'] = 'pc_final.ply'

    print('Writing transforms.json to', output_path / 'transforms.json')
    with open(output_path / 'transforms.json', 'w') as f:
        json.dump(transforms_json, f)


def glob_for_frames(images_glob_pattern: str) -> List[Frame]:
    # Load all frames from a glob pattern.
    # The glob pattern might be e.g.:
    # ./data/ace0/scene_1/*png
    # ./data/ace0/scene_2/*jpg
    print('Globbing for frames from glob pattern ', images_glob_pattern)
    frames = [
        Frame(rgb_path=Path(rgb_path))
        for rgb_path in glob.glob(images_glob_pattern)
    ]
    print(f'Found {len(frames)} frames')
    return frames


def split_frames_json(frames_json: list, sample_interval: int = 8) -> dict:
    # Split into train and test sets using every N images as test.
    # We use 8 as default because this is a standard choice used by many other NeRF-adjacent works.
    # Note that we don't start at frame 0 but at frame sample_interval//2
    frames_sorted = sorted(frames_json, key=lambda x: x['file_path'])
    frames_idxs = list(range(len(frames_json)))
    test_idxs = frames_idxs[int(sample_interval/2)::sample_interval]
    train_idxs = [idx for idx in frames_idxs if idx not in test_idxs]

    test_frames = [frames_sorted[idx] for idx in test_idxs]
    train_frames = [frames_sorted[idx] for idx in train_idxs]
    print('train split size', len(train_frames))
    print('test split size', len(test_frames))
    return {'train': train_frames, 'test': test_frames}


def get_resolution_from_frames(frames: List[Frame]) -> Resolution:
    assert len(frames) > 0, 'Expected at least one frame'
    # Load resolutions for all frames
    resolutions = [frame.get_image_resolution() for frame in frames]
    # Verify that all resolutions are equal
    assert len(set(resolutions)) == 1, f"Expected all frames' resolutions to be equal, but got {resolutions}"
    return resolutions[0]


def make_json_frame_for_missing_pose(frame: Frame, resolution: Resolution) -> dict:
    # If a frame doesn't have a pose, we still need to include it in the transforms.json file, but with an identity pose
    # This function implements this logic
    return {
        "file_path": str(frame.rgb_path),
        "transform_matrix": [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ],
        # Put in an approximate focal length since we didn't get any via the ACE0 pose for for this frame.
        # In practice these numbers aren't very important because with an identity pose, the results will come out
        # looking terrible anyway
        "fl_x": resolution.height * 0.7,
        "fl_y": resolution.height * 0.7,
        "confidence_score": 0. # Confidence score of 0 since we have had no pose at all for this frame. This will
                               # exclude the pose from the training set.
    }


def apply_precomputed_split(frames_json: List[Dict], split_file_path: Path) -> dict:
    # Imposes a precomputed split file on some frames.
    # We return a dict containing the train and test filenames.

    # First parse the split file:
    with open(split_file_path, 'r') as f:
        split_json = json.load(f)
    train_filenames = set(split_json['train_filenames'])
    test_filenames = set(split_json['test_filenames'])
    print('Splitting:', frames_json)

    # Now split frames into train and test
    test_frames = []
    train_frames = []
    for frame in frames_json:
        if frame['file_path'] in train_filenames:
            train_frames.append(frame)
        elif frame['file_path'] in test_filenames:
            test_frames.append(frame)
        else:
            raise Exception(f'Frame {frame} not found in split file {split_file_path}')

    return {
        'train': train_frames,
        'test': test_frames
    }


def parse_ace_poses_file(file_path: Path) -> List[tuple]:
    # Parses an ACE0 poses file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    def parse_line(line):
        data = line.strip().split()
        file_path = data[0]
        q = [float(data[i]) for i in range(1, 5)]
        t = [float(data[i]) for i in range(5, 8)]
        focal = float(data[-2])
        confidence_score = int(data[-1])
        # sanity check - how long is the line?
        assert len(data) == 10, f'Unexpected line length {len(data)}; expected 10'
        return file_path, q, t, focal, confidence_score

    return [parse_line(line) for line in lines]


def convert_ace_poses_to_nerf_blender_frames_json(poses: List[tuple]) -> List[dict]:
    # Converts some ACE poses - as loaded via parse_ace_poses_file - to the format expected by NerfStudio
    frames = []

    for (file_path, q, t, f, confidence_score) in poses:
        transform_matrix = quaternion_to_matrix(q, t)

        # Convert from world-to-camera to camera-to-world
        transform_matrix = np.linalg.inv(transform_matrix)

        # Convert from OpenCV to Blender coordinate system
        transform_matrix = convert_opencv_to_opengl(transform_matrix, transform_type='cam2world')

        # Convert matrix back to list of lists so that it can be serialized to JSON
        transform_matrix = transform_matrix.tolist()

        frame = {
            "file_path": file_path,
            "transform_matrix": transform_matrix,
            "confidence_score": confidence_score,
            "fl_x": f,
            "fl_y": f,
        }

        frames.append(frame)
    return frames


def convert_opencv_to_opengl(opencv_mat, transform_type='world2cam'):
    # If the input matrix is cam2world, invert it to get world2cam
    if transform_type == 'cam2world':
        opencv_mat = np.linalg.inv(opencv_mat)

    # This matrix converts between opengl (x right, y up, z back) and cv-style (x right, y down, z forward) coordinates
    # For nerfstudio, we want opengl coordinates
    coord_transform = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ])

    # NB for the following we expect the mat to be in world2cam format
    opengl_mat = coord_transform @ opencv_mat

    if transform_type == 'cam2world':
        opengl_mat = np.linalg.inv(opengl_mat)

    return opengl_mat


def quaternion_to_matrix(q: List[float], t: List[float], input_quat_type='wxyz') -> List[List[float]]:
    # Convert quaternion to rotation matrix
    # Scipy wants xyzw format, so we need to permute the components if the input is wxyz:
    if input_quat_type == 'wxyz':
        r = R.from_quat([q[1], q[2], q[3], q[0]])
    else:
        assert input_quat_type == 'xyzw', f'Unexpected input_quat_type {input_quat_type}'
        r = R.from_quat([q[0], q[1], q[2], q[3]])
    matrix = r.as_matrix()

    # Construct 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = matrix
    transform[:3, 3] = t

    return transform.tolist()


def make_filenames_json(split_frames: Dict):
    return {
        "train_filenames": [frame['file_path'] for frame in split_frames['train']],
        "val_filenames": [],
        "test_filenames": [frame['file_path'] for frame in split_frames['test']],
    }


def make_intrinsics_dict_from_focal(resolution: Resolution, focal_length: float) -> dict:
    return {
        "fl_x": focal_length,
        "fl_y": focal_length,
        "k1": 0.,
        "k2": 0.,
        "p1": 0.,
        "p2": 0.,
        "cx": resolution.width / 2.,
        "cy": resolution.height / 2.,
        "w": resolution.width,
        "h": resolution.height,
    }
