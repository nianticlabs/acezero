import json
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple

from benchmarks.preprocess_data import convert_ace_zero_to_nerf_blender_format
from benchmarks.run_nerfstudio import eval_nerf_with_nerfstudio, fit_nerf_with_nerfstudio


def run_benchmark(pose_file: Path, images_glob_pattern: str, working_dir: Path, split_json: Optional[Path] = None,
                  dry_run: bool = False, ns_train_extra_args: Optional[Dict] = None,
                  downscale_factor_override: Optional[int] = None) -> Optional[Path]:
    """
    Top-level function to benchmark poses by fitting a NeRF.

    :param pose_file: Path to the poses file, in ACE0 format. Poses with confidence <1000 will be excluded from the
        training set.
    :param images_glob_pattern: Pattern relative to working directory to glob for images
    :param working_dir: Output directory where the benchmark results will be written
    :param split_json: Path to a JSON file containing splits; if not given, every 8 images will be test images
    :param dry_run: If True, then don't actually invoke nerfstudio (useful for debugging)
    :param ns_train_extra_args: Extra arguments to pass to nerfstudio's ns-train command
    :param downscale_factor_override: If not None, then override the downscale factor that would be used

    :return: Path to the eval json containing metrics such as PSNR. Will be None iff dry_run is True.
    """
    working_dir.mkdir(exist_ok=True)

    print(f'Working directory: {working_dir}')
    print(f'Input poses path: {pose_file}')
    print(f'Images glob pattern: {images_glob_pattern}')

    # Preprocess ACE0 poses into nerfstudio's expected transforms.json format.
    nerf_data_path = working_dir / 'nerf_data'
    nerf_data_path.mkdir(exist_ok=True)
    convert_ace_zero_to_nerf_blender_format(
        poses_path=pose_file,
        images_glob_pattern=images_glob_pattern,
        output_path=nerf_data_path,
        split_file_path=split_json,
    )
    sanity_check_transforms_json(json_path=nerf_data_path / 'transforms.json')

    # Enforce limits on number of test images to ensure that evaluation remains fast and OOM-free
    # NB have to do this before downscaling so that the sorted filenames are the same, because downscaling renames the
    # images such that the results of sorting alphabetically can then be different!
    limit_num_test_images(target_num_test_images=1000, transforms_json_path=nerf_data_path / 'transforms.json')

    # Some versions of nerfstudio seem to ignore eval-num-images-to-sample-from, so if this has been passed then
    # we enforce it by manually shortening the test_images in the json
    if ns_train_extra_args and 'pipeline.datamanager.eval-num-images-to-sample-from' in ns_train_extra_args:
        print('WARNING: Enforcing eval num images to sample from to '
              f'{ns_train_extra_args["pipeline.datamanager.eval-num-images-to-sample-from"]}')
        enforce_eval_num_images(json_path=nerf_data_path / 'transforms.json',
                                num_images=ns_train_extra_args['pipeline.datamanager.eval-num-images-to-sample-from'])

    # Possibly downscale
    print('Downscale factor override is', downscale_factor_override)
    if downscale_factor_override is None:
        downscale_factor = calculate_downscale_factor(transforms_json_path=nerf_data_path / 'transforms.json',
                                                      max_resolution=640)
    else:
        downscale_factor = downscale_factor_override
    if downscale_factor > 1:
        downscale_images(nerf_data_path=nerf_data_path, downscale_factor=downscale_factor)

    # Ensure paths in transforms json are absolute
    resolve_relative_paths_in_transforms_json(transforms_json_path=nerf_data_path / 'transforms.json')

    # Decide whether there are too many images for us to preload without OOMing
    preload_images = should_preload_images(json_path=nerf_data_path / 'transforms.json')

    if dry_run:
        return None

    # Fit NeRF
    print('Fitting NeRF...')
    fitted_nerf_path = fit_nerf_with_nerfstudio(nerf_data_path=nerf_data_path,
                                                downscale_factor=downscale_factor,
                                                preload_images=preload_images,
                                                ns_train_extra_args=ns_train_extra_args)

    # Evaluate PSNR and other metrics
    print('Evaluating...')
    eval_json_path = eval_nerf_with_nerfstudio(nerf_output_dir=fitted_nerf_path)
    print('Eval json is at ', eval_json_path)
    return eval_json_path


def limit_num_test_images(target_num_test_images: int, transforms_json_path: Path) -> None:
    with open(transforms_json_path, 'r') as f:
        data = json.load(f)

    sorted_filenames = sorted(data['test_filenames'])

    # If we already have fewer than the target number of images, then there is nothing to do:
    if len(sorted_filenames) <= target_num_test_images:
        return

    # Try to sample max_num_test_images uniformly:
    print('The NeRF test set is currently very large, so will be subsampled!')
    print(f'Pre-subsampling there are {len(sorted_filenames)} test images')
    sampled_filenames = sorted_filenames[::len(sorted_filenames) // target_num_test_images]
    data['test_filenames'] = sampled_filenames
    print(f'After subsampling, there are {len(data["test_filenames"])} test images')

    with open(transforms_json_path, 'w') as f:
        json.dump(data, f, indent=4)


def get_height_and_width_from_transforms_json(transforms_json_path: Path) -> Tuple[int, int]:
    # Load transforms capture json to discover image res:
    with open(transforms_json_path, 'r') as f:
        data = json.load(f)

    # Height and width might be per-dataset...
    try:
        height = data['h']
        width = data['w']
        return height, width
    except KeyError:
        pass

    # ...Or might be per-image, in which case we get all of them and require them to be the same
    heights = {
        frame['h'] for frame in data['frames']
    }
    width = {
        frame['w'] for frame in data['frames']
    }
    assert len(heights) == 1, f'Expected all images to have the same height, but got {heights}'
    assert len(width) == 1, f'Expected all images to have the same width, but got {width}'
    return heights.pop(), width.pop()


def calculate_downscale_factor(transforms_json_path: Path, max_resolution: int) -> int:
    # Figures out a factor by which to downscale the images in order to keep the side length below max_resolution.
    # Doesn't actually do the downscaling, just returns the factor to downscale by.

    # Load transforms capture json to discover image res
    height, width = get_height_and_width_from_transforms_json(transforms_json_path)

    # Determine downscale factor. This should be the smallest integer such that
    # the downsampled resolution is <= max_resolution
    downscale_factor = 1
    while height // downscale_factor > max_resolution or width // downscale_factor > max_resolution:
        downscale_factor += 1
    return downscale_factor


def resolve_relative_paths_in_transforms_json(transforms_json_path: Path) -> None:
    # Makes all image paths in a transforms json absolute, by resolving relative paths where necessary.
    # We need this because we get paths from the user that are relative to the cwd, but nerfstudio likes to have
    # paths that are either absolute or are relative to its data dir.
    with open(transforms_json_path, 'r') as f:
        transforms_json = json.load(f)

    # Construct a dictionary mapping from file paths currently in the json to resolved paths:
    image_file_path_remapping = {}
    for frame in transforms_json['frames']:
        file_path = Path(frame['file_path'])
        if not file_path.is_absolute():
            absolute_path = str(file_path.resolve())
            image_file_path_remapping[frame['file_path']] = absolute_path
            # Apply remapping to the frame:
            frame['file_path'] = absolute_path
        else:
            # remapping will be a no-op (path is already absolute)
            image_file_path_remapping[frame['file_path']] = frame['file_path']


    # Apply remapping to splits:
    for split_key in ['train_filenames', 'test_filenames']:
        for i, filename in enumerate(transforms_json[split_key]):
            transforms_json[split_key][i] = image_file_path_remapping[filename]

    # Overwrite the json with the remapped paths:
    with open(transforms_json_path, 'w') as f:
        json.dump(transforms_json, f, indent=4)


def downscale_images(nerf_data_path: Path, downscale_factor: int) -> None:
    # Edge case: if downscale_factor is 1, then we don't need to do anything
    if downscale_factor == 1:
        print('Not downscaling images!')
        return

    # Load all frames
    with open(nerf_data_path / 'transforms.json', 'r') as f:
        transforms_json = json.load(f)

    # With blender-type data, the convention is that if the images are in a folder called 'images',
    # then the downsampled images should be in a folder called 'images_N' where N is the downscale factor
    # Make the output dir
    downsampled_images_path = nerf_data_path / f'images_{downscale_factor}'
    downsampled_images_path.mkdir(exist_ok=True)
    print(f'Downscaling images to output path {downsampled_images_path}')

    # Will be used to store mappings from the original image path to the downscaled image path
    # This is necessary to update the train_filenames and test_filenames in the json to point at the new paths,
    #   so that they are consistent with file_paths in the frames.
    frame_remappings = {}

    # Downscale all images; they could be either jpg or png
    for frame in transforms_json['frames']:
        file_path = Path(frame['file_path'])
        print('Downscaling', file_path)
        image = Image.open(file_path)
        image = image.resize((image.width // downscale_factor, image.height // downscale_factor))

        # Nerfstudio requires a flattened filestructure if we are using downscaling.
        # But if we naively do that then images with the same names in different subdirs clobber each other
        # (see below sanity check).
        # So instead we rename slashes to underscores in the relative path, which gives us a flattened structure
        # without the clobbering.
        output_file_path = downsampled_images_path / file_path.as_posix().replace('/', '_')

        # Rename the frame path in the json to point to the downsampled image, and store the mapping
        frame_remappings[frame['file_path']] = str(output_file_path)
        frame['file_path'] = str(output_file_path)

        # Sanity check: output image file should not already exist! Prevents issues if we have images with the same name
        # in different folders, which would clobber each other if we didn't do this
        assert not output_file_path.exists(), f'Internal error: output file {output_file_path} already exists'
        image.save(output_file_path)
        print('Downscaled image', file_path, 'to', output_file_path)

    # Also need to overwrite the test/train filenames in the json
    for frame_key in ['train_filenames', 'test_filenames']:
        for i, filename in enumerate(transforms_json[frame_key]):
            transforms_json[frame_key][i] = frame_remappings[filename]

    # Write the modified json back out
    with open(nerf_data_path / 'transforms.json', 'w') as f:
        json.dump(transforms_json, f, indent=4)


def should_preload_images(json_path: Path, max_frames_to_preload: int = 3500) -> bool:
    # We want to preload images only if there are not too many frames, to prevent OOMs.
    # So we load the transforms json and check whether the max of train & test frames is <= max_frames_to_preload
    with open(json_path, 'r') as f:
        data = json.load(f)
    max_num_frames = max(len(data['train_filenames']), len(data['test_filenames']))
    print('Max num frames is', max_num_frames)
    print('=> Preloading images?', max_num_frames <= max_frames_to_preload)
    return max_num_frames <= max_frames_to_preload


def enforce_eval_num_images(json_path: Path, num_images: int):
    with open(json_path, 'r') as f:
        data = json.load(f)
    print('Enforcing eval num images to', num_images)
    print('Currently there are ', len(data['test_filenames']), 'test images')
    data['test_filenames'] = data['test_filenames'][:num_images]
    print('Now there are ', len(data['test_filenames']), 'test images')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def sanity_check_transforms_json(json_path: Path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Verify that we have at least one train image and at least one test image
    # (otherwise nerfstudio gives mysterious errors)
    assert len(data['train_filenames']) > 0
    assert len(data['test_filenames']) > 0
