# ACE Zero pose evaluation module

This module allows you to benchmark the quality of some poses by training a NeRF on some of them, and using the rest to evaluate the novel view synthesis quality of the NeRF.

## Requirements

You must have Nerfstudio installed; follow the instructions [here](https://docs.nerf.studio/quickstart/installation.html) to do this. 
We recommend to install Nerfstudio in its own Conda environment as described in their installation instructions.
Firstly, this prevents library conflicts between Nerfstudio and ACE0.
Secondly, our benchmarking scripts assume that Nerfstudio lives in an environment called `nerfstudio`.

**Note:** All paper results were produced with Nerfstudio v0.3.4. Since then, we updated this repository to support newer version of Nerfstudio.
We verified that benchmark results did not change significantly when updating to Nerfstudio v1.1.4. 
However, if you observe benchmarking inconsistencies w.r.t. the paper, we advise to first down-grade to Nerfstudio v0.3.4 and checkout our code using the "eccv_2024_checkpoint" git tag.

## Running the benchmark

The main entrypoint for this is benchmarks/benchmark_poses.py, which takes a directory of poses and a dataset, and trains a NeRF on the poses and evaluates it on the rest of the poses. Call it using `python -m benchmarks.benchmark_poses <ARGS>`, where the arguments should be as follows:

`--pose_file`: Must point to an ACE0 pose file. This is a text file with one line per pose, containing the following entries for each line:
`image_path qw qx qy qz tx ty tz focal_length confidence_score`
These are:
- The image path relative to the working directory.
- The camera extrinsics as a world-to-camera transform, using the OpenCV camera convention (x right, y down, z forward), expressed as a rotation quaternion and a translation.
- The focal length of the camera in pixels.
- A confidence score for the pose. If it is less than 1000, the pose will be excluded from the NeRF training set (but not the test set).

`--output_dir`: Output directory for the benchmark. Downsampled images, NeRFs and evaluation scores will be written to this dir.
Will also contain Nerfstudio input data -- in particular a standard Nerfstudio transforms.json containing poses in OpenGL cam-to-world format,
and possibly an images directory containing downsampled images (if downsampling was performed).

`--images_glob_pattern`: Defines the set of images used in the dataset. Must be given as a glob pattern relative to the current working directory, e.g. `--images_glob_pattern '/path/to/dataset/*.jpg'`. (Note that the single quotes are important to prevent the glob being expanded prematurely by your shell!)

`--split_json`: Optional. Path to a JSON file defining a train/test split for NeRF evaluation, in the format:
```
{
    "train_filenames": ["image1.jpg", "image2.jpg", ...],
    "test_filenames": ["image3.jpg", "image4.jpg", ...]
}
```
If not given, a default split will be used in which every 8th frame (after sorting them alphabetically on their paths) will form the test set.

`--no_run_nerfstudio`: Optional argument. If given, will stop after generating Nerfstudio inputs without actually running Nerfstudio or conducting the evaluation. Useful if you just want to convert your ACE0-format pose file to Nerfstudio input format, rather than evaluating the pose quality.

`--method`: Optional argument. Pass `splatfacto`if you want to train Gaussian splats instead of the default Nerfacto model. For a splatfacto model, the benchmarking scripts will be looking for a point cloud file `pc_final.ply` in the same folder as the pose file. This file can be generated using `ace_zero.py` with the `--export_point_cloud True` option or the utility script `export_point_cloud.py`.

After the eval runs, there will be an evaluation JSON in the output dir containing PSNR (and other metrics). The output structure is quite nested:
`$OUTPUT_DIR/nerf_data/nerf_for_eval/nerfacto/run/eval.json`

Where OUTPUT_DIR is whatever you passed via the `--output_dir` argument above.

The images will be downscaled as necessary to a max side length of 640 pixels for NeRF training.
This is because NeRF quality and training speed can degrade dramatically with high-resolution images.

## FAQs

### Nerfstudio is running out of VRAM

We decide whether to cache images on GPU during NeRF fitting by calling 'should_preload_images' in `benchmarks/benchmark_poses.py`.
If you're using a GPU with not much VRAM, you might like to try a lower value for the `max_frames_to_preload` parameter than the default (which is 3500).
