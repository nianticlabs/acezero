#!/usr/bin/env bash

mapping_exe="train_ace.py"
reconstruction_exe="ace_zero.py"

# input folder with T2 images and COLMAP reconstruction
datasets_folder_sparse="datasets/t2_colmap/training"
# input folder with T2 video frames
datasets_folder_video="datasets/t2/training_videos"

# output directory for the reconstruction
out_dir="results_t2_training_videos_warmstart"

# render visualization of the reconstruction, slows down reconstruction considerably
render_visualization=false

# run view synthesis benchmarking after reconstruction
run_benchmark=true
# benchmarking needs to happen in the nerfstudio environment
benchmarking_environment="nerfstudio"
# target directory for benchmarking results
benchmarking_out_dir="${out_dir}_benchmark"

scenes=("Barn" "Caterpillar" "Church" "Courthouse" "Ignatius" "Meetingroom" "Truck")

for scene in ${scenes[*]}; do
  # start with data from COLMAP, images and poses
  input_rgb_files_sparse="${datasets_folder_sparse}/${scene}/*.jpg"
  input_pose_files_sparse="${datasets_folder_sparse}/${scene}/*_pose.txt"
  scene_out_dir="${out_dir}/${scene}"

  # read COLMAP focal length
  calibration_file="${datasets_folder_sparse}/${scene}/focal_length.txt"
  focal_length=$(cat ${calibration_file})
  echo "Using focal length from COLMAP stage: ${focal_length}"

  if $render_visualization; then
    visualization_cmd="--render_visualization True"
  else
    visualization_cmd="--render_visualization False"
  fi

  mkdir -p ${scene_out_dir}

  # network name set to a particular pattern to make sure the visualization works
  network_name="iteration0_seed0"

  # run ACE mapping without pose nor calibration refinement (trust COLMAP for now)
  python ${mapping_exe} "${input_rgb_files_sparse}" ${scene_out_dir}/${network_name}.pt --pose_files "${input_pose_files_sparse}" ${visualization_cmd} --render_target_path "${scene_out_dir}/renderings" --use_external_focal_length ${focal_length} 2>&1 | tee ${scene_out_dir}/log_${scene}_init.txt

  # switch to video frames
  input_rgb_files_video="${datasets_folder_video}/${scene}/*.jpg"

  # adjust COLMAP focal length to change in image resolution of video frames
  focal_length=$(echo "scale=6; ${focal_length} / 2" | bc)
  echo "Adjusted focal length for video frames: ${focal_length}"

  # run ACE0 reconstruction starting with the network from the ACE mapping call
  python $reconstruction_exe "${input_rgb_files_video}" ${scene_out_dir} --seed_network ${scene_out_dir}/${network_name}.pt  ${visualization_cmd} --use_external_focal_length ${focal_length} --refine_calibration False 2>&1 | tee ${scene_out_dir}/log_${scene}.txt

  # run benchmarking if requested
  if $run_benchmark; then
    benchmarking_scene_dir="${benchmarking_out_dir}/${scene}"
    mkdir -p ${benchmarking_scene_dir}
    conda run --no-capture-output -n ${benchmarking_environment} python -m benchmarks.benchmark_poses --pose_file ${scene_out_dir}/poses_final.txt --output_dir ${benchmarking_scene_dir} --images_glob_pattern "${input_rgb_files_video}" 2>&1 | tee ${benchmarking_out_dir}/log_${scene}.txt
  fi
done