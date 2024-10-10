#!/usr/bin/env bash

mapping_exe="train_ace.py"
reconstruction_exe="ace_zero.py"

# input folder with T2 images and COLMAP reconstruction
datasets_folder_sparse="datasets/t2_colmap/training"
# input folder with T2 video frames
datasets_folder_video="datasets/t2/training_videos"

# output directory for the reconstruction
out_dir="reconstructions/t2_training_videos_warmstart"
# target directory for benchmarking results
benchmarking_out_dir="benchmark/t2_training_videos_warmstart"

# render visualization of the reconstruction, slows down reconstruction considerably
render_visualization=false

# run view synthesis benchmarking after reconstruction
run_benchmark=true
# benchmarking needs to happen in the nerfstudio environment
benchmarking_environment="nerfstudio"
# benchmarking method, splatfacto or nerfacto
benchmarking_method="nerfacto"
# when using splatfacto, we need a point cloud initialization, either sparse or dense
# dense is recommended if you have very dense coverage of the scene, eg. > 2000 images
benchmarking_dense_pcinit=true

scenes=("Barn" "Caterpillar" "Church" "Ignatius" "Meetingroom" "Truck") # "Courthouse"

for scene in ${scenes[*]}; do
  # start with data from COLMAP, images and poses
  input_rgb_files_sparse="${datasets_folder_sparse}/${scene}/*.jpg"
  input_pose_files_sparse="${datasets_folder_sparse}/${scene}/*_pose.txt"
  scene_out_dir="${out_dir}/${scene}"

  # read COLMAP focal length
  calibration_file="${datasets_folder_sparse}/${scene}/focal_length.txt"
  focal_length=$(cat ${calibration_file})
  echo "Using focal length from COLMAP stage: ${focal_length}"

  visualization_cmd="--render_visualization ${render_visualization}"

  if ${run_benchmark} && [ "${benchmarking_method}" = "splatfacto" ]; then
    export_pc_cmd="--export_point_cloud True --dense_point_cloud ${benchmarking_dense_pcinit}"
  else
    export_pc_cmd="--export_point_cloud False --dense_point_cloud False"
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
  python $reconstruction_exe "${input_rgb_files_video}" ${scene_out_dir} --seed_network ${scene_out_dir}/${network_name}.pt  ${visualization_cmd} --use_external_focal_length ${focal_length} --refine_calibration False ${export_pc_cmd} 2>&1 | tee ${scene_out_dir}/log_${scene}.txt

  # run benchmarking if requested
  if $run_benchmark; then
    benchmarking_scene_dir="${benchmarking_out_dir}/${scene}"
    mkdir -p ${benchmarking_scene_dir}
    conda run --no-capture-output -n ${benchmarking_environment} python -m benchmarks.benchmark_poses --pose_file ${scene_out_dir}/poses_final.txt --output_dir ${benchmarking_scene_dir} --images_glob_pattern "${input_rgb_files_video}" --method ${benchmarking_method} 2>&1 | tee ${benchmarking_out_dir}/log_${scene}.txt
  fi
done