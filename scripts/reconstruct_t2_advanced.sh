#!/usr/bin/env bash

reconstruction_exe="ace_zero.py"

# folder with image files
datasets_folder="datasets/t2/advanced"
# output directory for the reconstruction
out_dir="reconstructions/t2_advanced"
# target directory for benchmarking results
benchmarking_out_dir="benchmark/t2_advanced"

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
benchmarking_dense_pcinit=false

scenes=("Auditorium" "Ballroom" "Courtroom" "Palace" "Temple") # "Museum"

for scene in ${scenes[*]}; do
  # find color images for the reconstruction
  input_rgb_files="${datasets_folder}/${scene}/*.jpg"
  scene_out_dir="${out_dir}/${scene}"

  if $render_visualization; then
    visualization_cmd="--render_visualization True --render_marker_size 0.05"
  else
    visualization_cmd="--render_visualization False"
  fi

  if ${run_benchmark} && [ "${benchmarking_method}" = "splatfacto" ]; then
    export_pc_cmd="--export_point_cloud True --dense_point_cloud ${benchmarking_dense_pcinit}"
  else
    export_pc_cmd="--export_point_cloud False --dense_point_cloud False"
  fi

  mkdir -p ${scene_out_dir}

  # run ACE0 reconstruction
  python $reconstruction_exe "${input_rgb_files}" ${scene_out_dir} --try_seeds 5 ${visualization_cmd} --seed_parallel_workers 5 ${export_pc_cmd} 2>&1 | tee ${scene_out_dir}/log_${scene}.txt

  # run benchmarking if requested
  if $run_benchmark; then
    benchmarking_scene_dir="${benchmarking_out_dir}/${scene}"
    mkdir -p ${benchmarking_scene_dir}
    conda run --no-capture-output -n ${benchmarking_environment} python -m benchmarks.benchmark_poses --pose_file ${scene_out_dir}/poses_final.txt --output_dir ${benchmarking_scene_dir} --images_glob_pattern "${input_rgb_files}" --method ${benchmarking_method} 2>&1 | tee ${benchmarking_out_dir}/log_${scene}.txt
  fi
done