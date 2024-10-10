#!/usr/bin/env bash

mapping_exe="train_ace.py"
register_exe="register_mapping.py"
export_pc_exe="export_point_cloud.py"

# folder with image files
datasets_folder="datasets/7scenes"
# output directory for the reconstruction
out_dir="reconstructions/7scenes_warmstart"
# target directory for benchmarking results
benchmarking_out_dir="benchmark/7scenes_warmstart"

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

# 7Scenes comes with a pre-defined train/test split, split files expected in this folder
# generate using create_7scenes_split.py
benchmarking_split_folder="split_files"

scenes=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")

for scene in ${scenes[*]}; do
  # find color images for the reconstruction
  input_rgb_files="${datasets_folder}/${scene}/seq-*/*.color.png"
  # find initial poses from KinectFusion
  input_pose_files="${datasets_folder}/${scene}/seq-*/*.pose.txt"
  scene_out_dir="${out_dir}/${scene}"

  if $render_visualization; then
    visualization_cmd="--render_visualization True --render_target_path ${scene_out_dir}/renderings --render_marker_size 0.02"
  else
    visualization_cmd="--render_visualization False"
  fi

  mkdir -p ${scene_out_dir}

  # network name set to a particular pattern to make sure the visualization works
  network_name="iteration0"

  # run ACE mapping with pose refinement enabled
  python ${mapping_exe} "${input_rgb_files}" ${scene_out_dir}/${network_name}.pt --pose_files "${input_pose_files}" ${visualization_cmd} --use_external_focal_length 525 --refine_calibration True --pose_refinement mlp --pose_refinement_wait 5000

  #extract focal length from pose file
  output_pose_file=${scene_out_dir}/poses_${network_name}_preliminary.txt
  focal_length=$(awk '{if (NR==1) print $9}' ${output_pose_file})
  echo "Using focal length from mapping stage: ${focal_length}"

  # re-estimate all poses
  python ${register_exe} "${input_rgb_files}" ${scene_out_dir}/${network_name}.pt ${visualization_cmd} --use_external_focal_length ${focal_length} --session ${network_name}

  # make a copy of the final pose file
  cp ${scene_out_dir}/poses_iteration0.txt ${scene_out_dir}/poses_final.txt

  # render final video if requested
  if $render_visualization; then
    python render_final_sweep.py ${scene_out_dir}/renderings --render_marker_size 0.02
    /usr/bin/ffmpeg -y -framerate 30 -pattern_type glob -i "${scene_out_dir}/renderings/*.png" -c:v libx264 -pix_fmt yuv420p ${scene_out_dir}/refinement.mp4
  fi

  # run benchmarking if requested
  if $run_benchmark; then
    benchmarking_scene_dir="${benchmarking_out_dir}/${scene}"
    mkdir -p ${benchmarking_scene_dir}

    # when using splatfacto for benchmarking, export point cloud
    if [ "${benchmarking_method}" = "splatfacto" ]; then
      # if visualisation is enabled, we can export the sparse point cloud from the existing visualisation buffer
      if ${render_visualization} && ! ${benchmarking_dense_pcinit}; then
        python ${export_pc_exe} ${scene_out_dir}/pc_final.ply --visualization_buffer ${scene_out_dir}/renderings/${network_name}_mapping.pkl --convention opencv
      else
        python ${export_pc_exe} ${scene_out_dir}/pc_final.ply --network ${scene_out_dir}/${network_name}.pt --pose_file ${scene_out_dir}/poses_final.txt --convention opencv --dense_point_cloud ${benchmarking_dense_pcinit}
      fi
    fi

    conda run --no-capture-output -n ${benchmarking_environment} python -m benchmarks.benchmark_poses --pose_file ${scene_out_dir}/poses_final.txt --output_dir ${benchmarking_scene_dir} --images_glob_pattern "${input_rgb_files}" --split_json ${benchmarking_split_folder}/7scenes_${scene}.json --method ${benchmarking_method} 2>&1 | tee ${benchmarking_out_dir}/log_${scene}.txt
  fi
done