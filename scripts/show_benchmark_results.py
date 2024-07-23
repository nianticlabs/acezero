import argparse
import os
import json
from pathlib import Path

# Parse command line argument for folder name
parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path,
                    help="Folder containing Nerfacto bechmark results with subfolders for each scene.")
args = parser.parse_args()

# metrics to report from the eval.json file
keys = ['psnr', 'ssim', 'lpips']

# get list of scenes as sub folders of benchmarking folder
scene_folders = [f for f in args.folder.iterdir() if f.is_dir()]

# Print header
header_str = "Scene: "
for key in keys:
    header_str += key + " "
print(header_str)

# Loop through scenes of dataset
for scene_folder in scene_folders:
    # Specify result file
    result_file = scene_folder / 'nerf_data/nerf_for_eval/nerfacto/run/eval.json'

    # Assemble an output string with the benchmarking results
    out_str = scene_folder.name + ": "

    # Check whether result file exists
    if not os.path.exists(result_file):
        out_str += "Results not found."
    else:
        # Load results
        with open(os.path.join(result_file), 'r') as f:
            data = json.load(f)

            # Print all requested values
            for key in keys:
                if key in data['results']:
                    out_str += str(data['results'][key]) + " "
                else:
                    out_str += "Invalid Key "

    print(out_str)