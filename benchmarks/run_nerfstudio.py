from pathlib import Path
import subprocess
from typing import Optional, Dict


def run_command(cmd):
    process = subprocess.run(cmd, shell=True, check=True)
    return process.returncode


def export_point_cloud_from_nerfstudio(config_path: Path, output_folder: Path) -> None:
    nerfstudio_args = {
        'load-config': config_path,
        'output-dir': output_folder,
        'num-points': 1000000,
        'remove-outliers': 'True',
        'normal-method': 'open3d',
        'use-bounding-box': 'False',
    }

    cmd = 'ns-export pointcloud ' + ' '.join([f'--{k} {v}' for k, v in nerfstudio_args.items()])
    print(f'Running command: {cmd}')
    run_command(cmd)


def fit_nerf_with_nerfstudio(nerf_data_path: Path, downscale_factor: Optional[int] = 1,
                             preload_images: bool = False, ns_train_extra_args: Optional[Dict] = None,
                             method: str = 'nerfacto', camera_optimizer: str = 'off') -> Path:
    ns_train_extra_args = ns_train_extra_args or {}

    # Build command
    # Check output dir doesn't already exist
    # Nerfstudio unfortunately forces this deeply nested directory structure on us:
    output_dir = nerf_data_path / 'nerf_for_eval' / method / 'run'

    # For eval purposes, it's bad news if we end up accidentally resuming a previous run
    # So we'll raise an exception if the output dir exists already
    print('Checking existence of output dir', output_dir)
    if output_dir.exists():
        raise ValueError(f'Output dir {output_dir} already exists. Aborting.')

    nerfstudio_args = {
        'data': nerf_data_path,
        'pipeline.model.camera-optimizer.mode': camera_optimizer,
        'pipeline.datamanager.images-on-gpu': str(preload_images),
        'method-name': method,
        'experiment_name': 'nerf_for_eval',
        'output-dir': nerf_data_path,
        'timestamp': 'run',
        'viewer.quit-on-train-completion': 'True',
        **ns_train_extra_args
    }

    dataparser_args = {
        'downscale-factor': downscale_factor,
    }

    # Convert dict of args to list of strings
    cmd = 'ns-train ' + method + ' ' + ' '.join([f'--{k} {v}' for k, v in nerfstudio_args.items()])
    cmd += ' nerfstudio-data ' + ' '.join([f'--{k} {v}' for k, v in dataparser_args.items()])

    # Execute
    print(f'Running command: {cmd}')
    run_command(cmd)

    # Return path to fitted nerf
    assert output_dir.exists(), 'Internal error'
    return output_dir


def eval_nerf_with_nerfstudio(nerf_output_dir: Path) -> Path:
    # Build command
    nerfstudio_args = {
        'load-config': nerf_output_dir / 'config.yml',
        'output-path': nerf_output_dir / 'eval.json',
        'render-output-path': nerf_output_dir / 'renders',
    }

    # Convert dict of args to list of strings
    cmd = 'ns-eval ' + ' '.join([f'--{k} {v}' for k, v in nerfstudio_args.items()])

    # Execute
    print(f'Running command: {cmd}')
    run_command(cmd)
    return nerf_output_dir / 'eval.json'
