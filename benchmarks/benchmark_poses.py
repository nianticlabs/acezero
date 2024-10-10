import argparse
from pathlib import Path

from benchmarks.run_benchmark import run_benchmark


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Benchmark some poses by fitting a NeRF. Consult the README.md for more info.'
    )
    parser.add_argument('--pose_file', type=str, required=True, help='Path to the poses file, in ACE0 format. '
                        ' Poses with confidence <1000 will be excluded from the training set.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory where the benchmark results will be written')
    parser.add_argument('--images_glob_pattern', type=str, required=True,
                        help='Pattern relative to working directory to glob for images')
    parser.add_argument('--split_json', type=str, required=False,
                        help='Path to a JSON file containing splits; if not given, every 8 images will be test images')
    parser.add_argument('--no_run_nerfstudio', action='store_true',
                        help='If given, the script will generate Nerfstudio input files but not run Nerfstudio')
    parser.add_argument('--method', type=str, default='nerfacto', choices=['nerfacto', 'splatfacto'],
                        help='Method to use for fitting NeRF')
    parser.add_argument('--camera_optimizer', type=str, default='off', choices=['off', 'SO3xR3', 'SE3'],
                        help='Type of camera optimizer to use, might improve quality but break the benchmark')
    parser.add_argument('--max_resolution', type=int, default=640,
                        help='Maximum resolution of the images to use for the benchmark')
    args = parser.parse_args()

    run_benchmark(
        pose_file=Path(args.pose_file),
        working_dir=Path(args.output_dir),
        split_json=Path(args.split_json) if args.split_json else None,
        images_glob_pattern=args.images_glob_pattern,
        dry_run=args.no_run_nerfstudio,
        method=args.method,
        max_resolution=args.max_resolution,
        camera_optimizer=args.camera_optimizer
    )
