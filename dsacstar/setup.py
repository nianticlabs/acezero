from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os
from pathlib import Path
import torch

opencv_inc_dir = os.environ.get("OPENCV_INC_DIR", "")  # directory containing OpenCV header files
opencv_lib_dir = os.environ.get("OPENCV_LIB_DIR", "")  # directory containing OpenCV library files

# If not explicitly provided, try to locate OpenCV in the active conda/pixi environment.
conda_env = os.environ.get("CONDA_PREFIX", "")
if (len(opencv_inc_dir) == 0 or len(opencv_lib_dir) == 0) and len(conda_env) > 0:
	print("Detected active conda environment:", conda_env)
	
	if len(opencv_inc_dir) == 0:
		candidate_inc_dirs = [
			Path(conda_env) / "include" / "opencv4",
			Path(conda_env) / "include",
		]
		for candidate in candidate_inc_dirs:
			if candidate.is_dir():
				opencv_inc_dir = str(candidate)
				break

	if len(opencv_lib_dir) == 0:
		candidate_lib_dirs = [
			Path(conda_env) / "lib" / "opencv4",
			Path(conda_env) / "lib",
		]
		for candidate in candidate_lib_dirs:
			if candidate.is_dir():
				opencv_lib_dir = str(candidate)
				break

	print("Assuming OpenCV dependencies in:")
	print(opencv_inc_dir)
	print(opencv_lib_dir)
elif (len(opencv_inc_dir) == 0 or len(opencv_lib_dir) == 0) and len(conda_env) == 0:
	print(
		"Error: OpenCV paths not set and CONDA_PREFIX is not set.\n"
		"Either activate a conda/pixi environment (CONDA_PREFIX) or set OPENCV_INC_DIR and OPENCV_LIB_DIR."
	)
	exit(1)

if len(opencv_inc_dir) == 0:
	print("Error: You have to provide an OpenCV include directory. Edit this file.")
	exit()
if len(opencv_lib_dir) == 0:
	print("Error: You have to provide an OpenCV library directory. Edit this file.")
	exit()

torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"

setup(
	name='dsacstar',
	ext_modules=[CppExtension(
		name='dsacstar', 
		sources=['dsacstar.cpp','thread_rand.cpp'],
		include_dirs=[opencv_inc_dir],
		library_dirs=[opencv_lib_dir],
		libraries=['opencv_core','opencv_calib3d'],
		extra_compile_args=['-fopenmp'],
		extra_link_args=['-fopenmp', f'-Wl,-rpath,{torch_lib_dir}'],
		)],		
	cmdclass={'build_ext': BuildExtension})
