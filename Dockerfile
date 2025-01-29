FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y wget \
    git libnvidia-egl-wayland1 \
    xvfb mesa-utils libgl1-mesa-glx \
    libgl1-mesa-dri libglib2.0-0

# Install Miniforge
RUN wget \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniforge3-Linux-x86_64.sh -b \
    && rm -f Miniforge3-Linux-x86_64.sh

RUN git clone https://github.com/nianticlabs/acezero --recursive

RUN conda install ipykernel

# Create ace0 env
RUN cd acezero && \
    conda env create -f environment.yml

RUN echo "source activate ace0" > ~/.bashrc && \
    conda run -n ace0 pip install ipykernel && \
    conda install -n ace0 -c conda-forge libstdcxx-ng && \
    /opt/conda/envs/ace0/bin/python -m ipykernel install --user --name=ace0

RUN cd acezero/dsacstar && \
    conda run -n ace0 python setup.py install

# Set up Xvfb
ENV DISPLAY=:99
RUN Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99

ENV TCNN_CUDA_ARCHITECTURES="50;52;60;61;70;75;80;86"

# Install nerfstudio
RUN conda create --name nerfstudio -y python=3.8 && \
    conda run -n nerfstudio pip install --upgrade pip && \
    conda run -n nerfstudio pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    conda run -n nerfstudio conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit && \
    conda run -n nerfstudio pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch && \
    conda run -n nerfstudio pip install nerfstudio && \
    conda run -n nerfstudio ns-install-cli

WORKDIR /workspace/acezero