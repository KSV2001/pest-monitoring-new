# build the image required for setting up the repository
# Example run:
# $ docker build -t wadhwaniai/pest-monitoring-detectron2:v0 .
# Creates a docker image with desired dependencies

FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build tmux vim
RUN ln -sv /usr/bin/python3 /usr/bin/python

# extra line
WORKDIR /
ENV PATH="/root/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard cmake   # cmake from apt-get is too old
RUN pip install --user torch==1.8 torchvision==0.9 -f https://download.pytorch.org/whl/cu101/torch_stable.html

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
# RUN git clone https://github.com/facebookresearch/detectron2 detectron2

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# RUN pip install --user -e detectron2

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /workspace/pest-monitoring-new

# set the PYTHONPATH required for using the repository
ENV PYTHONPATH /workspace/pest-monitoring-new

# install from requirements.txt
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/
