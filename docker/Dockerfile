# Modified as per usage

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

# Arguments
ARG USER=docker
ARG UID=1000

# Install linux packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	ca-certificates python3-dev git wget sudo curl \
    gcc \
    g++ \
    autoconf \
    automake \
    build-essential \
    cmake \
    make \
    vim \
    libatlas-base-dev \
    gfortran && \
    rm -rf /var/lib/apt/lists/*

# Setup conda
RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
RUN chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

RUN conda init
RUN conda install -y \
    python=3.7 numpy pyyaml scipy ipython mkl mkl-include ninja cython typing ninja setuptools cmake cffi typing && \
    conda clean -ya
RUN pip install opencv-python

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install 'git+https://github.com/facebookresearch/fvcore'

# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
ENV FORCE_CUDA="1"
# This will build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
RUN pip install -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

RUN useradd -m --no-log-init --system  --uid ${UID} ${USER} -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ${USER}
ENV HOME /home/${USER}
WORKDIR /home/${USER}
RUN mkdir /home/${USER}/Epic_Kitchens_Feature_Extractor_Detectron /home/${USER}/epic_kitchens