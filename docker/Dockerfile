# Source: https://github.com/facebookresearch/Detectron/blob/master/docker/Dockerfile
# Modified as per usage

# Use Caffe2 image as parent image
FROM caffe2/caffe2:snapshot-py2-cuda9.0-cudnn7-ubuntu16.04

# Arguments
ARG USER=docker
ARG UID=1000

# Install linux packages
RUN apt-get update
RUN apt-get install -y \
    ca-certificates \
    build-essential \
    pkg-config \
    automake \
    libtool \
    ffmpeg \
    git \
    cmake \
    wget \
    curl \
    vim \
    tmux

# Setup caffe2 build environment
RUN mv /usr/local/caffe2 /usr/local/caffe2_build && chmod -R +x /usr/local/caffe2_build
ENV Caffe2_DIR /usr/local/caffe2_build

ENV PYTHONPATH /usr/local/caffe2_build:${PYTHONPATH}
ENV LD_LIBRARY_PATH /usr/local/caffe2_build/lib:${LD_LIBRARY_PATH}

# Clone the Detectron repository
RUN git clone https://github.com/tridivb/Detectron.git /detectron
WORKDIR /detectron
RUN git checkout feature_extractor
RUN chmod -R +x /detectron

# Install Python dependencies
COPY requirements.txt requirements.txt
COPY requirements_others.txt requirements_others.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -r requirements_others.txt
RUN rm -f requirements.txt requirements_others.txt

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi && chmod -R +x /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install

# Build detectron
WORKDIR /detectron
RUN python setup.py develop
# [Optional] Build custom ops
RUN make ops

# Set up code repo [Optional -- can be linked from host via volumes]
#RUN git clone https://github.com/tridivb/Object_Feature_Extractor_with_Detectron.git
#COPY ek18-2gpu-e2e-faster-rcnn-R-101-FPN_1x.pkl ./Object_Feature_Extractor_with_Detectron/FasterRCNN/weights/ek18-2gpu-e2e-faster-rcnn-R-101-FPN_1x.pkl


# Setup user
RUN adduser --disabled-password --gecos "" --uid ${UID} ${USER}
USER ${UID}
WORKDIR /home/${USER}
ENV HOME /home/${USER}
RUN mkdir ./epic_kitchens ./Epic_Kitchens_Feature_Extractor_Detectron