# Use Caffe2 image as parent image
FROM caffe2/caffe2:snapshot-py2-cuda9.0-cudnn7-ubuntu16.04

ARG USER=docker
ARG UID=1000
ARG GID=1000
# default password for user
ARG PW=docker

#setup user
RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd
RUN usermod -aG sudo ${USER}
USER ${UID}:${GID}
WORKDIR /home/${USER}
ENV HOME /home/${USER}

RUN echo ${PW} | sudo -S mv /usr/local/caffe2 /usr/local/caffe2_build
ENV Caffe2_DIR /usr/local/caffe2_build

ENV PYTHONPATH /usr/local/caffe2_build:${PYTHONPATH}
ENV LD_LIBRARY_PATH /usr/local/caffe2_build/lib:${LD_LIBRARY_PATH}

# Clone the Detectron repository
RUN git clone https://github.com/tridivb/Detectron.git /home/${USER}/detectron
WORKDIR /home/${USER}/detectron
RUN git checkout feature_extractor

# Install Python dependencies
COPY requirements.txt /home/${USER}/requirements.txt
RUN pip install -r requirements.txt --user
RUN pip install -r requirements.txt --user

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /home/${USER}/cocoapi
WORKDIR /home/${USER}/cocoapi/PythonAPI
RUN echo ${PW} | sudo -S make install

# Go to Detectron root
WORKDIR /home/${USER}/detectron

# Set up Python modules
RUN make

# [Optional] Build custom ops
RUN make ops

#setup rulstm 
#RUN git clone https://github.com/fpv-iplab/rulstm rulstm
#COPY ek18-2gpu-e2e-faster-rcnn-R-101-FPN_1x.pkl ./rulstm/FasterRCNN/weights/ek18-2gpu-e2e-faster-rcnn-R-101-FPN_1x.pkl

# install ffmpeg
RUN echo ${PW} | sudo -S apt-get update && sudo apt-get install -y vim ffmpeg

# set up the directories for volume
WORKDIR /home/${USER}
RUN mkdir ./epic_kitchens ./Object_Feature_Extractor_with_Detectron
