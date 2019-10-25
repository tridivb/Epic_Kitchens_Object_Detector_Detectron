#!/bin/bash

########################################################
# Script to build the docker image                     #
# Author: tridivb                                      #
########################################################

# Check if the image already exists
if [ "$(docker images -q detectron:py2-caffe2-cuda9 | wc -l)" ] >0; then
    echo "Image already exists. Image will be rebuilt in case of any changes."
    image_id=$(docker images -q detectron:py2-caffe2-cuda9)
else
    image_id=0
fi

# Build the image (if it exists, on top of the previous one)
echo "Building docker image..."
docker build --build-arg USER=$USER \
    --build-arg UID=$UID \
    -t detectron:py2-caffe2-cuda9 \
    .

# Remove dangling images
echo "Removing dangling images..."
docker image prune -f
