#!/bin/bash

########################################################
# Script to clean up the docker images                 #
# Author: tridivb                                      #
########################################################

# Remove stopped containers
docker rm $(docker ps -a -q)

# Remove image
docker rmi detectron:py2-caffe2-cuda9