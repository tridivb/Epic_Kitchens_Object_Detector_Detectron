#!/bin/bash

########################################################
# Script to clean up the docker images                 #
# Author: tridivb                                      #
########################################################

# Remove stopped containers
docker rm $(docker ps -a -q)

# Remove image
docker rmi detectron2:py3-cuda10.1