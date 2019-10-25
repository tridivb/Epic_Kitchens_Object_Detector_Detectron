#!/bin/bash

########################################################
# Script to run inference on epic kitchens             #
# Author: tridivb                                      #
########################################################

source ../config/cfg.conf

docker run --rm --gpus $gpus \
	--name detectron \
    --workdir "/home/$USER/Epic_Kitchens_Feature_Extractor_Detectron" \
	-v $code_repo:"/home/$USER/Epic_Kitchens_Feature_Extractor_Detectron" \
	-v $epic_kitchens_path:"/home/$USER/epic_kitchens" \
	-v $output_path:"/home/$USER/detections" \
	-it detectron:py2-caffe2-cuda9 \
	python infer_epic_kitchens.py
