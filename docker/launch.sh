code_repo = "/media/remote_home/tridiv/Object_Feature_Extractor_with_Detectron"
epic_kitchens_path = "/media/data/tridiv/epic_kitchens"
output_path = "/media/data/tridiv/detections"

docker run --rm --gpus "device=1" \
	--name epic \
	-v $code_repo:"/home/$USER/Object_Feature_Extractor_with_Detectron" \
	-v $epic_kitchens_path:"/home/$USER/epic_kitchens" \
	-v $output_path:"/home/$USER/detections" \
	-it epic_kitchens:py2-caffe2-cuda9 \
	/bin/bash
