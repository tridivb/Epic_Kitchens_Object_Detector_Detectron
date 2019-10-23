# Object_Feature_Extractor_with_Detectron

#### First Setup the repo in your local drive
```
git clone git@github.com:tridivb/Object_Feature_Extractor_with_Detectron.git
```

Then download the weights from http://iplab.dmi.unict.it/rulstm/downloads/ek18-2gpu-e2e-faster-rcnn-R-101-FPN_1x.pkl and save it under "./Object_Feature_Extractor_with_Detectron/weights" directory.

#### Build your docker image and start the container
Modify the paths in "./Object_Feature_Extractor_with_Detectron/docker/launch.sh" as required for the code repo, epic_kitchens and output paths.
The docker image will be formed with the same username and uid as the current one being used in the host machine.

```
cd ./Object_Feature_Extractor_with_Detectron/docker
./setup.sh
./launch.sh
```

#### Once inside the docker container, run the inference as:

```
cd ./Object_Feature_Extractor_with_Detectron
python infer_epic_kitchens.py --cfg config/ek18-2gpu-e2e-faster-rcnn-R-101-FPN_1x.yaml --wts weights/ek18-2gpu-e2e-faster-rcnn-R-101-FPN_1x.pkl
```