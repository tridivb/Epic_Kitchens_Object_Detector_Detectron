# Epic Kitchens Object Feature Extractor with Detectron

Detect Objects in the Epic Kitchens dataset using Faster-RCNN as the backbone and the Detectron library 

## Getting Started

Clone the repo and set it up in your local drive.

```
git clone git@github.com:tridivb/Object_Feature_Extractor_with_Detectron.git
```

Also make sure docker is installed and set up with proper access rights. Detectron will be built inside the docker container.

### Prerequisites

Docker >= 19.0
Nvidia Graphics Drivers compatible with Cuda 9
The Epic Kitchens should be downloaded and the folder hierarchy should be in the following way:

```
|---<path to epic kitchens>
|   |---EPIC_KITCHENS_2018
|   |   |---frames_rgb_flow
|   |   |---videos
|   |   |   |---test
|   |   |   |   |---P01
|   |   |   |   |   |---P01_11.MP4
|   |   |   |   |   |---.
|   |   |   |   |   |---.
|   |   |   |---train
|   |   |   |   |---P01
|   |   |   |   |   |---P01_01.MP4
|   |   |   |   |   |---.
|   |   |   |   |   |---.

```

### Installing

Navigate to the Epic_Kitchens_Feature_Extractor_Detectron directory

```
git clone git@github.com:tridivb/Epic_Kitchens_Feature_Extractor_Detectron.git
cd Epic_Kitchens_Feature_Extractor_Detectron/docker
```

Download the pre-trained [weights](http://iplab.dmi.unict.it/rulstm/downloads/ek18-2gpu-e2e-faster-rcnn-R-101-FPN_1x.pkl) and move it to 
./Epic_Kitchens_Feature_Extractor_Detectron/docker directory

Setup the docker image

```
./setup.sh
```

Modify the paths in the config/cfg.conf file for the current code repo, 
the path to the epic kitchens directory and the output directory to save the detections.

Check if the setup was successful by running the script launch.sh

```
./launch.sh
```

If everything works, move on to setting up the parameters.

### Configure the paramters

Set the paths and parameters in the utils/settings.py file.

Alternatively create a new file utils/local_settings.py, copy the contents
of settings.py into it except the following lines and make the changes there. The modified paths should be automatically 
included then.
```
try:
    from utils.local_settings import *
except Exception:
    pass
```

### Extracting the features and detections

An automated script infer.sh is provided to run the inference on the Epic Kitchens dataset. To run the inference, simply run
this script. If you want to run inference on a selected batch of videos, modify the misc/vid_list.txt file accordingly.
Please make sure that the format of the file names are as follows:
```
test/P01/P01_11
train/P01/P01_10
```

If you want to run the inference manually from the container, execute the launch.sh script, navigate to 
./Epic_Kitchens_Feature_Extractor_Detectron directory and run the following command:

```
python infer_epic_kicthens.py
```

### Results

The detections are saved in the following format for each video:

```
|---<path to output>
|   |---test
|   |   |---P01
|   |   |   |---P01_11_detections.npy
|   |   |   |   |   |---.
|   |   |   |   |   |---.
|   |---test
|   |   |---P01
|   |   |   |---P01_01_detections.npy
|   |   |   |   |   |---.
|   |   |   |   |   |---.
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

1. The docker file was built on top of that provided by Facebook to build [Detectron](https://github.com/facebookresearch/Detectron/blob/master/docker/Dockerfile)

2. The script for inference was modified from the one provided by Antonio Furnari for his paper ["What Would You Expect? Anticipating Egocentric Actions with Rolling-Unrolling LSTMs and Modality Attention"](https://iplab.dmi.unict.it/rulstm/) and also the Faster-RCNN [weights](http://iplab.dmi.unict.it/rulstm/downloads/ek18-2gpu-e2e-faster-rcnn-R-101-FPN_1x.pkl) trained on Epic-Kitchens were provided by him.