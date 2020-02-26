# Epic Kitchens Object Detector and Feature Extractor with Detectron

WIP for Detectron2 and Python3. For the previous working version using python2 and detectron (of the object feature extractor only), refer to the [python2](https://github.com/tridivb/Epic_Kitchens_Feature_Extractor_Detectron/tree/python2) branch.

Detect Objects in the Epic Kitchens dataset using Faster-RCNN as the backbone and the Detectron library.

Once the updates are up, the python2 version will be deprecated.

## TODO
1. Implement batch and multi-gpu processing in inference script. --Done
2. Create pascal-voc evaluator for Epic Kitchens -- In Progress
3. Implement tester/inference script with detectron2 and python3. --Done
4. Provide conda environment file or requirements file to setup python environment. -- Done
5. Modify docker scripts to accomodate updated versions of python and detectron (Currently not working)
6. Create the feature extractor module with detectron2

## Getting Started

Clone the repo and set it up in your local drive.

```
git clone https://github.com/tridivb/Epic_Kitchens_Object_Detector_Detectron.git
```

### Prerequisites

1. Nvidia Graphics Drivers compatible with Cuda 10
2. Python >= 3.6
3. [Pytorch](https://pytorch.org/get-started/locally/) >= 1.3
4. [Detectron2](https://github.com/facebookresearch/detectron2)
\
\
The object frames of the Epic Kitchens Dataset should be downloaded and the folder hierarchy should be in the following way, whis is also the
default hierarchy of the dataset:

```
|---<path to epic kitchens>
|   |---annotations
|   |   |---EPIC_test_s1_object_video_list.csv
|   |   |---EPIC_test_s2_object_video_list.csv
|   |   |---EPIC_train_object_labels.csv
|   |   |---.
|   |   |---.
|   |---EPIC_KITCHENS_2018
|   |   |---object_detection_images
|   |   |   |---rgb
|   |   |   |   |---train
|   |   |   |   |   |---P01
|   |   |   |   |   |   |---P01_01
|   |   |   |   |   |   |   |---frame0000000001.jpg
|   |   |   |   |   |   |   |---.
|   |   |   |   |   |   |   |---.
|   |   |   |   |   |   |---.
|   |   |   |   |   |   |---.
|   |   |   |   |---test
|   |   |   |   |   |---P02
|   |   |   |   |   |   |---P02_12
|   |   |   |   |   |   |   |---frame0000000001.jpg
|   |   |   |   |   |   |   |---.
|   |   |   |   |   |   |   |---.
|   |   |   |   |   |   |---.
|   |   |   |   |   |   |---.

```

### Installing

Navigate to the Epic_Kitchens_Object_Detector_Detectron directory

```
cd <path to repo>/Epic_Kitchens_Object_Detector_Detectron
```

Setup the virtual python environment in your preferred way using virtualenv or conda. 
Then install the required packages using the requirements.txt file.

```
pip install -r requirements.txt
```

Install [Detectron2](https://github.com/facebookresearch/detectron2) by following the instructions provided in [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

Please note, the docker configuration is not yet ready, so it will not work for now.


### Configure the paramters

Set the paths and parameters in the config/faster_rcnn_R_101_FPN_3x.yaml file. 
\
Alternatively, if you want git to ignore the modified config file, copy it and rename the config file as <new_name>_local.yaml.

```
cp config/faster_rcnn_R_101_FPN_3x.yaml config/faster_rcnn_R_101_FPN_3x_local.yaml
```
\
The current set of parameters in the config are similar to the ones provided for the baseline in the [Epic-Kitchens paper](https://arxiv.org/abs/1804.02748).

### Training

The model can be trained on the full training set using the script `train_epic_kitchens.py`.

```
cd <path to repo>/Epic_Kitchens_Object_Detector_Detectron
python train_epic_kitchens.py --config-file config/<config_file>.yaml --root-dir <path_to_Epic_Kitchens_rgb_frames> --ann-dir <path_to_Epic_Kitchens_annotation_files>
```
\
For a comprehensive list of arguments available, run:
```
python train_epic_kitchens.py --help
```

The script does not yet support custom training and validation.

### Extracting the features and detections

The script `infer_epic_kitchens.sh` is provided to run the inference on the test object frames of Epic Kitchens. To run the inference, simply run
this script as follows:

```
cd <path to repo>/Epic_Kitchens_Object_Detector_Detectron
python infer_epic_kitchens.py --config-file config/<config_file>.yaml --root-dir <path_to_Epic_Kitchens_rgb_frames> --ann-dir <path_to_Epic_Kitchens_annotation_files>
```
\
The script does not yet support running inference on a custom set of videos. However you can always modify the annotation files 
`EPIC_test_s1_object_video_list.csv` and/or `EPIC_test_s2_object_video_list.csv` to specify which videos you want to infer for.
\
The feature extraction module is also not ready yet.

### Results

The detections of the test frames are saved in the default submission format for the Object Detection Challenge:

```
|---<path to output>
|   |---seen.json
|   |---unseen.json
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. \
\
Please note, Detectron2 is licensed under Apache2.0. Please respect the original licenses as well.

## Acknowledgments

1.  ```
    @INPROCEEDINGS{Damen2018EPICKITCHENS,
    title={Scaling Egocentric Vision: The EPIC-KITCHENS Dataset},
    author={Damen, Dima and Doughty, Hazel and Farinella, Giovanni Maria  and Fidler, Sanja and 
            Furnari, Antonino and Kazakos, Evangelos and Moltisanti, Davide and Munro, Jonathan 
            and Perrett, Toby and Price, Will and Wray, Michael},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2018}
    } 
    ```
2. [Detectron2](https://github.com/facebookresearch/detectron2)