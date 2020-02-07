# Epic Kitchens Object Feature Extractor with Detectron

WIP for Detectron2 and Python3. For the previous python2 working version, refer to the [python2](https://github.com/tridivb/Epic_Kitchens_Feature_Extractor_Detectron/tree/python2) branch.

Detect Objects in the Epic Kitchens dataset using Faster-RCNN as the backbone and the Detectron library.

Once the updates are up, the python2 version will be deprecated.

## TODO
1. Implement batch and multi-gpu processing in inference script
2. Create evaluator for Epic Kitchens
3. Implement tester/inference script with detectron2 and python3
4. Provide conda environment file or requirements file to setup python environment. -- Done
5. Modify docker scripts to accomodate updated versions of python and detectron (Currently not working)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

1. [Detectron2](https://github.com/facebookresearch/detectron2)