"""
Perform inference on object frames of the Epic Kitchens Dataset
"""

import logging
import os
import torch
import json
from tqdm import tqdm
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch,
    DefaultPredictor,
)
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from core.dataset import build_detection_test_loader
from core.utils import register_dataset

logger = logging.getLogger("detectron2")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def format_detections(outputs, inputs):
    detections = []
    for output, input in zip(outputs, inputs):
        vid_id = input["file_name"].split("/")[-2]
        frame = int(input["file_name"].split("/")[-1].split(".")[0])
        height, width = input["height"], input["width"]
        shift_y = round(1.0 / height, 5)
        shift_x = round(1.0 / width, 5)
        bboxes = output["instances"].pred_boxes
        bboxes.scale(1 / width, 1 / height)
        scores = output["instances"].scores
        pred_classes = output["instances"].pred_classes
        assert (
            len(bboxes) == len(scores) == len(pred_classes)
        ), f"Number of detected instances do not match"

        for idx in range(len(bboxes)):
            # if scores[idx] < 1e-4:
            #     continue
            det_dict = OrderedDict()
            det_dict["video_id"] = vid_id
            det_dict["frame"] = frame
            det_dict["category_id"] = int(pred_classes[idx])
            bbox = bboxes[idx].tensor[0].tolist()
            bbox = [
                round(bbox[1], 5),  # ymin
                round(bbox[0], 5),  # xmin
                round(bbox[3], 5),  # ymax
                round(bbox[2], 5),  # xmax
            ]
            if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
                print(f"Correcting bbox data for video {vid_id} and frame {frame}")
                if bbox[0] == bbox[2]:
                    if bbox[0] - shift_y >= 0:
                        bbox[0] -= shift_y
                    else:
                        bbox[2] += shift_y
                if bbox[1] == bbox[3]:
                    if bbox[1] - shift_x >= 0:
                        bbox[1] -= shift_x
                    else:
                        bbox[3] += shift_x
            assert (
                bbox[0] < bbox[2] and bbox[1] < bbox[3]
            ), f"Box data {bbox} for video {vid_id} and frame {frame}  is invalid"
            det_dict["bbox"] = bbox
            det_dict["score"] = float(scores[idx])
            detections.append(det_dict)
    return detections


def do_infer(cfg, args, model):

    if args.read_meta_cache:
        read_cache = True
    else:
        read_cache = False
    model.eval()
    for dataset_name in cfg.DATASETS.TEST:
        register_dataset(
            args.root_dir, args.ann_dir, dataset_name, read_cache=read_cache
        )
        results = OrderedDict()
        results["version"] = "0.1"
        results["challenge"] = "object_detection"
        data_loader = build_detection_test_loader(cfg, dataset_name)
        logger.info("Number of frames: {}".format(len(data_loader.dataset)))

        detections = []
        with torch.no_grad():
            for inputs in tqdm(data_loader):
                outputs = model(inputs)
                detections.extend(format_detections(outputs, inputs))

        results["results"] = detections
        if dataset_name == "epic_kitchens_test_s1":
            json_file = os.path.join(cfg.OUTPUT_DIR, "seen.json")
        elif dataset_name == "epic_kitchens_test_s2":
            json_file = os.path.join(cfg.OUTPUT_DIR, "unseen.json")
        else:
            json_file = os.path.join(cfg.OUTPUT_DIR, f"results_{dataset_name}.json")
        with open(json_file, "w") as f:
            json.dump(results, f)
        logger.info(f"Results for {dataset_name} saved to {json_file}")


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_infer(cfg, args, model)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--root-dir",
        dest="root_dir",
        required=True,
        default="",
        help="path to image files",
    )
    parser.add_argument(
        "--ann-dir",
        dest="ann_dir",
        required=True,
        default="",
        help="path to image files",
    )
    parser.add_argument(
        "--read-meta-cache", action="store_true", help="Read metadata from cache file"
    )
    args = parser.parse_args()
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise Exception(
            "No GPU found. The model is not implemented without GPU support."
        )
    print("Command Line Args:", args)
    launch(
        main,
        num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
