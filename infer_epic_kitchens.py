#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on frames
"""

import logging
import os
import torch
import cv2
import json
import pandas as pd
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
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer

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


def do_infer(cfg, args):
    predictor = DefaultPredictor(cfg)
    file_list = [
        "EPIC_test_s1_object_video_list.csv",
        "EPIC_test_s2_object_video_list.csv",
    ]
    # TODO Implement batch processing
    for ann_file in file_list:
        logger.info("Reading video list from {}".format(ann_file))
        df = pd.read_csv(os.path.join(args.ann_dir, ann_file))
        results = OrderedDict()
        results["version"] = "0.1"
        results["challenge"] = "object_detection"
        logger.info("Number of videos: {}".format(df.shape[0]))
        detections = []
        for _, row in df.iterrows():
            p_id = row.participant_id
            vid_id = row.video_id
            logger.info("Processing video {}...".format(vid_id))
            vid_path = os.path.join(args.root_dir, p_id, vid_id)
            for file in tqdm(sorted(os.listdir(vid_path))):
                if file.endswith("jpg"):
                    img = cv2.imread(os.path.join(vid_path, file))
                    outputs = predictor(img)
                    height, width = img.shape[0:2]
                    shift_y = round(1.0 / height, 5)
                    shift_x = round(1.0 / width, 5)
                    bboxes = outputs["instances"].pred_boxes
                    bboxes.scale(1 / width, 1 / height)
                    scores = outputs["instances"].scores
                    pred_classes = outputs["instances"].pred_classes
                    assert (
                        len(bboxes) == len(scores) == len(pred_classes) >= cfg.TEST.DETECTIONS_PER_IMAGE
                    ), f"Number of detected instances do not match or is less than {cfg.TEST.DETECTIONS_PER_IMAGE}"
                    for idx in range(cfg.TEST.DETECTIONS_PER_IMAGE):
                        # if scores[idx] < 1e-4:
                        #     continue
                        det_dict = OrderedDict()
                        det_dict["video_id"] = vid_id
                        det_dict["frame"] = int(file.split(".")[0])
                        det_dict["category_id"] = int(pred_classes[idx])
                        bbox = bboxes[idx].tensor[0].tolist()
                        bbox = [
                            round(bbox[1], 5),  # ymin
                            round(bbox[0], 5),  # xmin
                            round(bbox[3], 5),  # ymax
                            round(bbox[2], 5),  # xmax
                        ]
                        if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
                            print(
                                "Correcting bbox data for video {} and frame {}".format(
                                    vid_id, int(file.split(".")[0])
                                )
                            )
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
                        ), "Box data {} for video {} and frame {}  is invalid".format(
                            bbox, vid_id, int(file.split(".")[0])
                        )
                        det_dict["bbox"] = bbox
                        det_dict["score"] = float(scores[idx])
                        detections.append(det_dict)
            logger.info("Done.")
            logger.info("----------------------------------------------------------")
        results["results"] = detections
        if "s1" in ann_file:
            with open(os.path.join(cfg.OUTPUT_DIR, "seen.json"), "w") as f:
                json.dump(results, f)
        elif "s2" in ann_file:
            with open(os.path.join(cfg.OUTPUT_DIR, "unseen.json"), "w") as f:
                json.dump(results, f)


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    # TODO Implement multi gpu processing
    # distributed = comm.get_world_size() > 1
    # if distributed:
    #     model = DistributedDataParallel(
    #         model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
    #     )

    do_infer(cfg, args)


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
