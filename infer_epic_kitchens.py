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

"""Perform inference on all the frames of a video
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm
from parse import parse

from caffe2.python import core
from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

from utils.settings import config

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end inference")
    parser.add_argument(
        "--cfg",
        dest="cfg",
        help="cfg model file (/path/to/model_config.yaml)",
        default=config.cfg_file,
        type=str,
    )
    parser.add_argument(
        "--wts",
        dest="weights",
        help="weights model file (/path/to/model_weights.pkl)",
        default=config.weights,
        type=str,
    )
    parser.add_argument(
        "--top_predictions",
        dest="top_predictions",
        help="Number of predictions to store",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--video_root",
        dest="video_root",
        help="path_to_video_root",
        default=config.video_root,
        type=str,
    )
    parser.add_argument(
        "--video_list",
        dest="video_list",
        help="path_to_list_of_videos",
        default=config.video_list,
        type=str,
    )
    parser.add_argument(
        "--sample_fps",
        dest="sample_fps",
        help="fps_value_to_sample_videos",
        default=config.sample_fps,
        type=int,
    )
    parser.add_argument(
        "--out_path",
        dest="out_path",
        help="path_to_save_detections",
        default=config.out_path,
        type=str,
    )

    return parser.parse_args()


def format_dets(boxes):
    """
        Helper function to format the detections as N*1030 where,
        N = no of detections in a frame
        Column 1 = object index
        Columns 2-5 = bounding box coordinates
        Column 6 = Probability score
        Columns 7-1030 = Extracted Feature from Faster-RCNN right before the final classification layer
        Args:
            boxes - Extracted bounding box coordinates, probability score and object features for a frame
    """
    all_boxes = []
    for index, box in enumerate(boxes):
        if len(box) > 0:
            box = np.array(box)
            item_index = np.ones((len(box), 1)) * index - 1
            box = np.hstack([item_index, box])
            all_boxes.append(box)
    if len(all_boxes) > 0:
        all_boxes = np.concatenate(all_boxes)
    else:
        all_boxes = np.zeros((0, 1030))
    return all_boxes


def main(args):
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, "RPN models are not supported"
    assert (
        not cfg.TEST.PRECOMPUTED_PROPOSALS
    ), "Models that require precomputed proposals are not supported"

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    videos_root = args.video_root
    videos_list_file = args.video_list

    out_path = args.out_path

    print("Loading Video List ...")
    with open(videos_list_file) as f:
        videos = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    print("Done")
    print("----------------------------------------------------------")

    videos = [x for x in videos if "train" in x and os.path.split(x)[1] > "P03_27"]
    in_fps = args.sample_fps

    print("Total no of videos to be processed: {}".format(len(videos)))
    for v in videos:

        vid_path = os.path.join(videos_root, v + ".MP4")
        print("Processing {} at {} fps...".format(os.path.split(v)[1], in_fps))

        detections_path = os.path.join(out_path, os.path.split(v)[0])
        detections_file = os.path.join(
            detections_path, os.path.split(v)[1] + "_detections.npy"
        )

        if os.path.isfile(detections_file):
            logger.info(
                "{} already processed. The previous detections will be overwritten".format(
                    vid_path
                )
            )
            # continue

        if not os.path.exists(detections_path):
            os.makedirs(detections_path)

        vid = VideoFileClip(vid_path, audio=False, fps_source="fps")

        all_detections = []

        t = time.time()
        for _, in_frame in tqdm(vid.iter_frames(fps=in_fps, with_times=True)):
            timers = defaultdict(Timer)

            with c2_utils.NamedCudaScope(0):
                cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                    model, in_frame, None, timers=timers
                )
            all_detections.append(format_dets(cls_boxes))

        logger.info("Inference time: {:.3f}s".format(time.time() - t))
        np.save(detections_file, all_detections)
        print("Done")
        print("----------------------------------------------------------")


if __name__ == "__main__":
    workspace.GlobalInit(["caffe2", "--caffe2_log_level=0"])
    setup_logging(__name__)
    args = parse_args()
    main(args)
