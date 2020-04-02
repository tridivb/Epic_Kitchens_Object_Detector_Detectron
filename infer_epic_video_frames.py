"""
Perform inference on object frames of the Epic Kitchens Dataset
"""

import logging
import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import moviepy.editor as mpe
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch,
)

from core.utils import FramePredictor

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


def format_detections(frame_no, height, width, output):
    detections = []
    bboxes = output["instances"].pred_boxes
    bboxes.scale(1 / width, 1 / height)
    scores = output["instances"].scores.tolist()
    pred_classes = output["instances"].pred_classes.tolist()

    for idx in range(len(bboxes)):
        if scores[idx] < 5e-2:
            continue
        bbox = bboxes[idx].tensor[0].tolist()
        bbox = [
            round(bbox[1], 5),  # ymin
            round(bbox[0], 5),  # xmin
            round(bbox[3], 5),  # ymax
            round(bbox[2], 5),  # xmax
        ]
        if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
            continue

        # format: [class, score, ymin, xmin, ymax, xmax]
        det = np.zeros((6))
        # det[0] = frame_no
        det[0] = pred_classes[idx]
        det[1] = round(scores[idx], 5)
        det[2:] = np.array(bbox)
        detections.append(det)

    return detections


def do_infer(cfg, args):

    with open(args.vid_anns, "rb") as f:
        inference_data = pickle.load(f)

    predictor = FramePredictor(cfg, "RGB")

    logger.info(f"{len(inference_data.keys())} videos to be processed.")
    logger.info(
        "-------------------------------------------------------------------------"
    )
    rejected_dict = OrderedDict()
    for vid_id in inference_data.keys():
        time_frames = inference_data[vid_id]
        vid_file = os.path.join(args.vid_dir, f"{vid_id}.MP4")
        inference_out_dir = os.path.join(cfg.OUTPUT_DIR, vid_id)
        os.makedirs(inference_out_dir, exist_ok=True)
        logger.info(
            f"Processing {vid_id}. Output will be saved to {inference_out_dir}/"
        )
        video = mpe.VideoFileClip(vid_file, audio=False, fps_source="tbr")
        start_frame = video.get_frame(0)
        height, width = start_frame.shape[0:2]
        rejected = []
        for offset in tqdm(time_frames):
            offset = float(offset)
            frame_no = int(video.fps * offset)
            if offset > 0:
                frame = video.get_frame(offset)
                out = predictor(frame)
                detections = format_detections(frame_no, height, width, out)
                if len(detections) > 0:
                    detections = np.stack(detections)
                    out_file = os.path.join(inference_out_dir, f"offset_{offset}.npy")
                    np.save(out_file, detections)
                else:
                    rejected.append(offset)
                rejected.append(0.00)
                break
        video.close()
        if len(rejected) > 0:
            rejected_dict[vid_id] = rejected
            logger.warning(f"Rejected offsets:{rejected}")
        logger.info("Done")
        logger.info(
            "-------------------------------------------------------------------------"
        )
        break

    if len(rejected_dict.keys()) > 0:
        rejected_file = os.path.split(args.vid_anns)[1]
        rejected_file = os.path.join(cfg.OUTPUT_DIR, f"rejected_{rejected_file}")
        with open(rejected_file, "wb") as f:
            pickle.dump(rejected_dict, f)
        logger.info(f"List of rejected frames saved to {rejected_file}")


def main(args):
    cfg = setup(args)

    do_infer(cfg, args)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--vid-dir",
        dest="vid_dir",
        required=True,
        default="",
        help="Directory location of all videos",
    )
    parser.add_argument(
        "--vid-anns",
        dest="vid_anns",
        required=True,
        default="",
        help="File with video frame details for inference",
    )
    args = parser.parse_args()
    # Force it to run on single gpu
    num_gpus = 1
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
