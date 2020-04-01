"""
Perform inference on object frames of the Epic Kitchens Dataset
"""

import logging
import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import ffmpeg

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

        det = np.zeros((6))
        # det[0] = frame_no
        det[0] = pred_classes[idx]
        det[1] = round(scores[idx], 5)
        det[2:] = np.array(bbox)
        detections.append(det)

    return np.stack(detections)


def do_infer(cfg, args):

    with open(args.vid_anns, "rb") as f:
        inference_data = pickle.load(f)

    predictor = FramePredictor(cfg, "RGB")

    logger.info(f"{len(inference_data.keys())} videos to be processed.")
    logger.info(
        "-------------------------------------------------------------------------"
    )
    for vid_id in inference_data.keys():
        time_frames = inference_data[vid_id]
        vid_file = os.path.join(args.vid_dir, f"{vid_id}.MP4")
        probe = ffmpeg.probe(vid_file)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        fps = video_stream["avg_frame_rate"].split("/")
        fps = round(float(fps[0]) / float(fps[1]), 2)
        inference_out_dir = os.path.join(cfg.OUTPUT_DIR, vid_id)
        os.makedirs(inference_out_dir, exist_ok=True)
        logger.info(f"Processing {vid_id}. Output will be saved to {inference_out_dir}")
        for offset in tqdm(time_frames):
            offset = float(offset)
            if offset > 0:
                frame_no = int(fps * offset)
                buffer, _ = (
                    ffmpeg.input(vid_file)
                    .filter("select", "gte(n,{})".format(frame_no))
                    .output(
                        "pipe:",
                        format="rawvideo",
                        pix_fmt="rgb24",
                        vframes=1,
                        loglevel="quiet",
                    )
                    .run(capture_stdout=True)
                )
                frame = np.frombuffer(buffer, np.uint8).reshape([height, width, 3])
                out = predictor(frame)
                detections = format_detections(frame_no, height, width, out)
                out_file = os.path.join(inference_out_dir, f"offset_{offset}.npy")
                np.save(out_file, detections)
        logger.info("Done")
        logger.info(
            "-------------------------------------------------------------------------"
        )


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
