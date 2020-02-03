import os
import numpy as np
import json
import cv2
import argparse
import random
import pandas as pd
from ast import literal_eval

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer


def get_epic_dicts(root_dir, annotation_file, d):
    if d == "train":
        annotations = pd.read_csv(annotation_file).sort_values(["video_id", "frame"])[0:200]
    elif d == "val":
        annotations = pd.read_csv(annotation_file).sort_values(["video_id", "frame"])[201:250]

    dataset_dicts = []
    annotations.bounding_boxes = annotations.bounding_boxes.apply(literal_eval)
    for idx, row in annotations.iterrows():
        record = {}
        if len(row.bounding_boxes) == 0:
            continue

        filename = os.path.join(
            row.participant_id, row.video_id, "{:010d}.jpg".format(row.frame)
        )
        filepath = os.path.join(root_dir, filename)
        try:
            height, width = cv2.imread(filepath).shape[:2]
        except Exception as e:
            print(idx, row)
            raise Exception("Problem loading file {} with error {}".format(filepath, e))

        record["file_name"] = filepath
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annotations = []
        for bbox in row.bounding_boxes:
            obj = {
                "bbox": [bbox[1], bbox[0], bbox[3], bbox[2]],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": row.noun_class,
                "iscrowd": 0,
                "image_id": idx,
            }
            annotations.append(obj)
        record["annotations"] = annotations
        dataset_dicts.append(record)
    return dataset_dicts


def register_dataset(root_dir, ann_dir):
    
    noun_classes = pd.read_csv(
        os.path.join(ann_dir, "EPIC_noun_classes.csv")
    ).sort_values("noun_id")
    noun_classes = noun_classes.class_key.to_list()

    print("Metadata being created...")
    for d in ["train", "val"]:
        img_root = os.path.join(root_dir, "train")
        # if d == "train":
        ann_file = os.path.join(ann_dir, "EPIC_train_object_labels.csv")
        DatasetCatalog.register(
            "epic_kitchens_" + d, lambda d=d: get_epic_dicts(img_root, ann_file, d)
        )
        MetadataCatalog.get("epic_kitchens_" + d).set(thing_classes=noun_classes)
    print("Done")
    print("----------------------------------------------------------")
    

def visualize(root_dir, ann_dir):
    epic_kitchens_metadata = MetadataCatalog.get("epic_kitchens_train")

    print("Dictionary creation in progress...")
    dataset_dicts = get_epic_dicts(
        os.path.join(root_dir, "train"),
        os.path.join(ann_dir, "EPIC_train_object_labels.csv"),
        "val"
    )
    print("Done")
    print("----------------------------------------------------------")

    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(
            img[:, :, ::-1], metadata=epic_kitchens_metadata, scale=0.5
        )
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow("epic_img", vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
