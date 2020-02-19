import os
import numpy as np
import cv2
import argparse
import random
import pickle
import pandas as pd
from ast import literal_eval
from joblib import Parallel, delayed

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer


def create_metadata(idx, row, root_dir):
    """
    Helper function to create dictionary of metadata for Object Annotations of Epic Kitchens dataset
    """
    record = {}
    filename = os.path.join(
        row.participant_id, row.video_id, "{:010d}.jpg".format(row.frame)
    )
    filepath = os.path.join(root_dir, filename)
    if os.path.exists(filepath):
        height, width = cv2.imread(filepath).shape[:2]
        record["height"] = height
        record["width"] = width
    else:
        record["height"] = 0
        record["width"] = 0
    record["file_name"] = filepath
    record["image_id"] = idx

    annotations = []
    if "bounding_boxes" in row:
        for bbox in row.bounding_boxes:
            obj = {
                "bbox": [bbox[1], bbox[0], bbox[3], bbox[2]],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": row.noun_class,
                "iscrowd": 0,
                "image_id": idx,
            }
            annotations.append(obj)
    else:
        obj = {
            "bbox": [-1, -1, -1, -1],
            "bbox_mode": BoxMode.XYWH_ABS,
            "category_id": -1,
            "iscrowd": 0,
            "image_id": idx,
        }
        annotations.append(obj)
    record["annotations"] = annotations
    return record


def get_epic_dicts(root_dir, annotation_file, dict_file=None):
    """
    Helper function to get the list of metadata dictionaries for Object Annotations of Epic Kitchens dataset
    """
    if dict_file:
        print(f"Metadata file found. Loading metadata from {dict_file}")
        with open(dict_file, "rb") as f:
            dataset_dicts = pickle.load(f)
    else:
        print(f"Creating metadata...")
        if "test" in annotation_file:
            annotations = pd.read_csv(annotation_file).sort_values(["video_id"])
            p_id_list = []
            vid_id_list = []
            frames_list = []
            for _, row in annotations.iterrows():
                frames = os.listdir(
                    os.path.join(root_dir, row.participant_id, row.video_id)
                )
                frames = sorted([int(x.split(".")[0]) for x in frames])
                p_id_list.extend([row.participant_id] * len(frames))
                vid_id_list.extend([row.video_id] * len(frames))
                frames_list.extend(frames)
            new_ann_dict = {
                "video_id": vid_id_list,
                "participant_id": p_id_list,
                "frame": frames_list,
            }
            new_annotations = pd.DataFrame(
                new_ann_dict, columns=["video_id", "participant_id", "frame"]
            )
            dataset_dicts = Parallel(n_jobs=16)(
                delayed(create_metadata)(idx, row, root_dir)
                for idx, row in new_annotations.iterrows()
            )
        else:
            annotations = pd.read_csv(annotation_file).sort_values(
                ["video_id", "frame"]
            )
            annotations.bounding_boxes = annotations.bounding_boxes.apply(literal_eval)
            dataset_dicts = Parallel(n_jobs=16)(
                delayed(create_metadata)(idx, row, root_dir)
                for idx, row in annotations.iterrows()
            )

        annotation_path = os.path.split(annotation_file)[0]
        metadata_file = (
            os.path.split(annotation_file)[1].split(".")[0] + "_metadata.pkl"
        )
        metadata_file = os.path.join(annotation_path, metadata_file)
        with open(metadata_file, "wb") as f:
            pickle.dump(dataset_dicts, f)
        print(f"Metadata saved to {metadata_file}")
        print("----------------------------------------------------------")
    return dataset_dicts


def register_dataset(root_dir, ann_dir, dataset_name, read_cache=False):
    """
    Helper function to register catalogue for Epic Kitchens Dataset in Detectron2 library
    """
    noun_classes = pd.read_csv(
        os.path.join(ann_dir, "EPIC_noun_classes.csv")
    ).sort_values("noun_id")
    noun_classes = noun_classes.class_key.to_list()

    if dataset_name == "epic_kitchens_train":
        img_root = os.path.join(root_dir, "train")
        ann_file = os.path.join(ann_dir, "EPIC_train_object_labels.csv")
    elif dataset_name == "epic_kitchens_test_s1":
        img_root = os.path.join(root_dir, "test")
        ann_file = os.path.join(ann_dir, "EPIC_test_s1_object_video_list.csv")
    elif dataset_name == "epic_kitchens_test_s2":
        img_root = os.path.join(root_dir, "test")
        ann_file = os.path.join(ann_dir, "EPIC_test_s2_object_video_list.csv")
    metadata_file = ann_file.split(".")[0] + "_metadata.pkl"
    if read_cache and os.path.exists(metadata_file):
        DatasetCatalog.register(
            dataset_name,
            lambda d=dataset_name: get_epic_dicts(
                img_root, ann_file, dict_file=metadata_file
            ),
        )
    else:
        DatasetCatalog.register(
            dataset_name,
            lambda d=dataset_name: get_epic_dicts(img_root, ann_file, dict_file=None),
        )

    MetadataCatalog.get(dataset_name).set(thing_classes=noun_classes)
    # TODO Create evaluator for epic kitchens
    # MetadataCatalog.get("epic_kitchens_" + d).set(evaluator_type="coco")
    print(f"Dataset {dataset_name} registered.")


def visualize(root_dir, ann_dir):
    """
    Helper function to visualize samples from registered Epic Kitchens dataset in Detectron2 library
    """
    epic_kitchens_metadata = MetadataCatalog.get("epic_kitchens_train")

    print("Dictionary creation in progress...")
    dataset_dicts = get_epic_dicts(
        os.path.join(root_dir, "train"),
        os.path.join(ann_dir, "EPIC_train_object_labels.csv"),
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
