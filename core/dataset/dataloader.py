import torch
import logging
import operator

from detectron2.data import DatasetMapper
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data import samplers
from detectron2.data.build import (
    get_detection_dataset_dicts,
    trivial_batch_collator,
    get_world_size,
    worker_init_reset_seed,
)
from detectron2.data.common import AspectRatioGroupedDataset

from core.utils import register_dataset


# def build_detection_train_loader(cfg, args, mapper=None):
#     """
#     Influenced from detectron2's own build_detection_train_loader

#     A data loader is created by the following steps:

#     1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
#     2. Start workers to work on the dicts. Each worker will:

#        * Map each metadata dict into another format to be consumed by the model.
#        * Batch them by simply putting dicts into a list.

#     The batched ``list[mapped_dict]`` is what this dataloader will return.

#     Args:
#         cfg (CfgNode): the config
#         mapper (callable): a callable which takes a sample (dict) from dataset and
#             returns the format to be consumed by the model.
#             By default it will be `DatasetMapper(cfg, True)`.

#     Returns:
#         an infinite iterator of training data
#     """
#     num_workers = get_world_size()
#     images_per_batch = cfg.SOLVER.IMS_PER_BATCH
#     assert (
#         images_per_batch % num_workers == 0
#     ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
#         images_per_batch, num_workers
#     )
#     assert (
#         images_per_batch >= num_workers
#     ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
#         images_per_batch, num_workers
#     )
#     images_per_worker = images_per_batch // num_workers

#     dataset_list = []
#     for dataset_name in cfg.DATASETS.TRAIN:
#         register_dataset(args.root, args.ann_dir, dataset_name, read_cache=args.read_meta_cache)
#         dataset_dicts = get_detection_dataset_dicts(
#             dataset_name,
#             filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
#             min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
#             if cfg.MODEL.KEYPOINT_ON
#             else 0,
#             proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
#         )
#         dataset = DatasetFromList(dataset_dicts, copy=False)

#         if mapper is None:
#             mapper = DatasetMapper(cfg, True)
#         dataset = MapDataset(dataset, mapper)
#         dataset_list.append(dataset)

#     sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
#     logger = logging.getLogger(__name__)
#     logger.info("Using training sampler {}".format(sampler_name))
#     if sampler_name == "TrainingSampler":
#         sampler = samplers.TrainingSampler(len(dataset))
#     elif sampler_name == "RepeatFactorTrainingSampler":
#         sampler = samplers.RepeatFactorTrainingSampler(
#             dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
#         )
#     else:
#         raise ValueError("Unknown training sampler: {}".format(sampler_name))

#     if cfg.DATALOADER.ASPECT_RATIO_GROUPING:
#         data_loader = torch.utils.data.DataLoader(
#             dataset,
#             sampler=sampler,
#             num_workers=cfg.DATALOADER.NUM_WORKERS,
#             batch_sampler=None,
#             collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
#             worker_init_fn=worker_init_reset_seed,
#         )  # yield individual mapped dict
#         data_loader = AspectRatioGroupedDataset(data_loader, images_per_worker)
#     else:
#         batch_sampler = torch.utils.data.sampler.BatchSampler(
#             sampler, images_per_worker, drop_last=True
#         )
#         # drop_last so the batch always have the same size
#         data_loader = torch.utils.data.DataLoader(
#             dataset,
#             num_workers=cfg.DATALOADER.NUM_WORKERS,
#             batch_sampler=batch_sampler,
#             collate_fn=trivial_batch_collator,
#             worker_init_fn=worker_init_reset_seed,
#         )

#     return data_loader


def build_detection_test_loader(cfg, dataset_name, mapper=None):
    """
    Influenced from detectron2's own build_detection_test_loader but used to process mini-batch size greater than 1

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[
                list(cfg.DATASETS.TEST).index(dataset_name)
            ]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, cfg.SOLVER.IMS_PER_BATCH, drop_last=False
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader
