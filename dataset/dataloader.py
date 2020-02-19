import torch

from detectron2.data import DatasetMapper
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data import samplers
from detectron2.data.build import get_detection_dataset_dicts, trivial_batch_collator


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
