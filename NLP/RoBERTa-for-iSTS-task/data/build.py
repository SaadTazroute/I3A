# encoding: utf-8

from torch.utils import data

from .datasets.semeval_dataset import SemevalDatset


def build_dataset(csv):
    datasets = SemevalDatset(csv)
    return datasets


def make_data_loader(cfg, csv, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.BATCH_SIZE
        shuffle = True
    else:
        batch_size = cfg.TEST.BATCH_SIZE
        shuffle = False

    datasets = build_dataset(csv)

    num_workers = cfg.DATALOADER.NUM_WORKERS

    data_loader = data.DataLoader(
        dataset=datasets,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers)
    return data_loader
