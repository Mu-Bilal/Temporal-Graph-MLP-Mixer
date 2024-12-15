import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import ZINC
import pickle
import os

from model.transform import GraphPartitionTransform, PositionalEncodingTransform


class PlanarSATPairsDataset(InMemoryDataset):
    """
    Used for exp-classify
    """
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PlanarSATPairsDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["GRAPHSAT.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(
            open(os.path.join(self.root, "raw/GRAPHSAT.pkl"), "rb"))

        data_list = [Data(**g.__dict__) for g in data_list]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def create_dataset(cfg):
    pre_transform = PositionalEncodingTransform(rw_dim=cfg.pos_enc.rw_dim, lap_dim=cfg.pos_enc.lap_dim)

    transform_train = transform_eval = None

    if cfg.metis.n_patches > 0:
        transform_train = GraphPartitionTransform(n_patches=cfg.metis.n_patches,
                                                   metis=cfg.metis.enable,
                                                   drop_rate=cfg.metis.drop_rate,
                                                   num_hops=cfg.metis.num_hops,
                                                   is_directed=False,
                                                   patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                                                   patch_num_diff=cfg.pos_enc.patch_num_diff)

        transform_eval = GraphPartitionTransform(n_patches=cfg.metis.n_patches,
                                                  metis=cfg.metis.enable,
                                                  drop_rate=0.0,
                                                  num_hops=cfg.metis.num_hops,
                                                  is_directed=False,
                                                  patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                                                  patch_num_diff=cfg.pos_enc.patch_num_diff)

    if cfg.dataset == 'ZINC':
        root = 'dataset/ZINC'
        train_dataset = ZINC(root, subset=True, split='train', pre_transform=pre_transform, transform=transform_train)
        val_dataset = ZINC(root, subset=True, split='val', pre_transform=pre_transform, transform=transform_eval)
        test_dataset = ZINC(root, subset=True, split='test', pre_transform=pre_transform, transform=transform_eval)
    elif cfg.dataset == 'exp-classify':
        raise NotImplementedError("returns wrong type")
        root = "dataset/EXP/"
        dataset = PlanarSATPairsDataset(root, pre_transform=pre_transform)
        return dataset, transform_train, transform_eval
    else:
        print("Dataset not supported.")
        exit(1)

    # torch.set_num_threads(cfg.num_workers)
    # if not cfg.metis.online:
    #     train_dataset = [x for x in train_dataset]
    # val_dataset = [x for x in val_dataset]
    # test_dataset = [x for x in test_dataset]

    return train_dataset, val_dataset, test_dataset