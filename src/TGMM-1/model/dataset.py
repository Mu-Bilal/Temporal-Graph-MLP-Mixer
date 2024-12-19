import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import ZINC
import torch_geometric_temporal as tgt
import pickle
import os
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import trange
from omegaconf import OmegaConf
from torch_geometric_temporal.dataset import METRLADatasetLoader
from typing import Iterable

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


def create_train_val_test_split(dataset, max_len: int = None):
    if isinstance(dataset, tgt.signal.static_graph_temporal_signal.StaticGraphTemporalSignal):
        total_snapshots = dataset.snapshot_count
    else:
        total_snapshots = len(dataset)

    if max_len:
        total_snapshots = min(total_snapshots, max_len)
    train_size = int(0.7 * total_snapshots)
    val_size = int(0.15 * total_snapshots)
    test_size = total_snapshots - train_size - val_size

    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size+val_size] 
    test_data = dataset[train_size+val_size:total_snapshots]

    return train_data, val_data, test_data


class CustomTemporalData(object):
    """
    Wraps each batch in PyG-Temporal Data object to same fields as PyG Data object (as required by GMM transforms).
    """
    def __init__(self, edge_index, edge_weight, num_nodes, features, targets):
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.edge_attr = edge_weight
        self.num_nodes = num_nodes
        self.features = features
        self.targets = targets

    def __str__(self):
        return (f"CustomTemporalData(\n"
                f"  num_nodes: {self.num_nodes}\n"
                f"  edge_index: {tuple(self.edge_index.shape)}\n" 
                f"  edge_weight: {tuple(self.edge_weight.shape)}\n"
                f"  edge_attr: {tuple(self.edge_attr.shape)}\n"
                f"  features: {tuple(self.features.shape)}\n"
                f"  targets: {tuple(self.targets.shape)}\n"
                f")")
    
    def __repr__(self):
        return self.__str__()


class CustomTemporalDataset(Dataset):
    def __init__(
            self, 
            dataset: tgt.signal.static_graph_temporal_signal.StaticGraphTemporalSignal, 
            graph_transform = None
            ):
        self.data_template = self.wrap_single_data(dataset[0])
        self.graph_transform = graph_transform  # Graph transform (static, only needs to be computed once)
        self.data_template_transformed = self.apply_graph_transform(self.data_template)
        self.dataset = self.transform_via_template(dataset)

    def apply_graph_transform(self, datum):
        if isinstance(self.graph_transform, Iterable):
            for transform in self.graph_transform:
                datum = transform(datum)
        else:
            datum = self.graph_transform(datum)
        return datum

    def wrap_single_data(self, data):
        return CustomTemporalData(data.edge_index, data.edge_weight, data.num_nodes, data.x, data.y)
    
    def transform_via_template(self, dataset: tgt.signal.static_graph_temporal_signal.StaticGraphTemporalSignal):
        assert isinstance(dataset, tgt.signal.static_graph_temporal_signal.StaticGraphTemporalSignal)
        data_list = []

        data_cls = type(self.data_template_transformed)
        data_dict = {k: v for k, v in self.data_template_transformed.items()}

        for i in range(dataset.snapshot_count):
            # Basically copying the template
            datum = data_cls(**data_dict)  

            # Replace data
            datum.features = dataset[i].x[:, 0, :]  # FIXME: This is specific to METRLA
            datum.y = dataset[i].y
            
            data_list.append(datum)

        return data_list

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return data

    def __len__(self):
        return len(self.dataset)
    

def create_dataloaders(cfg: OmegaConf, dataset_name: str, max_len: int = None):
    assert dataset_name == 'METRLA', "Only METRLA dataset is currently supported"

    pre_transform = PositionalEncodingTransform(rw_dim=cfg.pos_enc.rw_dim, lap_dim=cfg.pos_enc.lap_dim)
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


    loader = METRLADatasetLoader()
    dataset_metrola = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
    train_data, val_data, test_data = create_train_val_test_split(dataset_metrola, max_len=max_len)

    train_data = CustomTemporalDataset(train_data, graph_transform=[pre_transform, transform_train])
    val_data = CustomTemporalDataset(val_data, graph_transform=[pre_transform, transform_eval])
    test_data = CustomTemporalDataset(test_data, graph_transform=[pre_transform, transform_eval])

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
