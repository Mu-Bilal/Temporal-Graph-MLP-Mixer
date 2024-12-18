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

from model.transform import GraphPartitionTransform, PositionalEncodingTransform


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
            transform = None,
            pre_transform = None,
            max_len = None
            ):
        self.data = self.wrap_temporal_data(dataset)
        if max_len:
            self.data = self.data[:max_len]
        self.pre_transform = pre_transform
        self.transform = transform
        if self.pre_transform:
            self.data = [self.pre_transform(self.data[i]) for i in trange(len(self.data), desc="Pre-transforming data")]

    def wrap_temporal_data(self, dataset: tgt.signal.static_graph_temporal_signal.StaticGraphTemporalSignal):
        assert isinstance(dataset, tgt.signal.static_graph_temporal_signal.StaticGraphTemporalSignal)

        # Static graph data
        edge_index = torch.tensor(dataset.edge_index, dtype=torch.long)
        edge_attr = torch.tensor(dataset.edge_weight, dtype=torch.float32)
        num_nodes = dataset.features[0].shape[0]

        # Dynamic temporal data
        data_list = []
        for i in range(dataset.snapshot_count):
            x = torch.tensor(dataset.features[i][:, 0, :], dtype=torch.float32)  # Node features
            y = torch.tensor(dataset.targets[i], dtype=torch.float32)  # Target
            
            data_list.append(CustomTemporalData(edge_index, edge_attr, num_nodes, x, y))

        return data_list

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform:
            if isinstance(data, list):
                data = [self.transform(data[i]) for i in trange(len(data), desc="Transforming data")]
            else:
                data = self.transform(data)
        return data

    def __len__(self):
        return len(self.data)

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

    train_data = CustomTemporalDataset(train_data, transform=transform_train, pre_transform=pre_transform)
    val_data = CustomTemporalDataset(val_data, transform=transform_eval, pre_transform=pre_transform)
    test_data = CustomTemporalDataset(test_data, transform=transform_eval, pre_transform=pre_transform)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
