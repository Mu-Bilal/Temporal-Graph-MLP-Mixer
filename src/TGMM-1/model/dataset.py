from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from torch_geometric_temporal.dataset import METRLADatasetLoader
import ssl
import numpy as np
import torch
import os
from torch.utils.data import Subset
from model.transform import GraphPartitionTransform, PositionalEncodingTransform


class DynamicNodeDataset(Dataset):
    def __init__(self, x, y):

        # Shape should be ()
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class StaticGraphTopologyData(object):
    """
    Static graph topology data fed into METIS subgraph sampler.
    """
    def __init__(self, edge_index, edge_weight, num_nodes):
        self.edge_index = torch.tensor(edge_index)
        self.edge_weight = torch.tensor(edge_weight)
        self.num_nodes = num_nodes


def split_dataset(cfg: OmegaConf, dataset):
    assert cfg.dataset.train_size + cfg.dataset.val_size <= 1, "Train size and validation size must sum to no more than 1"
    sizes_rel = np.array([cfg.dataset.train_size, cfg.dataset.val_size, 1-cfg.dataset.train_size-cfg.dataset.val_size])
    length = min(cfg.dataset.max_len, len(dataset)) if cfg.dataset.max_len is not None else len(dataset)
    sizes_abs = sizes_rel * length
    return torch.utils.data.random_split(Subset(dataset, range(length)), sizes_abs)

def create_dataloaders(cfg: OmegaConf):
    assert cfg.dataset.name == 'METRLA', "Only METRLA dataset is supported currently"

    # Load data
    ssl._create_default_https_context = ssl._create_unverified_context  # Fix for SSL verification error
    loader = METRLADatasetLoader(raw_data_dir=os.path.join(os.getcwd(), 'data', 'METRLA'))
    loader._get_edges_and_weights()
    loader._generate_task(cfg.dataset.window, cfg.dataset.horizon)

    # Dynamic
    x = np.stack(loader.features)[:, :, 0, :]
    y = np.stack(loader.targets)

    dataset = DynamicNodeDataset(x, y)
    train_dataset, val_dataset, test_dataset = split_dataset(cfg, dataset)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers, drop_last=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers, drop_last=True, pin_memory=True, persistent_workers=True) \
        if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers, drop_last=True, pin_memory=True, persistent_workers=True) \
        if len(test_dataset) > 0 else None

    # Static
    edges = loader.edges
    edge_weights = loader.edge_weights
    num_nodes = x.shape[1]

    topo_data = StaticGraphTopologyData(edges, edge_weights, num_nodes)
    # pre_transform = PositionalEncodingTransform(rw_dim=cfg.pos_enc.rw_dim, lap_dim=cfg.pos_enc.lap_dim)
    if cfg.metis.n_patches > 0:
        transform_train = GraphPartitionTransform(n_patches=cfg.metis.n_patches,
                                                    metis=cfg.metis.enable,
                                                    drop_rate=0,
                                                    num_hops=cfg.metis.num_hops,
                                                    is_directed=False,
                                                    patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                                                    patch_num_diff=cfg.pos_enc.patch_num_diff)
        
        # topo_data = pre_transform(topo_data)
        topo_data = transform_train(topo_data)

    return train_loader, val_loader, test_loader, topo_data
