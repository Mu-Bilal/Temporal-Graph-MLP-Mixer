from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from torch_geometric_temporal.dataset import METRLADatasetLoader
import ssl
import numpy as np
import torch
import os
from torch.utils.data import Subset
from model.transform import GraphPartitionTransform, PositionalEncodingTransform
from hdtts_dataset_creation import GraphMSO, EngRad, PvUS


class DynamicNodeFeatureDataset(Dataset):
    def __init__(self, x, y):

        # Shape should be ()
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return f"DynamicNodeDataset(x={self.x.shape}, y={self.y.shape})"


class StaticGraphTopologyData(object):
    """
    Static graph topology data fed into METIS subgraph sampler.
    """
    def __init__(self, edge_index, edge_weight, num_nodes):
        self.edge_index = torch.tensor(edge_index)
        self.edge_weight = torch.tensor(edge_weight)
        self.num_nodes = num_nodes

    def __repr__(self):
        return f"StaticGraphTopologyData(edge_index={self.edge_index.shape}, edge_weight={self.edge_weight.shape}, num_nodes={self.num_nodes})"
    
    def __str__(self):
        return self.__repr__()
    
def create_sliding_window_dataset(dataset, window, delay, horizon, stride):
    data = dataset.target.squeeze()
    assert data.ndim == 2  # (n_timesteps, n_nodes)
    steps = data.shape[0]

    x_idx = np.arange(window)[np.newaxis, :] + np.arange(steps-window-delay-horizon, step=stride)[:, np.newaxis]  # (batch_size, window)
    y_idx = np.arange(window+delay+1, window+delay+horizon+1)[np.newaxis, :] + np.arange(steps-window-delay-horizon, step=stride)[:, np.newaxis]  # (batch_size, horizon)
    assert x_idx.shape[0] == y_idx.shape[0]

    x, y = data[x_idx], data[y_idx]
    return x, y

def get_data_raw(cfg, root='/data'):
    if cfg.dataset.name == 'GraphMSO':
        dataset = GraphMSO(root=os.path.join(root, 'GraphMSO'))

        # Dynamic; (n_timesteps, n_nodes, n_features)
        x, y = create_sliding_window_dataset(dataset, cfg.dataset.window, cfg.dataset.delay, cfg.dataset.horizon, cfg.dataset.stride)

        # Static
        adj = dataset.get_connectivity({'include_self': False, 'layout': 'edge_index'})
        edge_index = adj[0]
        edge_weight = adj[1]
        n_nodes = dataset.target.shape[1]

    elif cfg.dataset.name == 'METRLA':
        assert cfg.dataset.delay == 0, "Delay must be 0 for METRLA"
        assert cfg.dataset.stride == 1, "Stride must be 1 for METRLA"

        ssl._create_default_https_context = ssl._create_unverified_context  # Fix for SSL verification error
        loader = METRLADatasetLoader(raw_data_dir=os.path.join(root, 'METRLA'))
        loader._get_edges_and_weights()
        loader._generate_task(cfg.dataset.window, cfg.dataset.horizon)

        # Dynamic
        x = np.stack(loader.features)[:, :, 0, :]
        y = np.stack(loader.targets)

        # Static
        edge_index = loader.edges
        edge_weight = loader.edge_weights
        n_nodes = x.shape[1]

    return x, y, edge_index, edge_weight, n_nodes

def split_dataset(cfg: OmegaConf, dataset):
    assert cfg.dataset.train_size + cfg.dataset.val_size <= 1, "Train size and validation size must sum to no more than 1"
    sizes_rel = np.array([cfg.dataset.train_size, cfg.dataset.val_size, 1-cfg.dataset.train_size-cfg.dataset.val_size])
    length = min(cfg.dataset.max_len, len(dataset)) if cfg.dataset.max_len is not None else len(dataset)
    sizes_abs = sizes_rel * length
    return torch.utils.data.random_split(Subset(dataset, range(length)), sizes_abs)


def create_dataloaders(cfg: OmegaConf, raw_data_dir=os.path.join(os.path.dirname(__file__), '../data')):
    x, y, edge_index, edge_weight, n_nodes = get_data_raw(cfg, raw_data_dir)

    dataset = DynamicNodeFeatureDataset(x, y)
    train_dataset, val_dataset, test_dataset = split_dataset(cfg, dataset)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers, drop_last=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers, drop_last=True, pin_memory=True, persistent_workers=True) \
        if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers, drop_last=True, pin_memory=True, persistent_workers=True) \
        if len(test_dataset) > 0 else None

    topo_data = StaticGraphTopologyData(edge_index, edge_weight, n_nodes)
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
