from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
import numpy as np
import torch
import os
from torch.utils.data import Subset
from model.transform import GraphPartitionTransform
from hdtts_dataset_creation import EngRad, PvUS
from hdtts_dataset_creation.dataset_util import load_dataset, load_synthetic_dataset
from tsl.ops.connectivity import adj_to_edge_index
from einops import rearrange

class DynamicNodeFeatureDataset(Dataset):
    def __init__(self, x, y, mask_x, mask_y):
        """
        mask (pandas.Dataframe or numpy.ndarray, optional): Boolean mask
        denoting if values in data are valid (:obj:`True`) or not
        (:obj:`False`).
        """
        assert len(x) == len(y) == len(mask_x) == len(mask_y), "x, y, mask_x, and mask_y must have the same length"
        assert len(x) > 0, "x, y, mask_x, and mask_y must have at least one sample"

        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
        self.mask_x = torch.tensor(mask_x, dtype=bool)
        self.mask_y = torch.tensor(mask_y, dtype=bool)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask_x[idx], self.mask_y[idx]
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return f"DynamicNodeDataset(x={self.x.shape}, y={self.y.shape}, mask_x={self.mask_x.shape}, mask_y={self.mask_y.shape})"

class StaticGraphTopologyData(object):
    """
    Static graph topology data fed into METIS subgraph sampler.
    """
    def __init__(self, edge_index, edge_weight, num_nodes):
        self.edge_index = torch.tensor(edge_index)
        self.edge_weight = torch.tensor(edge_weight)
        self.num_nodes = num_nodes

        assert self.edge_weight.shape[0] == self.edge_index.shape[1], "Edge weight must be a 1D tensor with the same length as the number of edges"
        assert self.edge_weight.ndim == 1, "Edge weight must be a 1D tensor"

    def __repr__(self):
        return f"StaticGraphTopologyData(edge_index={self.edge_index.shape}, edge_weight={self.edge_weight.shape}, num_nodes={self.num_nodes})"
    
    def __str__(self):
        return self.__repr__()
    
def create_sliding_window_dataset(data, window, delay, horizon, stride, max_steps=None, max_allowed_mem=8):
    steps = data.shape[0] - window - delay - horizon
    n_nodes = data.shape[1]

    if max_steps is not None:
        steps = min(steps, max_steps)

    # Calc memory needed
    memory_needed = (window + horizon) * steps * n_nodes * 8 * 1e-9  # GB
    print(f"Predicted raw dataset size: {memory_needed:.2f} GB")
    assert memory_needed <= max_allowed_mem, f"Memory needed ({memory_needed:.2f} GB) exceeds max allowed memory ({max_allowed_mem} GB)"

    x_idx = np.arange(window)[np.newaxis, :] + np.arange(steps, step=stride)[:, np.newaxis]  # (batch_size, window, n_features)
    y_idx = np.arange(window+delay+1, window+delay+horizon+1)[np.newaxis, :] + np.arange(steps, step=stride)[:, np.newaxis]  # (batch_size, horizon, n_features)
    assert x_idx.shape[0] == y_idx.shape[0]

    if data.ndim == 3:
        x, y = rearrange(data[x_idx], 'B w n f -> B n w f'), rearrange(data[y_idx], 'B h n f -> B n h f')
    else:
        x, y = rearrange(data[x_idx], 'B w n -> B n w'), rearrange(data[y_idx], 'B h n -> B n h')
    return x, y

def get_data_raw(cfg: OmegaConf, root='/data'):
    """
    cfg: Top-level config. Original HDTTS config is in cfg.raw_data.
    """

    if cfg.raw_data.name.startswith('mso'):
        dataset, adj, mask_original, mask = load_synthetic_dataset(cfg.raw_data, root_dir=root)
        mask = mask.squeeze()

        # Get data and set missing values to nan
        data = dataset.dataframe()
        masked_data = data.where(mask_original.reshape(mask_original.shape[0], -1), np.nan)
        # Fill nan with Last Observation Carried Forward
        data = masked_data.ffill().bfill()
    else:
        dataset, adj, mask_original = load_dataset(cfg.raw_data, root_dir=root)
        mask = dataset.mask.squeeze()

        # Get data and set missing values to nan
        data = dataset.dataframe()
        masked_data = data.where(mask_original.reshape(mask_original.shape[0], -1), np.nan)

        if isinstance(dataset, PvUS):
            # Fill nan with Last -24h Observation Carried Forward
            data = masked_data.groupby([data.index.hour, data.index.minute]).ffill()
            data = data.groupby([data.index.hour, data.index.minute]).bfill()
        else:
            # Fill nan with Last Observation Carried Forward
            data = masked_data.ffill().bfill()
        # Fill remaining nan with 0, if any
        data.fillna(0, inplace=True)


    if isinstance(dataset, EngRad):  # FIXME: Quickfix as model is currently univariate
        data = data.iloc[:, cfg.dataset.EngRADchannel::5]

    data = data.to_numpy()

    if cfg.dataset.normalize:
        # Calculate statistics on the entire dataset
        norm_mean, norm_std = data.mean(), data.std()
        data = (data - norm_mean) / norm_std
        metadata = {
            'norm_mean': norm_mean,
            'norm_std': norm_std
        }
    else:
        metadata = {}

    # FIXME: Make this dynamic for multivariate data (currently assuming data has single feature)

    # (n_timesteps, n_nodes, n_features)
    x, y = create_sliding_window_dataset(data, cfg.dataset.window, cfg.dataset.delay, cfg.dataset.horizon, cfg.dataset.stride, max_steps=cfg.dataset.max_len)
    mask_x, mask_y = create_sliding_window_dataset(mask, cfg.dataset.window, cfg.dataset.delay, cfg.dataset.horizon, cfg.dataset.stride, max_steps=cfg.dataset.max_len)

    n_nodes = dataset.n_nodes
    edge_index, edge_weight = adj_to_edge_index(adj.todense())
    edge_weight = np.asarray(edge_weight).squeeze()
    edge_index = np.asarray(edge_index)

    return x, y, mask_x, mask_y, edge_index, edge_weight, n_nodes, metadata

def split_dataset(cfg: OmegaConf, dataset):
    """
    These are temporal datasets with overlap in windows and horizons. We need to create mutually disjoint subsets.
    """
    assert cfg.dataset.train_size + cfg.dataset.val_size <= 1, "Train size and validation size must sum to no more than 1"
    sizes_rel = np.array([cfg.dataset.train_size, cfg.dataset.val_size, 1-cfg.dataset.train_size-cfg.dataset.val_size])
    length = min(cfg.dataset.max_len, len(dataset)) if cfg.dataset.max_len is not None else len(dataset)
    sizes_abs = (sizes_rel * length).astype(int)
    
    # Create subsets
    train_subset = Subset(dataset, np.arange(sizes_abs[0]))
    val_subset = Subset(dataset, np.arange(sizes_abs[0], sizes_abs[0]+sizes_abs[1]))
    test_subset = Subset(dataset, np.arange(sizes_abs[0]+sizes_abs[1], sizes_abs.sum()))
    
    return train_subset, val_subset, test_subset

def print_dataloaders_overview(cfg: OmegaConf, train_loader, val_loader, test_loader, topo_data):
    print("\nDataloaders and Topology Data Overview:")
    print(f"Batch size: {cfg.train.batch_size}")
    print("-" * 58)
    print(f"{'Name':20} | {'Samples':12} | {'Batches':12}")
    print("-" * 58)
    print(f"{'Train Loader':20} | {len(train_loader.dataset):<12} | {len(train_loader):<12}")
    if val_loader:
        print(f"{'Val Loader':20} | {len(val_loader.dataset):<12} | {len(val_loader):<12}")
    if test_loader:
        print(f"{'Test Loader':20} | {len(test_loader.dataset):<12} | {len(test_loader):<12}")
    print("\nTopology Data:")
    print(f"Number of nodes: {topo_data.num_nodes}")
    print(f"Number of edges: {topo_data.edge_index.shape[1]}")
    if hasattr(topo_data, 'patches'):
        print(f"Number of patches: {len(topo_data.patches)}")
    print("-" * 58)

def create_dataloaders(cfg: OmegaConf, raw_data_dir=os.path.join(os.path.dirname(__file__), '../data'), num_workers=4):
    x, y, mask_x, mask_y, edge_index, edge_weight, n_nodes, metadata = get_data_raw(cfg, raw_data_dir)

    dataset = DynamicNodeFeatureDataset(x, y, mask_x, mask_y)
    train_dataset, val_dataset, test_dataset = split_dataset(cfg, dataset)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=True, persistent_workers=True) \
        if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=True, persistent_workers=True) \
        if len(test_dataset) > 0 else None

    topo_data = StaticGraphTopologyData(edge_index, edge_weight, n_nodes)
    # pre_transform = PositionalEncodingTransform(rw_dim=cfg.pos_enc.rw_dim, lap_dim=cfg.pos_enc.lap_dim)
    if cfg.metis.n_patches > 0:
        transform_train = GraphPartitionTransform(
            n_patches=cfg.metis.n_patches,
            metis=cfg.metis.enable,
            drop_rate=0,
            num_hops=cfg.metis.num_hops,
            is_directed=False,
            patch_rw_dim=cfg.pos_enc.patch_rw_dim,
            patch_num_diff=cfg.pos_enc.patch_num_diff
        )
        
        # topo_data = pre_transform(topo_data)
        topo_data = transform_train(topo_data)

    print_dataloaders_overview(cfg, train_loader, val_loader, test_loader, topo_data)

    return train_loader, val_loader, test_loader, topo_data, metadata