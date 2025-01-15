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
    def __init__(self, x_synth_missing, y, mask_x_synth, mask_y, mask_y_synth):
        """
        mask (pandas.Dataframe or numpy.ndarray, optional): Boolean mask
        denoting if values in data are valid (:obj:`True`) or not
        (:obj:`False`).
        """
        assert len(x_synth_missing) == len(y) == len(mask_x_synth) == len(mask_y) == len(mask_y_synth), "x_synth_missing, y, mask_x_synth, mask_y, and mask_y_synth must have the same length"
        assert len(x_synth_missing) > 0, "x_synth_missing, y, mask_x_synth, mask_y, and mask_y_synth must have at least one sample"

        assert x_synth_missing.ndim == 3, "x_synth_missing must have 3 dimensions"
        assert y.ndim == 3, "y must have 3 dimensions"
        assert mask_x_synth.ndim == 3, "mask_x_synth must have 3 dimensions"
        assert mask_y.ndim == 3, "mask_y must have 3 dimensions"
        assert mask_y_synth.ndim == 3, "mask_y_synth must have 3 dimensions"

        self.x = torch.tensor(x_synth_missing)
        self.y = torch.tensor(y)
        self.mask_x = torch.tensor(mask_x_synth, dtype=bool)
        self.mask_y = torch.tensor(mask_y, dtype=bool)
        self.mask_y_synth = torch.tensor(mask_y_synth, dtype=bool)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask_x[idx], self.mask_y[idx], self.mask_y_synth[idx]
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return f"DynamicNodeDataset(x={self.x.shape}, y={self.y.shape}, mask_x={self.mask_x.shape}, mask_y={self.mask_y.shape}, mask_y_synth={self.mask_y_synth.shape})"

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
    
def create_sliding_window_dataset(data, window, delay, horizon, stride, max_steps=None):
    steps = data.shape[0] - window - delay - horizon
    n_nodes = data.shape[1]

    if max_steps is not None:
        steps = min(steps, max_steps)
        
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
        dataset, adj, mask_original, mask_synth = load_synthetic_dataset(cfg.raw_data, root_dir=root)
        mask_synth = mask_synth.squeeze()

        # Get data and set missing values to nan
        data = dataset.dataframe()
        masked_data = data.where(mask_original.reshape(mask_original.shape[0], -1), np.nan)
        # Fill nan with Last Observation Carried Forward
        data = masked_data.ffill().bfill()
    else:
        dataset, adj, mask_original = load_dataset(cfg.raw_data, root_dir=root)
        mask_synth = dataset.mask.squeeze()

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

    # Add missing values (mask already contains missing flags generated by synthetic point/block defects). 
    # Now need to apply this to dataset.
    data_synth_missing = data.where(mask_synth.reshape(mask_synth.shape[0], -1), np.nan)
    if isinstance(dataset, PvUS):
        data_synth_missing = data_synth_missing.groupby([data_synth_missing.index.hour, data_synth_missing.index.minute]).ffill()
        data_synth_missing = data_synth_missing.groupby([data_synth_missing.index.hour, data_synth_missing.index.minute]).bfill()
    else:
        data_synth_missing = data_synth_missing.ffill().bfill()
    data_synth_missing.fillna(0, inplace=True)


    if isinstance(dataset, EngRad):  # FIXME: Quickfix as model is currently univariate
        data = data.iloc[:, cfg.dataset.EngRADchannel::5]
        data_synth_missing = data_synth_missing.iloc[:, cfg.dataset.EngRADchannel::5]

    data = data.to_numpy()
    data_synth_missing = data_synth_missing.to_numpy()

    if cfg.dataset.normalize:
        # Calculate statistics on the entire dataset
        norm_mean, norm_std = data_synth_missing.mean(), data_synth_missing.std()  # Use the stats that would be available
        data = (data - norm_mean) / norm_std
        data_synth_missing = (data_synth_missing - norm_mean) / norm_std
        metadata = {
            'norm_mean': norm_mean,
            'norm_std': norm_std
        }
    else:
        metadata = {}

    # TODO: Make this dynamic for multivariate data (currently assuming data has single feature)
    # TODO: Clean this up. If we never need y_synth_missing then don't compute it.

    # (n_timesteps, n_nodes, n_features)
    x, y = create_sliding_window_dataset(data, cfg.dataset.window, cfg.dataset.delay, cfg.dataset.horizon, cfg.dataset.stride, max_steps=cfg.dataset.max_len)
    x_synth_missing, y_synth_missing = create_sliding_window_dataset(data_synth_missing, cfg.dataset.window, cfg.dataset.delay, cfg.dataset.horizon, cfg.dataset.stride, max_steps=cfg.dataset.max_len)
    mask_x, mask_y = create_sliding_window_dataset(mask_original.squeeze(), cfg.dataset.window, cfg.dataset.delay, cfg.dataset.horizon, cfg.dataset.stride, max_steps=cfg.dataset.max_len)
    mask_x_synth_missing, mask_y_synth_missing = create_sliding_window_dataset(mask_synth.squeeze(), cfg.dataset.window, cfg.dataset.delay, cfg.dataset.horizon, cfg.dataset.stride, max_steps=cfg.dataset.max_len)

    # Static topology data
    n_nodes = dataset.n_nodes
    edge_index, edge_weight = adj_to_edge_index(adj.todense())
    edge_weight = np.asarray(edge_weight).squeeze()
    edge_index = np.asarray(edge_index)

    return x, y, x_synth_missing, y_synth_missing, mask_x, mask_y, mask_x_synth_missing, mask_y_synth_missing, edge_index, edge_weight, n_nodes, metadata

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
    x, y, x_synth_missing, y_synth_missing, mask_x, mask_y, mask_x_synth, mask_y_synth, edge_index, edge_weight, n_nodes, metadata = get_data_raw(cfg, raw_data_dir)

    dataset = DynamicNodeFeatureDataset(x_synth_missing, y, mask_x_synth, mask_y, mask_y_synth)
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