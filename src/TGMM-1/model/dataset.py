import torch_geometric_temporal as tgt
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf
from torch_geometric_temporal.dataset import METRLADatasetLoader
from typing import Iterable

from model.transform import GraphPartitionTransform, PositionalEncodingTransform


def create_train_val_test_split(dataset, max_len: int = None, train_size: float = 0.7, val_size: float = 0.15):
    if isinstance(dataset, tgt.signal.static_graph_temporal_signal.StaticGraphTemporalSignal):
        total_snapshots = dataset.snapshot_count
    else:
        total_snapshots = len(dataset)
    if max_len:
        total_snapshots = min(total_snapshots, max_len)
    train_size_abs = int(train_size * total_snapshots)
    val_size_abs = int(val_size * total_snapshots)

    train_data = dataset[:train_size_abs]
    val_data = dataset[train_size_abs:train_size_abs+val_size_abs]

    if train_size_abs+val_size_abs < total_snapshots:
        test_data = dataset[train_size_abs+val_size_abs:total_snapshots]
    else:
        test_data = None

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
            graph_transform = None,
            batch_size: int = 1
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
        return CustomTemporalData(data.edge_index, data.edge_attr, data.num_nodes, data.x, data.y)
    
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
    

def create_dataloaders(cfg: OmegaConf, dataset_name: str, max_len: int = None, train_size: float = 0.7, val_size: float = 0.15):
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
    train_data, val_data, test_data = create_train_val_test_split(dataset_metrola, max_len=max_len, train_size=train_size, val_size=val_size)

    train_data = CustomTemporalDataset(train_data, graph_transform=[pre_transform, transform_train])
    train_loader = DataLoader(train_data, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    val_data = CustomTemporalDataset(val_data, graph_transform=[pre_transform, transform_eval])
    val_loader = DataLoader(val_data, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=True)

    if test_data is not None:
        test_data = CustomTemporalDataset(test_data, graph_transform=[pre_transform, transform_eval])
        test_loader = DataLoader(test_data, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=True)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
