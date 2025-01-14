from omegaconf import OmegaConf
import tsl.datasets as tsl_datasets
from scipy.sparse import csr_matrix
import os

from hdtts_dataset_creation import GraphMSO, EngRad, PvUS    # FIXME: What about GraphMSO
from hdtts_dataset_creation.mask import add_missing_values
from hdtts_dataset_creation.pooling_utils import make_graph_connected_

def make_graph_connected(dataset, dataset_cfg):
    # Check for unconnected components
    import scipy.sparse as sp
    dscfg = dict(dataset_cfg.connectivity)
    dscfg['layout'] = 'coo'
    adj = dataset.get_connectivity(**dscfg)

    num_components, _ = sp.csgraph.connected_components(adj, connection='weak')
    delta = num_components

    dscfg['layout'] = 'dense'
    adj = dataset.get_connectivity(**dscfg)

    sim = dataset.get_similarity()

    while num_components > 1 and delta > 0:
        adj = make_graph_connected_(
            adj, sim, dataset_cfg.connectivity.get('threshold', 0.1))
        new_components, _ = sp.csgraph.connected_components(
            sp.csr_matrix(adj), connection='weak')
        delta = num_components - new_components
        num_components = new_components

    # convert layout
    if dataset_cfg.connectivity.layout == 'edge_index':
        from tsl.ops.connectivity import adj_to_edge_index
        return adj_to_edge_index(adj)
    elif dataset_cfg.connectivity.layout == 'csr':
        return csr_matrix(adj)
    else:
        raise NotImplementedError()

def load_synthetic_dataset(dataset_cfg: OmegaConf, root_dir: str):
    if dataset_cfg.name.startswith('mso'):
        dataset = GraphMSO(root=os.path.join(root_dir, 'GraphMSO'), **dataset_cfg.hparams)
    else:
        raise ValueError(f"Dataset {dataset_cfg.name} not available.")

    mask_original = dataset.get_mask(dtype=bool, as_dataframe=False)

    # Override mask with injected missing values
    dataset.set_mask(dataset.training_mask)
    mask = dataset.get_mask(dtype=bool, as_dataframe=False)

    # Get connectivity
    adj = dataset.get_connectivity(**dataset_cfg.connectivity)

    return dataset, adj, mask_original, mask

def load_dataset(dataset_cfg: OmegaConf, root_dir: str):
    # Get the dataset
    name: str = dataset_cfg.name
    # Environmental datasets
    if name.startswith('air'):
        dataset = tsl_datasets.AirQuality(root=os.path.join(root_dir, 'AirQuality'), impute_nans=False)
        dataset.target.fillna(0, inplace=True)
    elif name.startswith('engrad'):
        dataset = EngRad(root=os.path.join(root_dir, 'EngRAD'), **dataset_cfg.hparams)
    elif name.startswith('pvus'):
        dataset = PvUS(root=os.path.join(root_dir, 'PvUS'), **dataset_cfg.hparams)
        # Remove broken node
        node_index = [i for i in range(dataset.n_nodes) if i != 485]
        dataset.reduce_(node_index=node_index)
    # Traffic datasets
    elif name.startswith('la'):
        dataset = tsl_datasets.MetrLA(root=os.path.join(root_dir, 'MetrLA'), impute_zeros=True)
    elif name.startswith('bay'):
        dataset = tsl_datasets.PemsBay(root=os.path.join(root_dir, 'PemsBay'))
    else:
        raise ValueError(f"Dataset {name} not available.")

    # Get connectivity
    if dataset_cfg.make_graph_connected:
        # Connect disconnected components
        adj = make_graph_connected(dataset, dataset_cfg)
    else:
        adj = dataset.get_connectivity(**dataset_cfg.connectivity)

    # Get original mask
    mask_original = dataset.get_mask().copy()  # [time, node, feature]

    # Add missing values to dataset
    if dataset_cfg.mode.name != 'normal':
        add_missing_values(dataset,
                           p_fault=dataset_cfg.mode.p_fault,
                           p_noise=dataset_cfg.mode.p_noise,
                           min_seq=dataset_cfg.mode.min_seq,
                           max_seq=dataset_cfg.mode.max_seq,
                           p_propagation=dataset_cfg.mode.get(
                               'p_propagation', 0),
                           connectivity=adj,
                           propagation_hops=dataset_cfg.mode.get(
                               'propagation_hops', 0),
                           seed=dataset_cfg.mode.seed)
        dataset.set_mask(dataset.training_mask)

    # Add just one valid night values for MinMaxScaler
    if isinstance(dataset, (PvUS, EngRad)):
        dataset.mask[0] = True

    return dataset, adj, mask_original
