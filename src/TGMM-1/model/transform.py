import torch


from model.subgraph_extractors import metis_subgraph, random_subgraph
from model.positional_encoding import RWSE, LapPE, random_walk


def cal_coarsen_adj(subgraphs_nodes_mask):
    """Calculates coarsened adjacency matrix from subgraph node masks"""
    mask = subgraphs_nodes_mask.to(torch.float)
    coarsen_adj = torch.matmul(mask, mask.t())
    return coarsen_adj


def to_sparse(node_mask, edge_mask):
    """Converts dense node and edge masks to sparse format"""
    subgraphs_nodes = node_mask.nonzero().T
    subgraphs_edges = edge_mask.nonzero().T
    return subgraphs_nodes, subgraphs_edges


def combine_subgraphs(edge_index, subgraphs_nodes, subgraphs_edges, num_selected=None, num_nodes=None):
    """Combines multiple subgraphs into a single graph by relabeling nodes.
    
    This function takes multiple subgraphs and combines them into one large graph by:
    1. Taking the original edge indices and selecting only edges that appear in subgraphs
    2. Creating a mapping to relabel nodes to ensure each subgraph has unique node IDs
    3. Offsetting node IDs for each subgraph to prevent overlap
    
    Args:
        edge_index: Original graph edge indices tensor of shape [2, num_edges]
        subgraphs_nodes: Tuple of [batch_idx, node_idx] tensors indicating node assignments
        subgraphs_edges: Tuple of [batch_idx, edge_idx] tensors indicating edge assignments
        num_selected: Number of subgraphs (defaults to max batch index + 1)
        num_nodes: Number of nodes in original graph (defaults to max node index + 1)
    
    Returns:
        combined_subgraphs: Edge index tensor for the combined graph with relabeled nodes
    """
    if num_selected is None:
        num_selected = subgraphs_nodes[0][-1] + 1
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1

    # Select edges that appear in subgraphs
    combined_subgraphs = edge_index[:, subgraphs_edges[1]]
    
    # Create mapping matrix to relabel nodes
    node_label_mapper = edge_index.new_full((num_selected, num_nodes), -1)
    node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]] = torch.arange(len(subgraphs_nodes[1]))
    node_label_mapper = node_label_mapper.reshape(-1)

    # Offset node IDs by subgraph index * num_nodes to ensure unique IDs
    inc = torch.arange(num_selected)*num_nodes
    combined_subgraphs += inc[subgraphs_edges[0]]
    
    # Apply node relabeling
    combined_subgraphs = node_label_mapper[combined_subgraphs]
    return combined_subgraphs


class PositionalEncodingTransform(object):
    """Adds random walk and Laplacian positional encodings to graph data"""
    def __init__(self, rw_dim=0, lap_dim=0):
        super().__init__()
        self.rw_dim = rw_dim
        self.lap_dim = lap_dim

    def __call__(self, data):
        if self.rw_dim > 0:
            data.rw_pos_enc = RWSE(data.edge_index, self.rw_dim, data.num_nodes)
        if self.lap_dim > 0:
            data.lap_pos_enc = LapPE(data.edge_index, self.lap_dim, data.num_nodes)
        return data


class GraphPartitionTransform(object):
    """
    Graph partitioning transform. Extracts subgraphs from the graph using either METIS or random partitioning.
    Adds positional encodings and handles diffusion between partitions.
    """
    def __init__(self, n_patches, metis=True, drop_rate=0.0, num_hops=1, is_directed=False, patch_rw_dim=0, patch_num_diff=0):
        super().__init__()
        self.n_patches = n_patches
        self.drop_rate = drop_rate
        self.num_hops = num_hops
        self.is_directed = is_directed
        self.patch_rw_dim = patch_rw_dim
        self.patch_num_diff = patch_num_diff
        self.metis = metis

    def _diffuse(self, A):
        """Performs diffusion on adjacency matrix by computing powers of random walk matrix"""
        if self.patch_num_diff == 0:
            return A
        Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)  # D^-1
        RW = A * Dinv
        M = RW
        M_power = M
        for _ in range(self.patch_num_diff-1):
            M_power = torch.matmul(M_power, M)
        return M_power

    def __call__(self, data):
        """
        data is now an instance of CustomTemporalData
        Main transform function that:
        1. Partitions graph into subgraphs
        2. Computes positional encodings
        3. Handles diffusion between partitions
        4. Creates masks and mappings for subgraph processing
        """
        if self.metis:
            node_masks, edge_masks = metis_subgraph(
                data, n_patches=self.n_patches, drop_rate=self.drop_rate, num_hops=self.num_hops, is_directed=self.is_directed)
        else:
            node_masks, edge_masks = random_subgraph(
                data, n_patches=self.n_patches, num_hops=self.num_hops)
        subgraphs_nodes, subgraphs_edges = to_sparse(node_masks, edge_masks)
        combined_subgraphs = combine_subgraphs(
            data.edge_index, subgraphs_nodes, subgraphs_edges, num_selected=self.n_patches, num_nodes=data.num_nodes)

        if self.patch_num_diff > -1 or self.patch_rw_dim > 0:
            coarsen_adj = cal_coarsen_adj(node_masks)
            if self.patch_rw_dim > 0:
                data.patch_pe = random_walk(coarsen_adj, self.patch_rw_dim)
            if self.patch_num_diff > -1:
                data.coarsen_adj = self._diffuse(coarsen_adj).unsqueeze(0)

        subgraphs_batch = subgraphs_nodes[0]
        mask = torch.zeros(self.n_patches).bool()
        mask[subgraphs_batch] = True
        data.subgraphs_batch = subgraphs_batch
        data.subgraphs_nodes_mapper = subgraphs_nodes[1]
        data.subgraphs_edges_mapper = subgraphs_edges[1]
        data.combined_subgraphs = combined_subgraphs
        data.mask = mask.unsqueeze(0)

        data.__num_nodes__ = data.num_nodes  # set number of nodes of the current graph
        return data
