from torch_scatter import scatter
from einops import rearrange
import torch
import torch.nn as nn

from model.elements import MLP


class SingleNodeReadout(nn.Module):
    """
    Uses a single MLP a patch + node input to the node output.
    """
    def __init__(self, n_features_patch, n_features_node, timesteps, horizon, topo_data, n_layers):
        super().__init__()

        in_dim = n_features_patch*timesteps + n_features_node*timesteps
        out_dim = horizon

        self.subgraphs_nodes_mapper = topo_data.subgraphs_nodes_mapper
        self.subgraphs_batch = topo_data.subgraphs_batch

        self.mlp = MLP(in_dim, out_dim, nlayer=n_layers, with_final_activation=False)

    def forward(self, patch_x, nodes_x):
        """
        patch_x: (batch_size, n_timesteps, n_patches, n_features_patch)
        nodes_x: (batch_size, n_timesteps, n_nodes, n_features_node)
        topo_data: CustomTemporalData

        For each node, we have a single MLP that takes the patch + node input and outputs the node output.
        """
        patch_x_nodes = scatter(patch_x[..., self.subgraphs_batch, :], self.subgraphs_nodes_mapper, dim=-2, reduce='mean')
        patch_x_nodes = rearrange(patch_x_nodes, 'B t n f -> B n (t f)')
        nodes_x = rearrange(nodes_x, 'B t n f -> B n (t f)')
        mlp_in = torch.cat([nodes_x, patch_x_nodes], dim=-1)  # (batch_size, n_nodes, n_timesteps*n_features_node + n_timesteps*n_features_patch)
        out = self.mlp(mlp_in)  # mlp_out: (batch_size, n_nodes*horizon); mlp_in: (batch_size, n_timesteps*n_features_node + n_timesteps*n_features_patch)
        return out  # (batch_size, n_nodes, horizon)
