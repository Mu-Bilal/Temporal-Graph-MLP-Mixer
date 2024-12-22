from torch_scatter import scatter
from einops import rearrange
import torch
import torch.nn as nn

from model.elements import MLP


class SingleNodeReadout(nn.Module):
    """
    Uses a single MLP a patch + node input to the node output.
    """
    def __init__(self, nhid, n_patches, timesteps, nout, horizon, topo_data):
        super().__init__()

        in_dim = nhid*timesteps + timesteps
        out_dim = horizon
        self.horizon = horizon
        self.n_nodes = topo_data.num_nodes

        # Following are for a whole batch, i.e. subgraph_count = batch_size * n_patches
        self.subgraph_count = topo_data.subgraphs_batch.max() + 1
        self.patch_node_map = [topo_data.subgraphs_nodes_mapper[topo_data.subgraphs_batch == i] for i in range(self.subgraph_count)]

        self.mlp = MLP(in_dim, out_dim, nlayer=4, with_final_activation=False)

    def forward(self, mixer_x, x, topo_data):
        """
        mixer_x: (batch_size, n_timesteps, n_patches, nhid)
        topo_data: CustomTemporalData

        For each node, we have a single MLP that takes the patch + node input and outputs the node output.
        """
        mixer_x_nodes = scatter(mixer_x[..., topo_data.subgraphs_batch, :], topo_data.subgraphs_nodes_mapper, dim=-2, reduce='mean')
        mixer_x_nodes = rearrange(mixer_x_nodes, 'B t n f -> B n (t f)')
        mlp_in = torch.cat([x, mixer_x_nodes], dim=-1)  # (batch_size, n_nodes, n_timesteps*nhid + n_timesteps)
        out = self.mlp(mlp_in)  # mlp_out: (batch_size, n_nodes*horizon); mlp_in: (batch_size, n_timesteps*nhid + n_timesteps)
        return out  # (batch_size, n_nodes, horizon)
