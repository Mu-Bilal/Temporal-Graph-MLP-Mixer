import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch_scatter import scatter
from einops.layers.torch import Rearrange
from einops import rearrange
from model.mlp_mixer import MLPMixerTemporal
import pandas as pd

from model.elements import MLP
from model.gnn import GNN



class CompletePatchReadout(nn.Module):
    """
    Uses a seperate MLP to decode each patch to all its nodes.
    """
    def __init__(self, nhid, n_patches, timesteps, nout, horizon, data_example):  # FIXME: Nout = nout*nNodes but here nout=1; Dont actually pass the whole data_example!
        super().__init__()
        self.rearrange_in = Rearrange('b t p f -> b p (t f)')
        self.rearrange_out = Rearrange('b p (t f) -> b t p f', t=timesteps)
        # out = rearrange(out, 'b (n_nodes horizon) -> b n_nodes horizon', n_nodes=self.n_nodes)

        self.mlps = []
        self.nparams = 0
        for i, patch_node_cnt in enumerate(pd.Series(data_example.subgraphs_batch).value_counts(sort=False)): 
            mlp = nn.Linear(nhid*timesteps, patch_node_cnt*horizon)  # in: Data from batch (n_timesteps*n_features) -> out: (n_nodes_per_THIS_patch, horizon)
            self.mlps.append(mlp)
            self.nparams += (nhid*timesteps+1)*patch_node_cnt*horizon
        
        print(f"Readout param count: {self.nparams}")
        self.subgraph_count = data_example.subgraphs_batch.max()
        self.patch_node_map = [data_example.subgraphs_nodes_mapper[data_example.subgraphs_batch == i] for i in range(self.subgraph_count)]
        self.horizon = horizon
        self.n_nodes = data_example.num_nodes

    def forward(self, x):
        x = self.rearrange_in(x)

        out_all = torch.zeros(x.shape[0], self.n_nodes, self.horizon)
        for i, (mlp, patch_node_map) in enumerate(zip(self.mlps, self.patch_node_map)):  # TODO: Use scatter for this instead?
            out_patch = mlp(x[:, i, :])
            out_reshaped = rearrange(out_patch, 'b (num_nodes h) -> b num_nodes h', h=self.horizon)  # FIXME
            out_all[:, patch_node_map, :] = out_reshaped

        return out_all
    

class SingleNodeReadout(nn.Module):
    """
    Uses a single MLP a patch + node input to the node output.
    """
    def __init__(self, nhid, n_patches, timesteps, nout, horizon, data_example):
        super().__init__()

        in_dim = nhid*timesteps + timesteps
        out_dim = horizon
        self.horizon = horizon
        self.n_nodes = data_example.num_nodes
        self.subgraph_count = data_example.subgraphs_batch.max()
        self.patch_node_map = [data_example.subgraphs_nodes_mapper[data_example.subgraphs_batch == i] for i in range(self.subgraph_count)]

        self.mlp = MLP(in_dim, out_dim, nlayer=2, with_final_activation=False)

    def forward(self, mixer_x, data):
        """
        mixer_x: (batch_size, n_timesteps, n_patches, n_features)
        data: CustomTemporalData

        For each node, we have a single MLP that takes the patch + node input and outputs the node output.
        """
        out_all = torch.zeros(1, self.n_nodes, self.horizon)
        for patch_idx, associated_nodes_idx in enumerate(self.patch_node_map):
            # In the following n_nodes is the number of nodes in the current patch
            patch_x = mixer_x[:, :, patch_idx, :]  # (batch_size, n_timesteps, n_hidden)
            patch_x = rearrange(patch_x, 'b t f -> b (t f)').unsqueeze(1)  # (batch_size, 1, n_timesteps*n_hidden)
            patch_x = patch_x.repeat(1, associated_nodes_idx.size(0), 1)  # (batch_size, n_nodes, n_timesteps*n_hidden)
            node_x = data.features[associated_nodes_idx].unsqueeze(0)  # (batch_size=1, n_nodes, n_timesteps) FIXME: Check batch size
            x = torch.cat([patch_x, node_x], dim=-1)  # (batch_size, n_nodes, n_timesteps*n_hidden + n_timesteps)
            out = self.mlp(x)
            out_all[0, associated_nodes_idx, :] = out
        return out_all

class GMMModel(pl.LightningModule):

    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 nlayer_mlpmixer,
                 gnn_type,
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 mlpmixer_dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean',
                 n_patches=32,
                 patch_rw_dim=0,
                 cfg=None,
                 data_example=None):

        super().__init__()
        assert data_example is not None
        self.dropout = dropout
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0

        self.pooling = pooling
        self.res = res
        self.patch_rw_dim = patch_rw_dim

        self.nhid = nhid
        self.n_patches = n_patches

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, self.nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, self.nhid, 1)
        if self.patch_rw_dim > 0:
            self.patch_rw_encoder = MLP(self.patch_rw_dim, self.nhid, 1)

        self.input_encoder = nn.Linear(nfeat_node, self.nhid)
        self.edge_encoder = nn.Linear(nfeat_edge, self.nhid)

        self.gnns = nn.ModuleList([GNN(nin=self.nhid, nout=self.nhid, nlayer_gnn=1, gnn_type=gnn_type,
                                  bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])
        self.U = nn.ModuleList([MLP(self.nhid, self.nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn-1)])

        
        self.reshape_patchpe = Rearrange('(B p) d ->  B p d', p=self.n_patches)  # (batch_size, n_patches, n_features) where n_features=n_hid is the number of features per patch
        # self.output_decoder = MLP(nhid, nout, nlayer=2, with_final_activation=False)
        
        # New 
        self.criterion = torch.nn.L1Loss()  # MAE
        self.cfg = cfg
        self.num_timesteps = data_example.features.shape[1]  # FIXME: make this dynamic
        self.horizon = data_example.targets.shape[1]
        self.n_nodes = data_example.num_nodes
        self.transformer_encoder = MLPMixerTemporal(nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=self.n_patches, num_timesteps=self.num_timesteps)
        self.readout = SingleNodeReadout(nhid, self.n_patches, self.num_timesteps, self.n_nodes, self.horizon, data_example)
        # torch.nn.Linear(nhid*self.n_patches*self.num_timesteps, self.n_nodes*self.horizon)

    def forward(self, data):
        """
        data is now an instance of CustomTemporalData

        CustomTemporalData(
            num_nodes: 207
            edge_index: (2, 1722)
            edge_weight: (1722,)
            features: (207, 12)
            targets: (207, 12)
        )
        """
        assert data.features.shape[1] == self.num_timesteps
        assert data.targets.shape[1] == self.horizon
        assert data.num_nodes == self.n_nodes

        all_mixer_x = torch.zeros(1, self.num_timesteps, self.n_patches, self.nhid)  # FIXME: Check if first dim (batch_size) is ever more than one?
        for i in range(data.features.shape[1]):  # Iterate over past timesteps
            # For each time step, encode the features
            x = self.input_encoder(data.features[:, i].unsqueeze(-1))
            # Node PE (positional encoding, either random walk or Laplacian eigenvectors, canonical choices for graphs)
            if self.use_rw:
                x += self.rw_encoder(data.rw_pos_enc)
            if self.use_lap:
                x += self.lap_encoder(data.lap_pos_enc)
            edge_attr = data.edge_attr
            if edge_attr is None:
                edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1), dtype=torch.float32)
            edge_attr = self.edge_encoder(edge_attr.unsqueeze(-1))

            # Patch Encoder
            x = x[data.subgraphs_nodes_mapper]
            e = edge_attr[data.subgraphs_edges_mapper]
            edge_index = data.combined_subgraphs
            batch_x = data.subgraphs_batch

            for i, gnn in enumerate(self.gnns):
                if i > 0:
                    subgraph = scatter(x, batch_x, dim=0, reduce=self.pooling)[batch_x]
                    x = x + self.U[i-1](subgraph)
                    x = scatter(x, data.subgraphs_nodes_mapper, dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
                x = gnn(x, edge_index, e)
            subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)

            # Patch PE
            if self.patch_rw_dim > 0:
                subgraph_x += self.patch_rw_encoder(data.patch_pe)
            mixer_x = self.reshape_patchpe(subgraph_x)
            all_mixer_x[0, i, :, :] = mixer_x  # TODO: Check if mixer_x does indeed have batch_size (first dim) = 0

        # MLPMixer
        mixer_x = self.transformer_encoder(all_mixer_x, None, None)

        # Decoding / Readout
        out = self.readout(mixer_x, data)

        return out

    def training_step(self, batch, batch_idx):
        """
        Batch is an instance of CustomTemporalData
        """
        if self.use_lap:
            batch_pos_enc = batch.lap_pos_enc
            sign_flip = (2 * (torch.rand(batch_pos_enc.size(1)) >= 0.5) - 1).float()  # array of +-1
            batch.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)

        pred = self.forward(batch)  # (batch_size, n_nodes*n_timesteps)
        targets = batch.y.unsqueeze(0)
        loss = self.criterion(pred, targets)
        metrics = self.calc_metrics(pred, targets, key_prefix='train')
        metrics.update({'train/loss': loss})
        self.log_dict(metrics, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        targets = batch.y.unsqueeze(0)
        loss = self.criterion(pred, targets)
        metrics = self.calc_metrics(pred, targets, key_prefix='valid')
        metrics.update({'valid/loss': loss})
        self.log_dict(metrics, batch_size=batch.num_graphs)
        return loss
    
    def calc_metrics(self, pred, targets, key_prefix='valid'):

        def calc_metrics_for_horizon(pred, targets, horizon):
            return {
                f'{key_prefix}/mae-{horizon}': torch.mean(torch.abs(pred[:, :, horizon-1] - targets[:, :, horizon-1])),
                f'{key_prefix}/rmse-{horizon}': torch.sqrt(torch.mean((pred[:, :, horizon-1] - targets[:, :, horizon-1])**2)),
                f'{key_prefix}/mape-{horizon}': 100*torch.mean(torch.abs((pred[:, :, horizon-1] - targets[:, :, horizon-1]) / targets[:, :, horizon-1])),
            }

        metrics = {}
        for horizon in [3, 6, 12]:
            metrics.update(calc_metrics_for_horizon(pred, targets, horizon))
        return metrics
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=self.cfg.train.lr_decay,
            patience=self.cfg.train.lr_patience,
            verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
        
