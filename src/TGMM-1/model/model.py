import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch_scatter import scatter
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

    def forward(self, x, data):
        x = rearrange(x, 'B t p f -> B p (t f)')

        out_all = torch.zeros(x.shape[0], self.n_nodes, self.horizon)
        for i, (mlp, patch_node_map) in enumerate(zip(self.mlps, self.patch_node_map)):  # TODO: Use scatter for this instead?
            out_patch = mlp(x[:, i, :])
            out_reshaped = rearrange(out_patch, 'B (n h) -> B n h', h=self.horizon)
            out_all[:, patch_node_map, :] = out_reshaped

        out_all = rearrange(out_all, 'B n h -> (n B) h')
        return out_all
    

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


class GMMModel(pl.LightningModule):

    def __init__(self, cfg, topo_data):
        super().__init__()
        assert cfg.metis.n_patches > 0

        self.bn = True
        self.res = True

        self.cfg = cfg
        self.nhid = cfg.model.hidden_size
        self.n_patches = cfg.metis.n_patches
        self.nlayer_gnn = cfg.model.nlayer_gnn
        self.nlayer_mlpmixer = cfg.model.nlayer_mlpmixer
        self.gnn_type = cfg.model.gnn_type
        self.pooling = cfg.model.pool
        self.dropout = cfg.train.dropout
        self.mlpmixer_dropout = cfg.train.mlpmixer_dropout
        self.rw_dim = cfg.pos_enc.rw_dim
        self.lap_dim = cfg.pos_enc.lap_dim
        self.patch_rw_dim = cfg.pos_enc.patch_rw_dim
        self.use_rw = self.rw_dim > 0
        self.use_lap = self.lap_dim > 0

        self.topo_data = topo_data
        self.criterion = torch.nn.L1Loss()  # MAE
        self.window = cfg.train.window
        self.horizon = cfg.train.horizon
        self.n_nodes = self.topo_data.num_nodes

        if self.use_rw:
            self.rw_encoder = MLP(self.rw_dim, self.nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(self.lap_dim, self.nhid, 1)
        if self.patch_rw_dim > 0:
            self.patch_rw_encoder = MLP(self.patch_rw_dim, self.nhid, 1)

        self.input_encoder = nn.Linear(1, self.nhid)
        self.edge_encoder = nn.Linear(1, self.nhid)

        self.gnns = nn.ModuleList([GNN(nin=self.nhid, nout=self.nhid, nlayer_gnn=1, gnn_type=self.gnn_type,
                                  bn=self.bn, dropout=self.dropout, res=self.res) for _ in range(self.nlayer_gnn)])
        self.U = nn.ModuleList([MLP(self.nhid, self.nhid, nlayer=1, with_final_activation=True) for _ in range(self.nlayer_gnn-1)])


        self.transformer_encoder = MLPMixerTemporal(nhid=self.nhid, dropout=self.mlpmixer_dropout, nlayer=self.nlayer_mlpmixer, n_patches=self.n_patches, n_timesteps=self.window)
        self.readout = SingleNodeReadout(self.nhid, self.n_patches, self.window, self.n_nodes, self.horizon, self.topo_data)
        # torch.nn.Linear(nhid*self.n_patches*self.num_timesteps, self.n_nodes*self.horizon)


        # Test run: Simple LSTM per node
        # nhid = 256
        # self.lstm = nn.LSTM(input_size=1, hidden_size=nhid, num_layers=4, batch_first=True)
        # self.readout2 = MLP(nhid, self.horizon, nlayer=3, with_final_activation=False)

    def forward(self, x):
        x_raw = x
        x = rearrange(self.input_encoder(x.unsqueeze(-1)), 'B n t f -> B t n f')
        # Node PE (positional encoding, either random walk or Laplacian eigenvectors, canonical choices for graphs)
        if self.use_rw:
            x += self.rw_encoder(self.topo_data.rw_pos_enc).unsqueeze(-3)  # Add time dimension
        if self.use_lap:
            x += self.lap_encoder(self.topo_data.lap_pos_enc).unsqueeze(-3)  # Add time dimension
        edge_attr = self.topo_data.edge_attr
        assert edge_attr is not None
        # if edge_attr is None:
        #     edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1), dtype=torch.float32)
        edge_attr = self.edge_encoder(edge_attr.unsqueeze(-1)).unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1, 1)  # Add time dimension

        # Patch Encoder
        x = x[..., self.topo_data.subgraphs_nodes_mapper, :]
        e = edge_attr[..., self.topo_data.subgraphs_edges_mapper, :]
        edge_index = self.topo_data.combined_subgraphs
        batch_x = self.topo_data.subgraphs_batch

        for i, gnn in enumerate(self.gnns):
            if i > 0:
                # x: (n_timesteps, n_nodes, n_features); subgraph: (n_timesteps, n_patches, n_features)
                subgraph = scatter(x, batch_x, dim=-2, reduce=self.pooling)[..., batch_x, :]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, self.topo_data.subgraphs_nodes_mapper, dim=-2, reduce='mean')[..., self.topo_data.subgraphs_nodes_mapper, :]
            x = gnn(x, edge_index, e)
        subgraph_x = scatter(x, self.topo_data.subgraphs_batch, dim=-2, reduce=self.pooling)  # (n_timesteps, n_patches, n_features)

        # Patch PE
        if self.patch_rw_dim > 0:
            subgraph_x += self.patch_rw_encoder(self.topo_data.patch_pe)

        # MLPMixer
        mixer_x = self.transformer_encoder(subgraph_x, None, None)

        # Decoding / Readout
        out = self.readout(mixer_x, x_raw, self.topo_data)

        # Test run: Simple LSTM per node
        # batch_size = x.shape[0]
        # x = rearrange(x, 'B n t -> (B n) t').unsqueeze(-1)[:2]
        # out, _ = self.lstm(x)  # out shape: (batch, seq_len, hidden_size)
        # out = self.readout2(out[:, -1, :])  # Transform to predict next 12 timesteps
        # out = rearrange(out, '(B n) h -> B n h', B=batch_size)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.use_lap:  # FIXME: Check if needed
            batch_pos_enc = self.topo_data.lap_pos_enc
            sign_flip = (2 * (torch.rand(batch_pos_enc.size(1)) >= 0.5) - 1).float()  # array of +-1
            self.topo_data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)

        y_pred = self.forward(x)  # (batch_size, n_nodes*n_timesteps)
        loss = self.criterion(y_pred, y)
        metrics = self.calc_metrics(y_pred, y, key_prefix='train')
        metrics.update({'train/loss': loss})
        metrics.update({'train/lr': self.optimizers().param_groups[0]['lr']})
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        metrics = self.calc_metrics(y_pred, y, key_prefix='valid')
        metrics.update({'valid/loss': loss})
        metrics.update({'valid/lr': self.optimizers().param_groups[0]['lr']})
        self.log_dict(metrics)
        return loss
    
    def calc_metrics(self, pred, targets, key_prefix='valid'):

        def calc_metrics_for_horizon(pred, targets, horizon):
            return {
                f'{key_prefix}/mae-{horizon}': torch.mean(torch.abs(pred[..., horizon-1] - targets[..., horizon-1])),
                f'{key_prefix}/rmse-{horizon}': torch.sqrt(torch.mean((pred[..., horizon-1] - targets[..., horizon-1])**2)),
                f'{key_prefix}/mape-{horizon}': 100*torch.mean(torch.abs((pred[..., horizon-1] - targets[..., horizon-1]) / targets[..., horizon-1])),
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
                "monitor": self.cfg.train.monitor,
                "interval": "epoch",
                "frequency": 1,
            },
        }
        
