import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch_scatter import scatter
from einops import rearrange
from model.mlp_mixer import MLPMixerTemporal

from model.elements import MLP
from model.gnn import GNN
from model.readouts import SingleNodeReadout


class GMMModel(pl.LightningModule):

    def __init__(self, cfg, topo_data):
        super().__init__()
        assert cfg.metis.n_patches > 0
        self.cfg = cfg
        self.topo_data = topo_data
        self.criterion = torch.nn.L1Loss()  # MAE

        # Shortcuts
        nhid = cfg.model.hidden_size
        self.use_rw = cfg.pos_enc.rw_dim > 0
        self.use_lap = cfg.pos_enc.lap_dim > 0
        self.pooling = cfg.model.pool
        self.patch_rw_dim = cfg.pos_enc.patch_rw_dim

        if cfg.pos_enc.rw_dim > 0:
            self.rw_encoder = MLP(cfg.pos_enc.rw_dim, nhid, 1)
        if cfg.pos_enc.lap_dim > 0:
            self.lap_encoder = MLP(cfg.pos_enc.lap_dim, nhid, 1)
        if cfg.pos_enc.patch_rw_dim > 0:
            self.patch_rw_encoder = MLP(cfg.pos_enc.patch_rw_dim, nhid, 1)

        self.input_encoder = nn.Linear(1, nhid)
        self.edge_encoder = nn.Linear(1, nhid)

        self.gnns = nn.ModuleList([
            GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type=cfg.model.gnn_type, bn=True, dropout=cfg.train.dropout, res=True)
            for _ in range(cfg.model.nlayer_gnn)
        ])
        self.U = nn.ModuleList([
            MLP(nhid, nhid, nlayer=1, with_final_activation=True)
            for _ in range(cfg.model.nlayer_gnn-1)
        ])

        self.transformer_encoder = MLPMixerTemporal(nhid=nhid, dropout=cfg.train.mlpmixer_dropout, nlayer=cfg.model.nlayer_mlpmixer, n_patches=cfg.metis.n_patches, n_timesteps=cfg.train.window)
        self.readout = SingleNodeReadout(nhid, cfg.train.window, cfg.train.horizon, self.topo_data, n_layers=2)

        # Test run: Simple LSTM per node
        # nhid = 256
        # self.lstm = nn.LSTM(input_size=1, hidden_size=nhid, num_layers=4, batch_first=True)
        # self.readout2 = MLP(nhid, self.horizon, nlayer=3, with_final_activation=False)

    def forward(self, x):
        x_raw = x
        x = rearrange(self.input_encoder(x.unsqueeze(-1)), 'B n t f -> B t n f')
        edge_weight = self.edge_encoder(self.topo_data.edge_weight.unsqueeze(-1)).unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1, 1)  # Add time dimension

        # Patch Encoder
        x = x[..., self.topo_data.subgraphs_nodes_mapper, :]
        e = edge_weight[..., self.topo_data.subgraphs_edges_mapper, :]
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
        
