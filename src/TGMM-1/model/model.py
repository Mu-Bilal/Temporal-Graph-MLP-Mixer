import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch_scatter import scatter
from einops.layers.torch import Rearrange
from model.mlp_mixer import MLPMixer

from model.elements import MLP
from model.gnn import GNN


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
                 cfg=None):

        super().__init__()
        self.dropout = dropout
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0

        self.pooling = pooling
        self.res = res
        self.patch_rw_dim = patch_rw_dim

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)
        if self.patch_rw_dim > 0:
            self.patch_rw_encoder = MLP(self.patch_rw_dim, nhid, 1)

        self.input_encoder = nn.Embedding(nfeat_node, nhid)
        self.edge_encoder = nn.Embedding(nfeat_edge, nhid)

        self.gnns = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type=gnn_type,
                                  bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])
        self.U = nn.ModuleList([MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn-1)])

        self.reshape = Rearrange('(B p) d ->  B p d', p=n_patches)

        self.transformer_encoder = MLPMixer(nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)
        self.output_decoder = MLP(nhid, nout, nlayer=2, with_final_activation=False)
        
        # New 
        self.criterion = torch.nn.L1Loss()  # MAE
        self.cfg = cfg

    def forward(self, data):
        x = self.input_encoder(data.x.squeeze())

        # Node PE (positional encoding, either random walk or Laplacian eigenvectors, canonical choices for graphs)
        if self.use_rw:
            x += self.rw_encoder(data.rw_pos_enc)
        if self.use_lap:
            x += self.lap_encoder(data.lap_pos_enc)
        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        edge_attr = self.edge_encoder(edge_attr)

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
        mixer_x = self.reshape(subgraph_x)

        # MLPMixer
        mixer_x = self.transformer_encoder(mixer_x, data.coarsen_adj if hasattr(data, 'coarsen_adj') else None, ~data.mask)

        # Global Average Pooling
        x = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / data.mask.sum(1, keepdim=True)

        # Readout
        out = self.output_decoder(x).squeeze()
        return out

    def training_step(self, batch, batch_idx):
        if self.use_lap:
            batch_pos_enc = batch.lap_pos_enc
            sign_flip = (2 * (torch.rand(batch_pos_enc.size(1)) >= 0.5) - 1).float()  # array of +-1
            batch.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)

        out = self.forward(batch)
        targets = batch.y.long()
        loss = self.criterion(out, targets)
        self.log_dict({'train_loss': loss}, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        targets = batch.y.long()
        loss = self.criterion(out, targets)
        self.log_dict({'val_loss': loss}, batch_size=batch.num_graphs)
        return loss
        
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
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
        
