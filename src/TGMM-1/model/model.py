import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch_scatter import scatter
from einops.layers.torch import Rearrange
from einops import rearrange
from model.mlp_mixer import MLPMixerTemporal

from model.elements import MLP
from model.gnn import GNN



class Readout(nn.Module):
    def __init__(self, nhid, n_patches, timesteps, nout, horizon):  # FIXME: Nout = nout*nNodes but here nout=1
        super().__init__()
        self.mlp = MLP(nhid*n_patches*timesteps, nout*horizon, nlayer=2, with_final_activation=False)

    def forward(self, x):
        return self.mlp(x)


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

        
        self.reshape = Rearrange('(B p) d ->  B p d', p=self.n_patches)  # (batch_size, n_patches, n_features) where n_features=n_hid is the number of features per patch
        # self.output_decoder = MLP(nhid, nout, nlayer=2, with_final_activation=False)
        
        # New 
        self.criterion = torch.nn.L1Loss()  # MAE
        self.cfg = cfg
        self.num_timesteps = 12  # FIXME: make this dynamic
        self.horizon = 12
        self.n_nodes = 207
        self.transformer_encoder = MLPMixerTemporal(nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=self.n_patches, num_timesteps=self.num_timesteps)
        self.readout = Readout(nhid, self.n_patches, self.num_timesteps, self.n_nodes, self.horizon)

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
        for i in range(data.features.shape[1]):
            # For each time step, encode the features
            x = self.input_encoder(data.features[:, i].unsqueeze(-1))
            # Node PE (positional encoding, either random walk or Laplacian eigenvectors, canonical choices for graphs)
            if self.use_rw:
                x += self.rw_encoder(data.rw_pos_enc)
            if self.use_lap:
                x += self.lap_encoder(data.lap_pos_enc)
            edge_attr = data.edge_attr
            if edge_attr is None:
                edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
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
            mixer_x = self.reshape(subgraph_x)
            all_mixer_x[0, i, :, :] = mixer_x  # TODO: Check if mixer_x does indeed have batch_size (first dim) = 0


        # MLPMixer  # TURN INTO MULTI-HEAD REGRESSION (invariant to permutation of patches, check HDTTS paper for this)
        mixer_x = self.transformer_encoder(all_mixer_x, None, None)

        # Decoding / Readout

        out = self.readout(rearrange(mixer_x, 'b t p f -> b (t p f)'))  # (batch_size, n_timesteps, n_patches, n_features) -> (batch_size, n_nodes*n_timesteps)
        out = rearrange(out, 'b (n_nodes horizon) -> b n_nodes horizon', n_nodes=self.n_nodes)
        # # Global Average Pooling
        # x = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / data.mask.sum(1, keepdim=True)

        # # Readout
        # out = self.output_decoder(x).squeeze()
        return out

    def training_step(self, batch, batch_idx):
        """
        Batch is an instance of CustomTemporalData
        """
        if self.use_lap:
            batch_pos_enc = batch.lap_pos_enc
            sign_flip = (2 * (torch.rand(batch_pos_enc.size(1)) >= 0.5) - 1).float()  # array of +-1
            batch.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)

        out = self.forward(batch)  # (batch_size, n_nodes*n_timesteps)
        targets = batch.targets.unsqueeze(0)
        loss = self.criterion(out, targets)
        self.log_dict({'train_loss': loss}, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        targets = batch.targets.unsqueeze(0)
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
        
