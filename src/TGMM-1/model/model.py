from lightning.pytorch import LightningModule
import torch.nn as nn
import torch
from torch_scatter import scatter
from einops import rearrange
from model.mlp_mixer import MLPMixerTemporal
from typing import List
from model.elements import MLP
from model.gnn import GNN
from model.readouts import SingleNodeReadout
from model.dataset import StaticGraphTopologyData
from typing import Dict, Union

class GMMModel(LightningModule):

    def __init__(self, cfg, topo_data, metadata: Dict):
        super().__init__()
        assert cfg.metis.n_patches > 0
        self.cfg = cfg
        self.topo_data: StaticGraphTopologyData = topo_data
        self.criterion = torch.nn.L1Loss()  # MAE
        
        # Logging
        self.train_step_metrics_buffer = []
        self.valid_step_metrics_buffer = []
        self.test_step_metrics_buffer = []
        
        # Dataset unnormalisation (for reporting true-scale metrics)
        self.unnormalise = ('norm_mean' in metadata.keys() and 'norm_std' in metadata.keys())
        self.metadata = metadata

        # Shortcuts 
        self.pooling = cfg.model.pool

        self.input_encoder_patch = nn.Linear(1, cfg.model.nfeatures_patch)
        self.input_encoder_node = nn.Linear(1, cfg.model.nfeatures_node)
        self.edge_encoder = nn.Linear(1, cfg.model.nfeatures_patch)

        self.gnns = nn.ModuleList([
            GNN(nin=cfg.model.nfeatures_patch, nout=cfg.model.nfeatures_patch, nlayer_gnn=1, gnn_type=cfg.model.gnn_type, bn=True, dropout=cfg.train.dropout_gnn, res=True)
            for _ in range(cfg.model.nlayer_gnn)
        ])
        self.U = nn.ModuleList([
            MLP(cfg.model.nfeatures_patch, cfg.model.nfeatures_patch, nlayer=1, with_final_activation=True)
            for _ in range(cfg.model.nlayer_gnn-1)
        ])

        self.mixer_patch = MLPMixerTemporal(
            n_features=cfg.model.nfeatures_patch,
            n_spatial=cfg.metis.n_patches,
            n_timesteps=cfg.dataset.window,
            n_layer=cfg.model.nlayer_patch_mixer,
            dropout=cfg.train.dropout_patch_mixer,
            with_final_norm=True
        )
        self.mixer_node = MLPMixerTemporal(
            n_features=cfg.model.nfeatures_node,
            n_spatial=self.topo_data.num_nodes,
            n_timesteps=cfg.dataset.window,
            n_layer=cfg.model.nlayer_node_mixer,  
            dropout=cfg.train.dropout_node_mixer,
            with_final_norm=True
        )
        self.readout = SingleNodeReadout(
            cfg.model.nfeatures_patch, 
            cfg.model.nfeatures_node, 
            cfg.dataset.window, 
            cfg.dataset.horizon, 
            self.topo_data, 
            n_layers=cfg.model.nlayer_readout, 
            dropout=cfg.train.dropout_readout
        )

        # Test run: Simple LSTM per node
        # nhid = 256
        # self.lstm = nn.LSTM(input_size=1, hidden_size=nhid, num_layers=4, batch_first=True)
        # self.readout2 = MLP(nhid, cfg.dataset.horizon, nlayer=3, with_final_activation=False)

    def setup(self, stage):
        self._move_tensors_to_device(self.topo_data)
        self._move_tensors_to_device(self.readout)
        
    def _move_tensors_to_device(self, obj):
        for attr_name in dir(obj):
            if not attr_name.startswith('__'):  # Skip built-in attributes
                attr = getattr(obj, attr_name)
                if isinstance(attr, torch.Tensor):
                    setattr(obj, attr_name, attr.to(self.device))
    
    def forward(self, x):
        x_raw = x        
        x = rearrange(self.input_encoder_patch(x_raw.unsqueeze(-1)), 'B n t f -> B t n f')
        nodes_x = rearrange(self.input_encoder_node(x_raw.unsqueeze(-1)), 'B n t f -> B t n f')
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
        patch_x = scatter(x, self.topo_data.subgraphs_batch, dim=-2, reduce=self.pooling)  # (n_timesteps, n_patches, n_features)

        # Patch Mixer
        patch_x = self.mixer_patch(patch_x)

        # Node Mixer
        nodes_x = self.mixer_node(nodes_x)

        # Decoding / Readout
        out = self.readout(patch_x, nodes_x)

        # Test run: Simple LSTM per node
        # batch_size = x.shape[0]
        # x = rearrange(x, 'B n t -> (B n) t').unsqueeze(-1)
        # out, _ = self.lstm(x)  # out shape: (batch, seq_len, hidden_size)
        # out = self.readout2(out[:, -1, :])  # Transform to predict next 12 timesteps
        # out = rearrange(out, '(B n) h -> B n h', B=batch_size)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)  # (batch_size, n_nodes*n_timesteps)
        loss = self.criterion(y_pred, y)
        metrics = self.calc_metrics(y_pred, y, key_prefix='train', ignore_missing=True, missing_val=0)
        metrics.update({'train/loss': loss})
        metrics.update({'train/lr': self.optimizers().param_groups[0]['lr']})
        return {'loss': loss, 'step_metrics': metrics}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        metrics = self.calc_metrics(y_pred, y, key_prefix='valid', ignore_missing=True, missing_val=0)
        metrics.update({'valid/loss': loss})
        metrics.update({'valid/lr': self.optimizers().param_groups[0]['lr']})
        return {'loss': loss, 'step_metrics': metrics}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        metrics = self.calc_metrics(y_pred, y, key_prefix='test', ignore_missing=True, missing_val=0)
        metrics.update({'test/loss': loss})
        return {'loss': loss, 'step_metrics': metrics}
    
    def calc_metrics(self, pred: torch.Tensor, targets: torch.Tensor, key_prefix='valid', ignore_missing=False, missing_val=0):
        if self.unnormalise:
            pred = pred * self.metadata['norm_std'] + self.metadata['norm_mean']
            targets = targets * self.metadata['norm_std'] + self.metadata['norm_mean']

        def calc_metrics_for_horizon(pred: torch.Tensor, targets: torch.Tensor, horizon: Union[int, str]):
            if isinstance(horizon, int):
                pred = pred[..., horizon-1]
                targets = targets[..., horizon-1]
            elif horizon == 'all':
                pass
            else:
                raise ValueError(f'Invalid horizon: {horizon}')
            
            if ignore_missing:
                missing_mask = (targets == missing_val)
                pred = pred[~missing_mask]
                targets = targets[~missing_mask]

            mae = torch.mean(torch.abs(pred - targets))
            rmse = torch.sqrt(torch.mean((pred - targets)**2))
            mape = 100*torch.mean(torch.abs((pred - targets) / targets))
            
            return {
                f'{key_prefix}/mae-{horizon}': mae,
                f'{key_prefix}/rmse-{horizon}': rmse, 
                f'{key_prefix}/mape-{horizon}': mape,
            }
        metrics = {}
        for horizon in [3, 6, 12, 'all']:
            metrics.update(calc_metrics_for_horizon(pred, targets, horizon))
        return metrics
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.wd)
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
        
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if torch.isnan(outputs['loss']):
            raise ValueError("NaN detected in training loss")

        # Store and log metrics every N steps
        self.train_step_metrics_buffer.append(outputs['step_metrics'])
        
        if batch_idx % self.trainer.log_every_n_steps == 0:
            self._log_from_buffer(self.train_step_metrics_buffer)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        # Store and log metrics every N steps
        self.valid_step_metrics_buffer.append(outputs['step_metrics'])
        
        if batch_idx % self.trainer.log_every_n_steps == 0:
            self._log_from_buffer(self.valid_step_metrics_buffer)

    def on_test_batch_end(self, outputs, batch, batch_idx):
        # Store and log metrics every N steps
        self.test_step_metrics_buffer.append(outputs['step_metrics'])
        
        if batch_idx % self.trainer.log_every_n_steps == 0:
            self._log_from_buffer(self.test_step_metrics_buffer)
    
    def on_validation_epoch_end(self):
        self._log_from_buffer(self.valid_step_metrics_buffer)

    def on_test_epoch_end(self):
        self._log_from_buffer(self.test_step_metrics_buffer)

    def on_train_epoch_end(self) -> None:
        self._log_from_buffer(self.train_step_metrics_buffer)
        
    def _log_from_buffer(self, buffer: List):
        if len(buffer) == 0:
            return

        # Average the metrics over the last N steps
        avg_metrics = {k: sum(step_dict[k] for step_dict in buffer) / len(buffer) for k in buffer[0]}

        # Log the averaged metrics and clear the buffer
        self.log_dict(avg_metrics, sync_dist=True)
        buffer.clear()
