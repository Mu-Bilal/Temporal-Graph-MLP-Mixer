import sys
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dataset import create_dataset
from model.model import GMMModel


def create_model(cfg):  # TODO: Put into GMMModel
    assert cfg.dataset == 'ZINC'
    nfeat_node = 28
    nfeat_edge = 4
    nout = 1  # regression

    assert cfg.metis.n_patches > 0
    return GMMModel(nfeat_node=nfeat_node,
                    nfeat_edge=nfeat_edge,
                    nhid=cfg.model.hidden_size,
                    nout=nout,
                    nlayer_gnn=cfg.model.nlayer_gnn,
                    nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
                    gnn_type=cfg.model.gnn_type,
                    rw_dim=cfg.pos_enc.rw_dim,
                    lap_dim=cfg.pos_enc.lap_dim,
                    pooling=cfg.model.pool,
                    dropout=cfg.train.dropout,
                    mlpmixer_dropout=cfg.train.mlpmixer_dropout,
                    n_patches=cfg.metis.n_patches,
                    patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                    cfg=cfg)


def train_model(cfg):
    # Create dataloaders
    train_dataset, val_dataset, test_dataset = create_dataset(cfg)
    train_loader = DataLoader(train_dataset, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
    
    val_loader = DataLoader(val_dataset, cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)
    # test_loader = DataLoader(test_dataset, cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    # Create model
    model = create_model(cfg)

    # Set up logging
    logger = WandbLogger(
        save_dir='./logs',
        project='GMM-1',
        entity='Temporal-GMM'
    )
    
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        deterministic=True if cfg.seed else False
    )
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

if __name__ == '__main__':
    cfg = OmegaConf.load('/Users/luis/Desktop/ETH/Courses/AS24-DL/Project/Temporal-Graph-MLP-Mixer/src/GMM/model/config.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.load('/Users/luis/Desktop/ETH/Courses/AS24-DL/Project/Temporal-Graph-MLP-Mixer/src/GMM/train/zinc.yaml'))
    train_model(cfg)
