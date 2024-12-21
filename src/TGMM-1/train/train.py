import sys
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dataset import create_dataloaders
from model.model import GMMModel

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def create_model(cfg, data_example):  # TODO: Put into GMMModel
    assert cfg.dataset == 'METRLA'
    nfeat_node = 1
    nfeat_edge = 1
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
                    cfg=cfg,
                    data_example=data_example)


def train_model(cfg):
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg, 
        'METRLA', 
        max_len=100, 
        train_size=0.9, 
        val_size=0.1
    )
    
    # Create model
    model = create_model(cfg, train_loader.dataset[0])

    # Set up logging
    # logger = None
    logger = WandbLogger(
        save_dir='./logs',#
        project='GMM-1',
        entity='Temporal-GMM'
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        deterministic=True if cfg.seed else False,
        log_every_n_steps=min(50, len(train_loader)),
        gradient_clip_val=5.0
    )
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

if __name__ == '__main__':
    cfg = OmegaConf.load('/Users/luis/Desktop/ETH/Courses/AS24-DL/Project/Temporal-Graph-MLP-Mixer/src/TGMM-1/model/config.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.load('/Users/luis/Desktop/ETH/Courses/AS24-DL/Project/Temporal-Graph-MLP-Mixer/src/TGMM-1/train/metrla.yaml'))
    train_model(cfg)
