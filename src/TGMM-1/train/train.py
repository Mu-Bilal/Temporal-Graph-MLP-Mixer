import sys
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.dataset import create_dataloaders
from model.model import GMMModel


def train_model(cfg):
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    
    # Create model
    model = GMMModel(cfg, train_loader.dataset[0])

    # Set up logging
    logger = WandbLogger(save_dir='logs', project='GMM-1', entity='Temporal-GMM')
    logger.log_hyperparams(OmegaConf.to_container(cfg))

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='{epoch}',
        save_top_k=0,
        monitor=cfg.train.monitor,
        mode='min',
        save_last=True,
        save_weights_only=True
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        deterministic=True if cfg.seed else False,
        log_every_n_steps=min(50, len(train_loader)),
        gradient_clip_val=5.0,
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

if __name__ == '__main__':
    cfg = OmegaConf.load('/Users/luis/Desktop/ETH/Courses/AS24-DL/Project/Temporal-Graph-MLP-Mixer/src/TGMM-1/train/config.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.load('/Users/luis/Desktop/ETH/Courses/AS24-DL/Project/Temporal-Graph-MLP-Mixer/src/TGMM-1/train/metrla.yaml'))
    assert cfg.dataset == 'METRLA'
    train_model(cfg)
