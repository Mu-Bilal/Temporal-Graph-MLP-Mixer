import sys
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.dataset import create_dataloaders
from model.model import GMMModel


def train_model(cfg):
    # Create dataloaders
    train_loader, val_loader, _, topo_data = create_dataloaders(cfg, raw_data_dir='/mnt/cephfs/store/gr-mc2473/lc865/workspace/GNN/data')
    
    # Create model
    model = GMMModel(cfg, topo_data)

    # Set up logging
    logging_path = '/mnt/cephfs/store/gr-mc2473/lc865/workspace/GNN/wandb_logs'
    os.makedirs(logging_path, exist_ok=True)
    logger = WandbLogger(save_dir=logging_path, project='GMM-1', entity='Temporal-GMM')
    logger.log_hyperparams(OmegaConf.to_container(cfg))

    checkpoint_callback = ModelCheckpoint(
        dirpath='/mnt/cephfs/store/gr-mc2473/lc865/workspace/GNN/checkpoints',
        filename='{epoch}',
        save_top_k=0,
        monitor=cfg.train.monitor,
        mode='min',
        save_last=True,
        save_weights_only=True
    )
    
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator='gpu',  # gpu' if torch.cuda.is_available() else 
        devices=1,
        logger=logger,
        deterministic=True if cfg.seed else False,
        log_every_n_steps=min(50, len(train_loader)),
        gradient_clip_val=5.0,
        callbacks=[checkpoint_callback],
    )
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    wandb.finish()

if __name__ == '__main__':
    cfg = OmegaConf.load('/home/lc865/workspace/DL-GNNs/Temporal-Graph-MLP-Mixer/src/TGMM-1/train/config.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.load('/home/lc865/workspace/DL-GNNs/Temporal-Graph-MLP-Mixer/src/TGMM-1/train/metrla.yaml'))
    train_model(cfg)
