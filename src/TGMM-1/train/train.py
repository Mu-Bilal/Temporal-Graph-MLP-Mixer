import sys
import os
import torch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import SingleDeviceStrategy, DDPStrategy
from omegaconf import OmegaConf
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.dataset import create_dataloaders
from model.model import GMMModel


def train_model(cfg):
    # Dataloader and device setup
    if 'SLURM_JOB_ID' in os.environ:  # Running distributed via SLURM
        num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
        devices = int(os.environ['SLURM_NNODES']) * int(os.environ['SLURM_NTASKS_PER_NODE'])
        if devices > 1:
            strategy = DDPStrategy(find_unused_parameters=False, process_group_backend="nccl")
        else:
            strategy = "auto"  # SingleDeviceStrategy()
    else:
        num_workers = min(os.cpu_count(), 8)
        strategy = "auto"
        devices = 1  # Default to 1 if not running on SLURM or GPU count not specified
    # strategy, devices = SingleDeviceStrategy(), 1
    print(f"Using strategy: {strategy} and {devices} device(s)")
    
    # Create dataloaders
    print(f"Creating dataloaders with {num_workers} workers")
    train_loader, val_loader, _, topo_data = create_dataloaders(cfg, raw_data_dir='/mnt/cephfs/store/gr-mc2473/lc865/workspace/GNN/data', num_workers=num_workers)
    
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
    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        logger=logger,
        deterministic=True if cfg.seed else False,
        log_every_n_steps=min(50, len(train_loader)),
        gradient_clip_val=5.0,
        callbacks=[checkpoint_callback],
        strategy=strategy,
        devices=devices
    )
    trainer.fit(model, train_loader, val_loader)
    
    if trainer.is_global_zero:
        wandb.finish()

if __name__ == '__main__':
    cfg = OmegaConf.load('/home/lc865/workspace/DL-GNNs/Temporal-Graph-MLP-Mixer/src/TGMM-1/train/config.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.load('/home/lc865/workspace/DL-GNNs/Temporal-Graph-MLP-Mixer/src/TGMM-1/train/metrla.yaml'))
    train_model(cfg)
