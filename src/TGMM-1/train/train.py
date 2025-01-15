import sys
import os
import torch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.dataset import create_dataloaders
from model.model import GMMModel
from configs.utils import load_config

class MiscCallback(Callback):
    """
    Custom callback to access the WandB run data. This cannot be accessed during setup as Logger is initialised only when trainer.fit() is called.

    From Docs:
    trainer.logger.experiment: Actual wandb object. To use wandb features in your :class:`~lightning.pytorch.core.LightningModule` do the
    following. self.logger.experiment.some_wandb_function()

    # Only available in rank0 process, others have _DummyExperiment
    """
    def on_train_start(self, trainer, pl_module):
        if isinstance(trainer.logger, WandbLogger) and trainer.is_global_zero:
            # Dynamically set the checkpoint directory in ModelCheckpoint
            print(f"Checkpoints will be saved in: {trainer.logger.experiment.dir}")
            trainer.checkpoint_callback.dirpath = trainer.logger.experiment.dir

        print(f'Node rank: {trainer.node_rank}, Global rank: {trainer.global_rank}, Local rank: {trainer.local_rank}')
        print(f'Trainer strategy: {trainer.strategy}')

    def on_train_end(self, trainer, pl_module):
        if isinstance(trainer.logger, WandbLogger) and trainer.is_global_zero:
            # print(f'Files in wandb dir: {os.listdir(trainer.logger.experiment.dir)}')
            # FIXME: Quickfix to make sure last checkpoint is saved.
            trainer.logger.experiment.save(os.path.join(trainer.logger.experiment.dir, 'last.ckpt'),
                                           base_path=trainer.logger.experiment.dir)


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
    train_loader, val_loader, test_loader, topo_data, metadata = create_dataloaders(cfg, raw_data_dir='/mnt/cephfs/store/gr-mc2473/lc865/workspace/GNN/data', num_workers=num_workers)
    
    # Create model
    model = GMMModel(cfg, topo_data, metadata)

    # Set up logging
    logging_path = '/mnt/cephfs/store/gr-mc2473/lc865/workspace/GNN/wandb_logs'
    os.makedirs(logging_path, exist_ok=True)
    logger = WandbLogger(save_dir=logging_path, project=cfg.project, entity='Temporal-GMM')
    logger.log_hyperparams(OmegaConf.to_container(cfg))

    checkpoint_callback = ModelCheckpoint(
        #  dirpath='/mnt/cephfs/store/gr-mc2473/lc865/workspace/GNN/checkpoints', Supplied later
        filename='{epoch}',
        save_top_k=1,
        monitor=cfg.train.monitor,
        mode='min',
        save_last=True,
        save_weights_only=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor=cfg.train.monitor,
        mode='min',
        patience=cfg.train.early_stop_patience,
        verbose=True
    )
    
    torch.set_float32_matmul_precision('medium')
    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        logger=logger,
        deterministic=True if cfg.seed else False,
        log_every_n_steps=min(50, len(train_loader)),
        gradient_clip_val=5.0,
        callbacks=[checkpoint_callback, early_stop_callback, MiscCallback()],
        strategy=strategy,
        devices=devices
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    if trainer.is_global_zero:
        wandb.finish()

if __name__ == '__main__':
    dataset = 'la'
    cfg = load_config(configs_dir='/home/lc865/workspace/DL-GNNs/Temporal-Graph-MLP-Mixer/src/TGMM-1/train/configs', dataset_name=dataset)
    train_model(cfg)
