import sys
import os
import torch
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.dataset import create_dataloaders
from model.model import GMMModel


def single_forward_backward():
    batch_size = 2

    print('Loading config...')
    cfg = OmegaConf.load('src/TGMM-1/train/config.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.load('src/TGMM-1/train/metrla.yaml'))
    
    # Override config values
    cfg.train.batch_size = 2
    cfg.dataset.max_len = 10
    cfg.dataset.train_size = 1
    cfg.dataset.val_size = 0

    print('Creating dataloader...')
    train_loader, val_loader, _, topo_data, metadata = create_dataloaders(cfg, raw_data_dir='/mnt/cephfs/store/gr-mc2473/lc865/workspace/GNN/data', num_workers=1)

    print('Initialising model...')
    model = GMMModel(cfg, topo_data, metadata)

    print('Getting batch...')
    for batch in train_loader:
        x, y = batch
        print(f'{x.shape = }')
        break

    print('Running forward pass...')
    y_pred = model.forward(x)
    loss = model.criterion(y_pred, y)
    metrics = model.calc_metrics(y_pred, y, key_prefix='train')
    metrics.update({'train/loss': loss})

    print('Running backward pass...')
    loss.backward()
    
    print('Checking gradients...')
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f'Warning: Found parameter with no gradient: {name}')
    
    print('Stepping optimizer...')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
    optimizer.step()
    print('All good!')
    
    
if __name__ == '__main__':
    # check_dataloader()
    single_forward_backward()