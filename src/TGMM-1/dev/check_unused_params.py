import sys
import os
import torch
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.dataset import create_dataloaders
from model.model import GMMModel
from train.configs.utils import load_config


def single_forward_backward():
    print('Loading config...')
    cfg = load_config(configs_dir='src/TGMM-1/train/configs', dataset_name='la')
    
    # Override config values
    cfg.train.batch_size = 2
    cfg.dataset.max_len = 10
    cfg.dataset.train_size = 1
    cfg.dataset.val_size = 0

    print('Creating dataloader...')
    train_loader, val_loader, test_loader, topo_data, metadata = create_dataloaders(cfg, raw_data_dir='/mnt/cephfs/store/gr-mc2473/lc865/workspace/GNN/data', num_workers=1)

    print('Initialising model...')
    model = GMMModel(cfg, topo_data, metadata)

    print('Getting batch...')
    for batch in train_loader:
        x, y, mask_x, mask_y = batch
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