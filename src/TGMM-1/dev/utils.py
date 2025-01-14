import wandb
import yaml
from lightning.pytorch.utilities.model_summary import ModelSummary
from typing import Dict

import sys
import os
from omegaconf import OmegaConf
sys.path.append(os.path.dirname(os.path.abspath('.')))
from model.dataset import create_dataloaders
from model.model import GMMModel

def dict_lit2num(d: Dict, verbose=False):
    """
    Convert all literal values in a dictionary to numbers (int or float). Needed as WandB stores all values as strings?
    """
    def _convert(x):
        if isinstance(x, dict):
            return {k: _convert(v) for k, v in x.items()}
        else:  # Leaf node
            try:
                tmp = float(x)
                if tmp.is_integer():
                    tmp = int(tmp)
            except:
                tmp = x
            if verbose:
                print(f'Leaf node:{x} -> {tmp}; type: {type(tmp)}')
            return tmp
    return _convert(d)

def load_wandb_model(run: str, name:str = 'last.ckpt', device='cpu', wandb_cache_path='/mnt/cephfs/store/gr-mc2473/lc865/workspace/GNN/wandb_logs', project='GMM-1', replace=True, raw_data_dir='/mnt/cephfs/store/gr-mc2473/lc865/workspace/GNN/data'):
    # Load model checkpoint from wandb
    with wandb.restore(name, run_path=f"Temporal-GMM/{project}/runs/{run}", root=wandb_cache_path, replace=replace) as io:
        checkpoint_path = io.name

    # Read the model parameters from the WandB config.yaml file  
    with wandb.restore('config.yaml', run_path=f"Temporal-GMM/{project}/runs/{run}", root=wandb_cache_path, replace=True) as config_file:
        cfg_dict = yaml.safe_load(config_file)

    # Convert wandb config to OmegaConf
    cfg = OmegaConf.create({k: dict_lit2num(v['value']) for k, v in list(cfg_dict.items()) if k not in ['wandb_version', '_wandb']})
    
    # Create dataloaders
    train_loader, val_loader, test_loader, topo_data, metadata = create_dataloaders(cfg, raw_data_dir=raw_data_dir)
    
    # Load the Lightning checkpoint
    model = GMMModel.load_from_checkpoint(checkpoint_path, cfg=cfg, topo_data=topo_data, metadata=metadata, map_location=device)
    model.setup(stage='test')
    model.eval()

    summary = ModelSummary(model, max_depth=1)
    print(f'Imported model from run "{run}".')
    print(summary)

    return model, cfg, train_loader, val_loader, test_loader, topo_data, metadata