import os
from omegaconf import OmegaConf

def load_config(configs_dir: str, dataset_name: str):
    failure_mode_path = os.path.join(configs_dir, 'mode')

    cfg = OmegaConf.merge(
        OmegaConf.load(os.path.join(configs_dir, 'config.yaml')),   # Used for default values
        OmegaConf.load(os.path.join(configs_dir, f'{dataset_name}.yaml'))
    )

    failure_mode = '_'.join(cfg.raw_data.name.split('_')[1:])

    failure_cfg = OmegaConf.load(os.path.join(failure_mode_path, failure_mode + '.yaml'))
    cfg.raw_data.mode = OmegaConf.merge(cfg.raw_data.mode, failure_cfg)

    return cfg
