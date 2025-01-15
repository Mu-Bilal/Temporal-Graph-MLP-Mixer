import os
from omegaconf import OmegaConf

def load_config(configs_dir: str, dataset_name: str):
    failure_mode_path = os.path.join(configs_dir, 'mode')

    cfg = OmegaConf.merge(
        OmegaConf.load(os.path.join(configs_dir, 'config.yaml')),   # Used for default values
        OmegaConf.load(os.path.join(configs_dir, f'{dataset_name}.yaml'))
    )


    failure_mode = '_'.join(cfg.raw_data.name.split('_')[1:])
    if failure_mode == '':
        raise ValueError(f"Failure mode not found for dataset {dataset_name}. Please add this to dataset name like so: `datasetName_failureMode`.")

    if cfg.raw_data.name.split('_')[0] == 'mso':  # Synthetic dataset, treat seperately.
        if failure_mode == 'normal':
            raise ValueError(f"Normal mode is not allowed for synthetic dataset {dataset_name}.")
        failure_cfg = OmegaConf.load(os.path.join(failure_mode_path, f'mso_{failure_mode}.yaml'))
        cfg.raw_data.hparams = OmegaConf.merge(cfg.raw_data.hparams, failure_cfg)
    else:
        failure_cfg = OmegaConf.load(os.path.join(failure_mode_path, f'{failure_mode}.yaml'))
        cfg.raw_data.mode = OmegaConf.merge(cfg.raw_data.mode, failure_cfg)

    return cfg
