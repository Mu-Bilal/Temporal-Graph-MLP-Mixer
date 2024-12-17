import sys
import torch
from sklearn.model_selection import StratifiedKFold

sys.path.append('/Users/luis/Desktop/ETH/Courses/AS24-DL/Project/Temporal-Graph-MLP-Mixer/src/GMM-new')
from util.dataset import create_dataset
from model.config import cfg, update_cfg

cfg.merge_from_file('/Users/luis/Desktop/ETH/Courses/AS24-DL/Project/Temporal-Graph-MLP-Mixer/src/GMM-new/train/zinc.yaml')
cfg = update_cfg(cfg)

train_dataset, val_dataset, test_dataset = create_dataset(cfg)
