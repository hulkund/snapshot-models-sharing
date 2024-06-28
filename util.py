'''
    Various utility functions used (possibly) across scripts.

    2022 Benjamin Kellenberger
'''

import random
import torch
from torch.backends import cudnn
from dataset import CTDataset
from model import CustomResNet18
import glob


def init_seed(seed):
    '''
        Initalizes the seed for all random number generators used. This is
        important to be able to reproduce results and experiment with different
        random setups of the same code and experiments.
    '''
    if seed is not None:
        random.seed(seed)
        # numpy.random.seed(seed)       # we don't use NumPy in this code, but you would want to set its random number generator seed, too
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = True
        cudnn.deterministic = True

DATASET_DICT= {
    'Snapshot_Kruger':
        {'train_csv':'Snapshot_Kruger_train.csv',
        'val_csv':'Snapshot_Kruger_val.csv',
        'all_csv':'Snapshot_Kruger_all.csv'},
    'Snapshot_Enonkishu':
        {'train_csv':'Snapshot_Eno_train.csv',
        'val_csv':'Snapshot_Eno_val.csv',
        'all_csv':'Snapshot_Eno_all.csv'},
    'Snapshot_Camdeboo':
        {'train_csv':'Snapshot_Camdeboo_train.csv',
        'val_csv':'Snapshot_Camdeboo_val.csv',
        'all_csv':'Snapshot_Camdebooc_all.csv'},
    'Snapshot_Karoo':
        {'train_csv':'Snapshot_Karoo_train.csv',
        'val_csv':'Snapshot_Karoo_val.csv',
        'all_csv':'Snapshot_Karoo_all.csv'},
    'Snapshot_Kgalagadi':
        {'train_csv':'Snapshot_Kga_train.csv',
        'val_csv':'Snapshot_Kga_val.csv',
        'all_csv': 'Snapshot_Kga_all.csv'},
    'Snapshot_Mountain_Zebra':
        {'train_csv':'Snapshot_Mountain_Zebra_train.csv',
        'val_csv':'Snapshot_Mountain_Zebra_val.csv',
        'all_csv':'Snapshot_Mountain_Zebra_all.csv'}
    # 'Snapshot_Serengeti':
    #     {'train_csv':'Snapshot_SerengetiS11.json'}
}

def snapshot_train_val_dataloader(cfg):
    dataset_list=cfg['train_val_dataset']
    val_dataset = dataset_file_list=[DATASET_DICT[dataset]['val_csv'] for dataset in dataset_list]
    train_dataset = get_multiple_datasets(data_root=cfg['data_root'], dataset_file_list=[DATASET_DICT[dataset]['train_csv'] for dataset in dataset_list])
    train_dataloader, val_dataloader = create_dataloader(cfg,train_dataset), create_dataloader(cfg, val_dataset)
    return train_dataloader, val_dataloader

def snapshot_test_dataloader(cfg):
    dataset_list=[DATASET_DICT[cfg['test_dataset']]['all_csv']]
    dataset_instance = get_multiple_datasets(cfg['data_root'], dataset_list) 
    return create_dataloader(cfg,dataset_instance)

def create_dataloader(cfg,dataset):
    dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers']
        )  
    return dataloader


