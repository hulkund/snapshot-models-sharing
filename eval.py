import yaml
from train import load_model, create_dataloader
import torch
import numpy as np # NOTE: since we're using these functions across files, it could make sense to put them in e.g. a "util.py" script.
# import wandb
from dataset import CTDataset
import argparse
import pandas as pd
import pdb 
import sys

# split = 'test'

# load config
parser = argparse.ArgumentParser(description='evaluate snapshot experiments')
parser.add_argument('--dataset_split', help='which test to evaluate on', default='test')
parser.add_argument('--config', help='Path to config file', default='configs/cfg_all.yaml')
args = parser.parse_args()
cfg = yaml.safe_load(open(args.config, 'r'))
print(f'Using config "{cfg}"')
torch.manual_seed(42)

def dict_flip(original_dict):
    flipped_dict = {value: key for key, value in original_dict.items()}
    return flipped_dict

def evaluate(cfg):
    dl_train=create_dataloader(cfg,split='train')
    train_mapping=dict_flip(dl_train.dataset.mapping)
    if args.dataset_split == 'test':
        dl_test=create_dataloader(cfg,split="test")
        test_mapping=dict_flip(dl_test.dataset.mapping)
    else:
        dl_test=create_dataloader(cfg,split="val",mapping=dl_train.dataset.mapping)
        test_mapping=train_mapping
    # DIST VERSION
    if weighting:
        weighting=dl_test.dataset.labels['question__species'].value_counts(normalize=True)
        for species in dl_train.dataset.labels['question__species'].unique():
            if species not in weighting.index:
                weighting.loc[species]=0
        weighting = weighting.rename(index=dict_flip(train_mapping))
        weighting = weighting[pd.to_numeric(weighting.index, errors='coerce').notna()]
        sorted_series = weighting.sort_index()
        weights_tensor = torch.tensor(sorted_series.values, device='cuda')
        combined_mapping = {key: i for i, key in enumerate(set(train_mapping) | set(test_mapping))}
    
    print("created dataloader")
    # load model
    model, current_epoch = load_model(cfg, num_classes=dl_train.dataset.num_classes, state='eval') 
    device = cfg['device']
    model.to(device)
    model.eval() 

    predictions = []
    ground_truth = []
    with torch.no_grad():
        for data, labels in dl_test:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            weighted_output = outputs * weights_tensor
            _, predicted = torch.max(weighted_output, 1)
            predictions.extend(predicted.cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())
    df = pd.DataFrame()
    df['predictions'], df['labels'] = predictions, ground_truth
    # mapping the label numbers to their original strings
    df['predictions_str_orig'], df['labels_str_orig'] = df['predictions'].map(train_mapping), df['labels'].map(test_mapping)
    df.to_csv('predictions/{}_bayesian.csv'.format(cfg['exp_name']))
    unique_labels, label_counts = np.unique(predictions, return_counts=True)
    total_correct = np.sum(df['predictions_str_orig']==df['labels_str_orig'])
    print("Accuracy: ", total_correct/len(predictions))

if __name__ == '__main__':
    print("before")
    print(torch.cuda.is_available())
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'
        sys.exit()
    evaluate(cfg)
