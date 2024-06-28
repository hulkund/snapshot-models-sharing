'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    2022 Benjamin Kellenberger
'''

import os
import argparse
import yaml
import glob
import math
from tqdm import trange
from datetime import datetime
import sys

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, random_split 
from torch.optim import SGD 
from dataset import create_dataloader
from util import init_seed
from model import CustomResNet18
import pdb

parser = argparse.ArgumentParser(description='Train deep learning model.')
parser.add_argument('--config', help='Path to config file', default='configs/cfg_all.yaml')
args = parser.parse_args()
torch.manual_seed(42)

def save_model(cfg, epoch, model, stats):
    exp_name = cfg['exp_name']
    learning_rate = cfg['learning_rate']
    batch_size=cfg['batch_size']
    model_state_path = f'model_states/model_states_{exp_name}'
    os.makedirs(model_state_path, exist_ok=True)
    state_dict = {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "stats": stats
                }
    torch.save(state_dict, open(f'{model_state_path}/{epoch}.pt', 'wb'))
    cfpath = f'{model_state_path}/config.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)


def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer


def train(cfg, dataLoader, model, epoch, n_steps_per_epoch, optimizer):
    device = cfg['device']
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_total, oa_total = 0.0, 0.0                      
    example_ct = 0
    progressBar = trange(len(dataLoader))
    for idx, (data, labels) in enumerate(dataLoader):       
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()                       
        pred_label = torch.argmax(prediction, dim=1)    
        oa = torch.mean((pred_label == labels).float()) 
        oa_total += oa.item()
        example_ct += len(data)
        metrics = {"train/train_loss": loss, 
                   "train/epoch": (idx + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch, 
                   "train/example_ct": example_ct}
        # WANB LOGGING
        # if idx + 1 < n_steps_per_epoch:
            # wandb.log(metrics)
        progressBar.set_description(
            '[Train ] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            ))
        progressBar.update(1)
    progressBar.close()
    loss_total /= len(dataLoader)           
    oa_total /= len(dataLoader)
    return loss_total, oa_total, metrics

def validate(cfg, dataLoader, model):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    model.eval()
    criterion = nn.CrossEntropyLoss()   
    loss_total, oa_total = 0.0, 0.0     
    progressBar = trange(len(dataLoader))
    with torch.no_grad():               
        for idx, (data, labels) in enumerate(dataLoader):
            # put data and labels on device
            data, labels = data.to(device), labels.to(device)
            # forward pass
            prediction = model(data)
            # loss
            # print(type(prediction), type(labels))
            # prediction = prediction.to(torch.float32)
            # labels = labels.to(torch.int64)
            loss = criterion(prediction, labels)
            # pdb.set_trace()
            loss_total += loss.item()
            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total

def load_model(cfg, num_classes, state='train'):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet18(num_classes)         # create an object instance of our CustomResNet18 class
    # load latest model state
    exp_name = cfg['exp_name']
    learning_rate = cfg['learning_rate']
    batch_size=cfg['batch_size']
    model_states_folder = f'model_states/model_states_{exp_name}'
    model_states = glob.glob(model_states_folder+'/*.pt')
    if len(model_states):
        # at least one save state found; get latest
        model_epochs = [int(m.replace(model_states_folder+"/",'').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)
        # load state dict and apply weights to model
        state = torch.load(open(f'{model_states_folder}/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])
        print("loading ", f'{model_states_folder}/{start_epoch}.pt')
    else:
        print('Starting new model')
        if state == 'eval':
            print("THIS IS BAD THERE IS NO MODEL ! ")
            return
        # no save state found; start anew
        start_epoch = 0
    return model_instance, start_epoch


def main():
    cfg = yaml.safe_load(open(args.config, 'r'))
    print("loaded config")
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    init_seed(cfg.get('seed', None))
    device = cfg['device']
    print(torch.cuda.is_available())
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'
        sys.exit()
    dl_train = create_dataloader(cfg,split="train")
    dl_val = create_dataloader(cfg,split="val", mapping=dl_train.dataset.mapping)
    print("created dataloaders")
    num_classes = dl_train.dataset.num_classes
    model, current_epoch = load_model(cfg, num_classes=num_classes)
    optim = setup_optimizer(cfg, model)
    best_val_loss = float('inf')
    n_steps_per_epoch = math.ceil(len(dl_train.dataset) / cfg['batch_size'])
    numEpochs = cfg['num_epochs']
    early_stop_counter = 0
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')
        loss_train, oa_train, metrics = train(cfg, dl_train, model, current_epoch, n_steps_per_epoch, optim)
        loss_val, oa_val = validate(cfg, dl_val, model)
        val_metrics = {"val/val_loss": loss_val, "val/val_accuracy": oa_val}
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            stats = {
                'loss_train': loss_train,
                'loss_val': loss_val,
                'oa_train': oa_train,
                'oa_val': oa_val
            }
            save_model(cfg, current_epoch, model, stats)
        else:
            early_stop_counter+=1
        patience = 2
        if early_stop_counter > patience-1:
            print(f'Epoch {current_epoch} was stopped after early stopping didnt improve after {patience} epochs')
            break


if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()
