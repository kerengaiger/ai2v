# -*- coding: utf-8 -*-
import datetime
import pathlib
import pickle
import os

import numpy as np
import torch as t
import torch.nn as nn
from torch import FloatTensor as FT
from torch.optim import Adagrad, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train_utils import save_model, configure_weights, UserBatchIncrementDataset, set_random_seed
from dataset import generate_train_files
import models
import argparse
import optuna


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--data_cnfg', type=str, default='./config/ml-1m.json',
                        help="data config to generate train files")
    parser.add_argument('--save_dir', type=str, default='./output/', help="model directory path")
    parser.add_argument('--train', type=str, default='full_train.dat', help="train file name")
    parser.add_argument('--test', type=str, default='test.dat', help="test users file name")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    parser.add_argument('--device', type=str, default=0, help="cude device to use")
    parser.add_argument('--log_dir', type=str, default='tensorboard/logs/mylogdir', help="logs dir for tensorboard")
    parser.add_argument('--best_cnfg', type=str, default='best_cnfg.pkl', help="best cnfg of hyper params")
    parser.add_argument('--fine_tune', action='store_true', help="user a pre-trained model and fine-tune  ")
    parser.add_argument('--weights_init', type=str, default='model.pt',
                        help="model weights file to initiate the new model when fine_tune is True")

    return parser.parse_args()


def calc_loss_on_set(sgns, valid_dl, pad_idx):
    pbar = tqdm(valid_dl)
    valid_losses = []

    for batch_u_ids, batch_titems, batch_citems in pbar:
        batch_u_ids, batch_titems, batch_citems = batch_u_ids.to(sgns.device), batch_titems.to(sgns.device), \
                                                  batch_citems.to(sgns.device)

        loss = sgns(batch_u_ids, batch_titems, batch_citems)
        valid_losses.append(loss.item())

    return np.array(valid_losses).mean()


def train(cnfg, train_file, valid_dl=None, trial=None):
    idx2item = pickle.load(pathlib.Path(cnfg['data_dir'], 'idx2item.dat').open('rb'))
    item2idx = pickle.load(pathlib.Path(cnfg['data_dir'], 'item2idx.dat').open('rb'))
    user2idx = pickle.load(pathlib.Path(cnfg['data_dir'], 'user2idx.dat').open('rb'))

    weights = configure_weights(cnfg, idx2item)
    vocab_size = len(idx2item)

    model_base_c = getattr(models, cnfg['model'])
    sgns_c = getattr(models, 'sgns_' + cnfg['model'])

    cnfg['padding_idx'] = item2idx['pad']
    cnfg['vocab_size'] = vocab_size
    if cnfg['cuda']:
        device = 'cuda:' + str(cnfg['device'])
    else:
        device = 'cpu'
    cnfg['device'] = device
    cnfg['num_users'] = len(user2idx)

    model_init = {k: cnfg[k] for k in getattr(models, cnfg['model'] + '_cnfg_keys')}
    model = model_base_c(**model_init)
    sgns = sgns_c(base_model=model, vocab_size=vocab_size, n_negs=cnfg['n_negs'], weights=weights,
                  loss_method=cnfg['loss_method'], device=device)

    # TODO: this is a hack - remove later
    if cnfg['fine_tune']:
        sgns = t.load(cnfg['weights_init'])
        sgns.device = device
        sgns.ai2v.device = device
        for param in sgns.parameters():
            param.requires_grad = False
        for l in sgns.ai2v.mha_layers:
            l.local_pos_bias = nn.Parameter(FT(cnfg['num_users'], sgns.ai2v.window_size).uniform_(
                -0.5 / sgns.ai2v.window_size, 0.5 / sgns.ai2v.window_size))
            l.local_pos_bias.requires_grad = True
            l.device = device
    sgns.to(device)
    optim = Adagrad(sgns.parameters(), lr=cnfg['lr'])
    scheduler = lr_scheduler.MultiStepLR(optim, milestones=[2, 4, 5, 6, 7, 8, 10, 12, 14, 16], gamma=0.5)

    log_dir = cnfg['log_dir'] + '/' + str(datetime.datetime.now().timestamp())
    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = np.inf
    best_epoch = cnfg['max_epoch'] + 1
    t.autograd.set_detect_anomaly(True)

    pin_memory = cnfg['num_workers'] > 0

    train_dataset = UserBatchIncrementDataset(pathlib.Path(cnfg['data_dir'], train_file), item2idx['pad'],
                                              cnfg['window_size'])

    for epoch in range(1, cnfg['max_epoch'] + 1):
        train_loader = DataLoader(train_dataset, batch_size=cnfg['mini_batch'], shuffle=True,
                                  num_workers=cnfg['num_workers'], pin_memory=pin_memory)
        train_loss, sgns = sgns.run_epoch(train_loader, epoch, sgns, optim)
        writer.add_scalar("Loss/train", train_loss, epoch)

        if valid_dl:
            valid_loss = calc_loss_on_set(sgns, valid_dl, item2idx['pad'])
            writer.add_scalar("Loss/validation", valid_loss, epoch)
            print(f'valid loss:{valid_loss}')

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_epoch = epoch
            scheduler.step()
            # valid loss is reported to decide on pruning the epoch
            trial.report(valid_loss, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        else:
            # Save model in each iteration in case we are not in early_stop mode
            save_model(cnfg, model, sgns)

    writer.flush()
    writer.close()

    return best_val_loss, best_epoch


def train_evaluate(cnfg, trial):
    print(cnfg)
    valid_users_path = pathlib.Path(cnfg['data_dir'], 'valid.dat')
    item2idx = pickle.load(pathlib.Path(cnfg['data_dir'], 'item2idx.dat').open('rb'))
    valid_dataset = UserBatchIncrementDataset(valid_users_path, item2idx['pad'], cnfg['window_size'])
    pin_memory = cnfg['num_workers'] > 0
    valid_dl = DataLoader(valid_dataset, batch_size=cnfg['mini_batch'], shuffle=False,
                          num_workers=cnfg['num_workers'], pin_memory=pin_memory)
    set_random_seed(cnfg['seed'])
    best_val_loss, best_epoch = train(cnfg, 'train.dat', valid_dl, trial)
    return best_val_loss, best_epoch


def main():
    args = parse_args()
    if not len(os.listdir(args.data_dir)):
        print("Generating train files...")
        generate_train_files(args.data_cnfg)

    cnfg = pickle.load(open(args.best_cnfg, "rb"))
    args = vars(args)
    cnfg['max_epoch'] = int(cnfg['best_epoch'])
    set_random_seed(cnfg['seed'])
    train({**cnfg, **args}, 'full_train.dat')


if __name__ == '__main__':
    main()
