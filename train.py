# -*- coding: utf-8 -*-
import datetime
import pathlib
import pickle

import numpy as np
import torch as t
from torch.optim import Adagrad, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train_utils import save_model, configure_weights, UserBatchIncrementDataset, set_random_seed
import models
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./output/', help="model directory path")
    parser.add_argument('--train', type=str, default='full_train.dat', help="train file name")
    parser.add_argument('--test', type=str, default='test.dat', help="test users file name")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    parser.add_argument('--device', type=str, default=0, help="cude device to use")
    parser.add_argument('--log_dir', type=str, default='tensorboard/logs/mylogdir', help="logs dir for tensorboard")
    parser.add_argument('--best_cnfg', type=str, default='best_cnfg.pkl', help="best cnfg of hyper params")

    return parser.parse_args()


def calc_loss_on_set(sgns, valid_dl, pad_idx):
    pbar = tqdm(valid_dl)
    valid_losses = []

    for batch_titems, batch_citems in pbar:
        batch_titems, batch_citems = batch_titems.to(sgns.device), batch_citems.to(sgns.device)

        mask_pad_ids = (batch_citems == pad_idx)
        loss = sgns(batch_titems, batch_citems, mask_pad_ids)
        valid_losses.append(loss.item())

    return np.array(valid_losses).mean()


def train(cnfg, valid_dl=None):
    idx2item = pickle.load(pathlib.Path(cnfg['data_dir'], cnfg['idx2item']).open('rb'))
    item2idx = pickle.load(pathlib.Path(cnfg['data_dir'], cnfg['item2idx']).open('rb'))

    weights = configure_weights(cnfg, idx2item)
    vocab_size = len(idx2item)

    model_base_c = getattr(models, cnfg['model'])
    sgns_c = getattr(models, 'sgns_' + cnfg['model'])

    cnfg['padding_idx'] = item2idx['pad']
    cnfg['vocab_size'] = vocab_size
    model_init = {k: cnfg[k] for k in getattr(models, cnfg['model'] + '_cnfg_keys')}
    print(model_init)

    if cnfg['cuda']:
        device = 'cuda:' + str(cnfg['device'])
    else:
        device = 'cpu'

    model = model_base_c(**model_init)
    sgns = sgns_c(base_model=model, vocab_size=vocab_size, n_negs=cnfg['n_negs'], weights=weights,
                  loss_method=cnfg['loss_method'], device=device)
    sgns.to(device)
    optim = Adagrad(sgns.parameters(), lr=cnfg['lr'])
    scheduler = lr_scheduler.MultiStepLR(optim, milestones=[2, 4, 5, 6, 7, 8, 10, 12, 14, 16], gamma=0.5)

    log_dir = cnfg['log_dir'] + '/' + str(datetime.datetime.now().timestamp())
    writer = SummaryWriter(log_dir=log_dir)

    best_epoch = cnfg['max_epoch'] + 1
    valid_losses = [np.inf]
    patience_count = 0
    t.autograd.set_detect_anomaly(True)

    pin_memory = cnfg['num_workers'] > 0

    train_dataset = UserBatchIncrementDataset(pathlib.Path(cnfg['data_dir'], cnfg['train']), item2idx['pad'],
                                        cnfg['window_size'])

    for epoch in range(1, cnfg['max_epoch'] + 1):
        train_loader = DataLoader(train_dataset, batch_size=cnfg['mini_batch'], shuffle=True,
                                  num_workers=cnfg['num_workers'], pin_memory=pin_memory)
        train_loss, sgns = sgns.run_epoch(train_loader, epoch, sgns, optim, item2idx['pad'])
        writer.add_scalar("Loss/train", train_loss, epoch)

        if valid_dl:
            valid_loss = calc_loss_on_set(sgns, valid_dl, item2idx['pad'])
            writer.add_scalar("Loss/validation", valid_loss, epoch)
            print(f'valid loss:{valid_loss}')

            if valid_loss < valid_losses[-1]:
                patience_count = 0
                best_epoch = epoch
                save_model(cnfg, model, sgns)

            else:
                patience_count += 1
                if patience_count == cnfg['patience']:
                    print(f"Early stopping")
                    break

            valid_losses.append(valid_loss)
            scheduler.step()
        else:
            # Save model in each iteration in case we are not in early_stop mode
            save_model(cnfg, model, sgns)

    writer.flush()
    writer.close()

    return best_epoch


def train_evaluate(cnfg):
    print(cnfg)
    valid_users_path = pathlib.Path(cnfg['data_dir'], cnfg['valid'])
    item2idx = pickle.load(pathlib.Path(cnfg['data_dir'], cnfg['item2idx']).open('rb'))
    valid_dataset = UserBatchIncrementDataset(valid_users_path, item2idx['pad'], cnfg['window_size'])
    pin_memory = cnfg['num_workers'] > 0
    valid_dl = DataLoader(valid_dataset, batch_size=cnfg['mini_batch'], shuffle=False,
                          num_workers=cnfg['num_workers'], pin_memory=pin_memory)
    set_random_seed(cnfg['seed'])
    best_epoch = train(cnfg, valid_dl)

    best_model = t.load(pathlib.Path(cnfg['save_dir'], 'model.pt'))

    valid_loss = calc_loss_on_set(best_model, valid_dl, item2idx['pad'])
    return {'valid_loss': (valid_loss, 0.0), 'early_stop_epoch': (best_epoch, 0.0)}


def main():
    args = parse_args()
    cnfg = pickle.load(open(args.best_cnfg, "rb"))
    args = vars(args)
    cnfg['max_epoch'] = int(cnfg['best_epoch'])
    set_random_seed(cnfg['seed'])
    train({**cnfg, **args})


if __name__ == '__main__':
    main()
