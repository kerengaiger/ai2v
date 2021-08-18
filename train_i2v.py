# -*- coding: utf-8 -*-

import datetime
import pathlib
import pickle

import numpy as np
import torch as t
from torch.optim import Adagrad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from i2v_model import Item2Vec
from i2v_model import SGNS
from train_utils import save_model, configure_weights, UserBatchIncrementDataset
from evaluation import hr_k, mrr_k

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./output/', help="model directory path")
    parser.add_argument('--train', type=str, default='full_train.dat', help="train file name")
    parser.add_argument('--test', type=str, default='test.dat', help="test users file name")
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    parser.add_argument('--max_batch_size', type=int, default=200, help="max number of training obs in batch")
    parser.add_argument('--log_dir', type=str, default='tensorboard/logs/mylogdir', help="logs dir for tensorboard")
    parser.add_argument('--k', type=int, default=20, help="k to calc hrr_k and mrr_k evaluation metrics")
    parser.add_argument('--hr_out', type=str, default='hr.csv', help="hit at K for each test row")
    parser.add_argument('--rr_out', type=str, default='mrr.csv', help="hit at K for each test row")
    parser.add_argument('--best_cnfg', type=str, default='best_cnfg.pkl', help="best cnfg of hyper params")
    parser.add_argument('--max_epochs', type=int, default=50, help='number of early stop epochs to train the model over')

    return parser.parse_args()


def run_epoch(train_dl, epoch, sgns, optim):
    pbar = tqdm(train_dl)
    pbar.set_description("[Epoch {}]".format(epoch))
    train_losses = []

    for batch_titems, batch_citems in pbar:
        if next(sgns.parameters()).is_cuda:
            batch_titems, batch_citems = batch_titems.cuda(), batch_citems.cuda()
        loss = sgns(batch_titems, batch_citems)

        train_losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_postfix(train_loss=loss.item())

    train_loss = np.array(train_losses).mean()
    print(f'train_loss: {train_loss}')
    return train_loss, sgns


def train(cnfg):
    idx2item = pickle.load(pathlib.Path(cnfg['data_dir'], cnfg['idx2item']).open('rb'))
    item2idx = pickle.load(pathlib.Path(cnfg['data_dir'], cnfg['item2idx']).open('rb'))

    weights = configure_weights(cnfg, idx2item)
    vocab_size = len(idx2item)

    model = Item2Vec(padding_idx=item2idx['pad'], vocab_size=vocab_size, embedding_size=cnfg['e_dim'])
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=cnfg['n_negs'], weights=weights)
    dataset = UserBatchIncrementDataset(pathlib.Path(cnfg['data_dir'], cnfg['train']), item2idx['pad'],
                                        cnfg['window_size'])

    if cnfg['cuda']:
        sgns = sgns.cuda()

    optim = Adagrad(sgns.parameters(), lr=cnfg['lr'])

    for epoch in range(1, cnfg['max_epoch'] + 1):
        train_loader = DataLoader(dataset, batch_size=cnfg['mini_batch'], shuffle=True, num_workers=16, pin_memory=True)
        train_loss, sgns = run_epoch(train_loader, epoch, sgns, optim)

    save_model(cnfg, model, sgns)


def calc_loss_on_set(sgns, valid_users_path, pad_idx, batch_size, window_size):
    dataset = UserBatchIncrementDataset(valid_users_path, pad_idx, window_size)
    valid_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    pbar = tqdm(valid_dl)
    valid_losses = []

    for batch_titems, batch_citems in pbar:
        if next(sgns.parameters()).is_cuda:
            batch_titems, batch_citems = batch_titems.cuda(), batch_citems.cuda()
        loss = sgns(batch_titems, batch_citems)
        valid_losses.append(loss.item())

    return np.array(valid_losses).mean()


def train_early_stop(cnfg, valid_users_path, pad_idx):
    idx2item = pickle.load(pathlib.Path(cnfg['data_dir'], cnfg['idx2item']).open('rb'))

    weights = configure_weights(cnfg, idx2item)
    vocab_size = len(idx2item)

    model = Item2Vec(padding_idx=pad_idx, vocab_size=vocab_size, embedding_size=cnfg['e_dim'])
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=cnfg['n_negs'], weights=weights)

    if cnfg['cuda']:
        sgns = sgns.cuda()

    optim = Adagrad(sgns.parameters(), lr=cnfg['lr'])
    log_dir = cnfg['log_dir'] + '/' + str(datetime.datetime.now().timestamp())
    writer = SummaryWriter(log_dir=log_dir)

    best_epoch = cnfg['max_epoch'] + 1
    valid_losses = [np.inf]
    patience_count = 0

    dataset = UserBatchIncrementDataset(pathlib.Path(cnfg['data_dir'], cnfg['train']), pad_idx, cnfg['window_size'])

    for epoch in range(1, cnfg['max_epoch'] + 1):
        train_loader = DataLoader(dataset, batch_size=cnfg['mini_batch'], shuffle=True, num_workers=16, pin_memory=True)
        train_loss, sgns = run_epoch(train_loader, epoch, sgns, optim)
        writer.add_scalar("Loss/train", train_loss, epoch)
        # log specific training example loss

        valid_loss = calc_loss_on_set(sgns, valid_users_path, pad_idx, cnfg['mini_batch'], cnfg['window_size'])
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

    writer.flush()
    writer.close()

    return best_epoch


def train_evaluate(cnfg):
    print(cnfg)
    valid_users_path = pathlib.Path(cnfg['data_dir'], cnfg['valid'])
    item2idx = pickle.load(pathlib.Path(cnfg['data_dir'], cnfg['item2idx']).open('rb'))

    best_epoch = train_early_stop(cnfg, valid_users_path, item2idx['pad'])

    best_model = t.load(pathlib.Path(cnfg['save_dir'], 'model.pt'))

    valid_loss = calc_loss_on_set(best_model, valid_users_path, item2idx['pad'], cnfg['mini_batch'], cnfg['window_size'])
    return {'valid_loss': (valid_loss, 0.0), 'early_stop_epoch': (best_epoch, 0.0)}


def main():
    args = parse_args()
    cnfg = pickle.load(open(args.best_cnfg, "rb"))
    args = vars(args)
    train({**cnfg, **args})


if __name__ == '__main__':
    main()
