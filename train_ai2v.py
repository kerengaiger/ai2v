# -*- coding: utf-8 -*-

import pathlib
import pickle
import random

import numpy as np
import torch as t
from torch.optim import Adagrad
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ai2v_model import AttentiveItemToVec
from ai2v_model import SGNS

from train_utils import save_model, configure_weights


def run_epoch(train_dl, epoch, sgns, optim, pad_idx):
    pbar = tqdm(train_dl)
    pbar.set_description("[Epoch {}]".format(epoch))
    train_losses = []

    for batch_titem, batch_citems in pbar:
        batch_titems = t.tensor(batch_titem)
        batch_citems = batch_citems.squeeze(0)

        batch_pad_ids = (batch_citems == pad_idx).nonzero(as_tuple=True)
        loss = sgns(batch_titems, batch_citems, batch_pad_ids)

        train_losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_postfix(train_loss=loss.item())

    train_loss = np.array(train_losses).mean()
    print(f'train_loss: {train_loss}')
    return train_loss, sgns


class UserBatchIncrementDataset(Dataset):
    def __init__(self, datapath, max_batch_size, pad_idx, ws=None):
        data = pickle.load(datapath.open('rb'))
        if ws is not None:
            data_ws = []
            for iitem, oitems in data:
                if random.random() > ws[iitem]:
                    data_ws.append((iitem, oitems))
            data = data_ws

        data_batches = []
        for user in data:
            batch = ([], [])
            for sub_user_i in range(len(user)):
                batch[0].append(user[sub_user_i][1])
                batch[1].append(user[sub_user_i][0] + [pad_idx for _ in range(len(user) - len(user[sub_user_i][0]))])

                if len(batch[0]) == max_batch_size and sub_user_i < (len(user) - 1):
                    data_batches.append(batch)
                    batch = ([], [])
            data_batches.append(batch)

        self.data = data_batches

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch_iitems, batch_oitems = self.data[idx]
        return batch_iitems, np.array(batch_oitems)


def calc_loss_on_set(sgns, valid_users_path, cnfg, pad_idx):

    item2idx = pickle.load(pathlib.Path(cnfg['data_dir'], 'item2idx.dat').open('rb'))
    dataset = UserBatchIncrementDataset(valid_users_path, cnfg['max_batch_size'],
                                        item2idx['pad'])
    valid_dl = DataLoader(dataset, batch_size=1, shuffle=False)

    pbar = tqdm(valid_dl)
    valid_losses = []

    for batch_titem, batch_citems in pbar:
        batch_titem = t.tensor(batch_titem)
        batch_citems = batch_citems.squeeze(0)
        batch_pad_ids = (batch_citems == pad_idx).nonzero(as_tuple=True)

        loss = sgns(batch_titem, batch_citems, batch_pad_ids)
        valid_losses.append(loss.item())

    return np.array(valid_losses).mean()


def train_early_stop(cnfg, valid_users_path, pad_idx):
    idx2item = pickle.load(pathlib.Path(cnfg['data_dir'], 'idx2item.dat').open('rb'))

    weights = configure_weights(cnfg, idx2item)
    vocab_size = len(idx2item)

    model = AttentiveItemToVec(vocab_size=vocab_size, embedding_size=cnfg['e_dim'])
    sgns = SGNS(ai2v=model, vocab_size=vocab_size, n_negs=cnfg['n_negs'], weights=weights)

    if cnfg['cuda']:
        sgns = sgns.cuda()

    optim = Adagrad(sgns.parameters(), lr=cnfg['lr'])
    writer = SummaryWriter()

    best_epoch = cnfg['max_epoch'] + 1
    valid_losses = [np.inf]
    patience_count = 0
    t.autograd.set_detect_anomaly(True)

    for epoch in range(1, cnfg['max_epoch'] + 1):
        dataset = UserBatchIncrementDataset(pathlib.Path(cnfg['data_dir'], cnfg['train']), cnfg['max_batch_size'],
                                            pad_idx)
        train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

        train_loss, sgns = run_epoch(train_loader, epoch, sgns, optim, pad_idx)
        writer.add_scalar("Loss/train", train_loss, epoch)
        # log specific training example loss

        valid_loss = calc_loss_on_set(sgns, valid_users_path, cnfg, pad_idx)
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


def train(cnfg):
    idx2item = pickle.load(pathlib.Path(cnfg['data_dir'], 'idx2item.dat').open('rb'))
    item2idx = pickle.load(pathlib.Path(cnfg['data_dir'], 'item2idx.dat').open('rb'))

    weights = configure_weights(cnfg, idx2item)
    vocab_size = len(idx2item)

    model = AttentiveItemToVec(vocab_size=vocab_size, embedding_size=cnfg['e_dim'])
    sgns = SGNS(ai2v=model, vocab_size=vocab_size, n_negs=cnfg['n_negs'], weights=weights)
    dataset = UserBatchIncrementDataset(pathlib.Path(cnfg['data_dir'], cnfg['train']), cnfg['max_batch_size'],
                                        item2idx['pad'])
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    if cnfg['cuda']:
        sgns = sgns.cuda()

    optim = Adagrad(sgns.parameters(), lr=cnfg['lr'])

    for epoch in range(1, cnfg['max_epoch'] + 1):
        _train_loss = run_epoch(train_loader, epoch, sgns, optim, item2idx['pad'])

    save_model(cnfg, model, sgns)


def train_evaluate(cnfg):
    print(cnfg)
    valid_users_path = pathlib.Path(cnfg['data_dir'], cnfg['valid'])
    item2idx = pickle.load(pathlib.Path(cnfg['data_dir'], 'item2idx.dat').open('rb'))

    best_epoch = train_early_stop(cnfg, valid_users_path, item2idx['pad'])

    best_model = t.load(pathlib.Path(cnfg['save_dir'], cnfg['model'] + '_best.pt'))

    valid_loss = calc_loss_on_set(best_model, valid_users_path, cnfg, item2idx['pad'])
    return {'valid_loss': (valid_loss, 0.0), 'early_stop_epoch': (best_epoch, 0.0)}