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

from i2v_model import Item2Vec
from i2v_model import SGNS
from train_utils import save_model, configure_weights


class UserBatchDataset(Dataset):

    def __init__(self, datapath, max_batch_size, ws=None):
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
            for i in range(len(user)):
                oitems = [j for j in user if j != user[i]]
                batch[0].append(user[i])
                batch[1].append(oitems)
                if len(batch[0]) == max_batch_size and i < (len(user) - 1):
                    data_batches.append(batch)
                    batch = ([], [])
            data_batches.append(batch)

        self.data = data_batches

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch_iitems, batch_oitems = self.data[idx]
        return batch_iitems, np.array(batch_oitems)


def run_epoch(train_dl, epoch, sgns, optim):
    pbar = tqdm(train_dl)
    pbar.set_description("[Epoch {}]".format(epoch))
    train_losses = []

    ##### Remove #####
    i = 0
    for batch_iitem, batch_oitems in pbar:
        batch_iitem = t.tensor(batch_iitem)
        batch_oitems = batch_oitems.squeeze(0)
        loss = sgns(batch_iitem, batch_oitems)

        ##### Remove #####
        if i == 0:
            print('first batch loss:', loss.item())
        train_losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_postfix(train_loss=loss.item())
        ### Remove ####
        i += 1

    train_loss = np.array(train_losses).mean()
    print(f'train_loss: {train_loss}')
    return train_loss, sgns


def train(cnfg):
    idx2item = pickle.load(pathlib.Path(cnfg['data_dir'], 'idx2item.dat').open('rb'))

    weights = configure_weights(cnfg, idx2item)
    vocab_size = len(idx2item)

    model = Item2Vec(vocab_size=vocab_size, embedding_size=cnfg['e_dim'])
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=cnfg['n_negs'], weights=weights)
    dataset = UserBatchDataset(pathlib.Path(cnfg['data_dir'], cnfg['train']), cnfg['max_batch_size'])
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    if cnfg['cuda']:
        sgns = sgns.cuda()

    optim = Adagrad(sgns.parameters(), lr=cnfg['lr'])

    for epoch in range(1, cnfg['max_epoch'] + 1):
        _train_loss = run_epoch(train_loader, epoch, sgns, optim)

    save_model(cnfg, sgns)


def calc_loss_on_set(sgns, valid_users_path, cnfg):
    dataset = UserBatchDataset(valid_users_path, cnfg['max_batch_size'])
    valid_dl = DataLoader(dataset, batch_size=1, shuffle=False)

    pbar = tqdm(valid_dl)
    valid_losses = []

    for batch_iitem, batch_oitems in pbar:
        batch_iitem = t.tensor(batch_iitem)
        batch_oitems = batch_oitems.squeeze(0)
        loss = sgns(batch_iitem, batch_oitems)
        valid_losses.append(loss.item())

    return np.array(valid_losses).mean()


def train_early_stop(cnfg, valid_users_path):
    idx2item = pickle.load(pathlib.Path(cnfg['data_dir'], 'idx2item.dat').open('rb'))

    weights = configure_weights(cnfg, idx2item)
    vocab_size = len(idx2item)

    model = Item2Vec(vocab_size=vocab_size, embedding_size=cnfg['e_dim'])
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=cnfg['n_negs'], weights=weights)

    if cnfg['cuda']:
        sgns = sgns.cuda()

    optim = Adagrad(sgns.parameters(), lr=cnfg['lr'])
    writer = SummaryWriter()

    best_epoch = cnfg['max_epoch'] + 1
    valid_losses = [np.inf]
    best_valid_loss = np.inf
    patience_count = 0

    for epoch in range(1, cnfg['max_epoch'] + 1):
        dataset = UserBatchDataset(pathlib.Path(cnfg['data_dir'], cnfg['train']), cnfg['max_batch_size'])
        train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

        train_loss, sgns = run_epoch(train_loader, epoch, sgns, optim)
        writer.add_scalar("Loss/train", train_loss, epoch)
        # log specific training example loss

        valid_loss = calc_loss_on_set(sgns, valid_users_path, cnfg)
        writer.add_scalar("Loss/validation", valid_loss, epoch)
        print(f'valid loss:{valid_loss}')

        diff_loss = abs(valid_loss - valid_losses[-1])
        if diff_loss > cnfg['conv_thresh']:
            patience_count = 0
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                save_model(cnfg, sgns)

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

    best_epoch = train_early_stop(cnfg, valid_users_path)

    best_model = t.load(pathlib.Path(cnfg['save_dir'], 'best_model.pt'))

    valid_loss = calc_loss_on_set(best_model, valid_users_path, cnfg)
    return {'valid_loss': (valid_loss, 0.0), 'early_stop_epoch': (best_epoch, 0.0)}

