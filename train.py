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
from ai2v_model import SGNS as sgns_ai2v
from i2v_model import Item2Vec
from i2v_model import SGNS as sgns_i2v

import matplotlib.pyplot as plt

ITEM2VEC = 'i2v'
ATTENTIVE_ITEM2VEC = 'ai2v'


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


def run_epoch(train_dl, epoch, sgns, optim, model_name, pad_idx):
    pbar = tqdm(train_dl)
    pbar.set_description("[Epoch {}]".format(epoch))
    train_losses = []

    ##### Remove #####
    i = 0
    for batch_iitem, batch_oitems in pbar:
        batch_iitem = t.tensor(batch_iitem)
        batch_oitems = batch_oitems.squeeze(0)
        if model_name == ITEM2VEC:
            loss = sgns(batch_iitem, batch_oitems)
        else:
            batch_pad_ids = (batch_oitems == pad_idx).nonzero(as_tuple=True)
            loss = sgns(batch_iitem, batch_oitems, batch_pad_ids)
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


def configure_weights(cnfg, idx2item):
    ic = pickle.load(pathlib.Path(cnfg['data_dir'], 'ic.dat').open('rb'))

    ifr = np.array([ic[item] for item in idx2item])
    ifr = ifr / ifr.sum()

    assert (ifr > 0).all(), 'Items with invalid count appear.'
    istt = 1 - np.sqrt(cnfg['ss_t'] / ifr)
    istt = np.clip(istt, 0, 1)
    weights = istt if cnfg['weights'] else None
    return weights


def save_model(cnfg, sgns):
    ivectors = sgns.embedding.ivectors.weight.data.cpu().numpy()
    ovectors = sgns.embedding.ovectors.weight.data.cpu().numpy()
    pickle.dump(ivectors, open(pathlib.Path(cnfg['save_dir'], 'idx2ivec.dat'), 'wb'))
    pickle.dump(ovectors, open(pathlib.Path(cnfg['save_dir'], 'idx2ovec.dat'), 'wb'))
    t.save(sgns, pathlib.Path(cnfg['save_dir'], 'best_model.pt'))


def train(cnfg):
    idx2item = pickle.load(pathlib.Path(cnfg['data_dir'], 'idx2item.dat').open('rb'))
    item2idx = pickle.load(pathlib.Path(cnfg['data_dir'], 'item2idx.dat').open('rb'))

    weights = configure_weights(cnfg, idx2item)
    vocab_size = len(idx2item)

    if cnfg['model'] == Item2Vec:
        model = Item2Vec(vocab_size=vocab_size, embedding_size=cnfg['e_dim'])
        sgns = sgns_i2v(embedding=model, vocab_size=vocab_size, n_negs=cnfg['n_negs'], weights=weights)
        dataset = UserBatchDataset(pathlib.Path(cnfg['data_dir'], cnfg['train']), cnfg['max_batch_size'])
        train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    else:
        model = AttentiveItemToVec(vocab_size=vocab_size, embedding_size=cnfg['e_dim'])
        sgns = sgns_ai2v(ai2v=model, vocab_size=vocab_size, n_negs=cnfg['n_negs'], weights=weights)
        dataset = UserBatchIncrementDataset(pathlib.Path(cnfg['data_dir'], cnfg['train']), cnfg['max_batch_size'],
                                            item2idx['pad'])
        train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    if cnfg['cuda']:
        sgns = sgns.cuda()

    optim = Adagrad(sgns.parameters(), lr=cnfg['lr'])

    for epoch in range(1, cnfg['max_epoch'] + 1):
        _train_loss = run_epoch(train_loader, epoch, sgns, optim, cnfg['model'], item2idx['pad'])

    save_model(cnfg, sgns)


def calc_loss_on_set(sgns, valid_users_path, cnfg):
    if cnfg['model'] == Item2Vec:
        dataset = UserBatchDataset(valid_users_path, cnfg['max_batch_size'])
        valid_dl = DataLoader(dataset, batch_size=1, shuffle=False)
    else:
        item2idx = pickle.load(pathlib.Path(cnfg['data_dir'], 'item2idx.dat').open('rb'))
        dataset = UserBatchIncrementDataset(valid_users_path, cnfg['max_batch_size'],
                                            item2idx['pad'])
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
    item2idx = pickle.load(pathlib.Path(cnfg['data_dir'], 'item2idx.dat').open('rb'))

    weights = configure_weights(cnfg, idx2item)
    vocab_size = len(idx2item)

    if cnfg['model'] == 'Item2Vec':
        model = Item2Vec(vocab_size=vocab_size, embedding_size=cnfg['e_dim'])
        sgns = sgns_i2v(embedding=model, vocab_size=vocab_size, n_negs=cnfg['n_negs'], weights=weights)

    else:
        model = AttentiveItemToVec(vocab_size=vocab_size, embedding_size=cnfg['e_dim'])
        sgns = sgns_ai2v(ai2v=model, vocab_size=vocab_size, n_negs=cnfg['n_negs'], weights=weights)

    if cnfg['cuda']:
        sgns = sgns.cuda()

    optim = Adagrad(sgns.parameters(), lr=cnfg['lr'])
    writer = SummaryWriter()

    best_epoch = cnfg['max_epoch'] + 1
    valid_losses = [np.inf]
    best_valid_loss = np.inf
    patience_count = 0

    for epoch in range(1, cnfg['max_epoch'] + 1):
        if cnfg['model'] == 'Item2Vec':
            dataset = UserBatchDataset(pathlib.Path(cnfg['data_dir'], cnfg['train']), cnfg['max_batch_size'])
            train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        else:
            dataset = UserBatchIncrementDataset(pathlib.Path(cnfg['data_dir'], cnfg['train']), cnfg['max_batch_size'],
                                                item2idx['pad'])
            train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        train_loss, sgns = run_epoch(train_loader, epoch, sgns, optim, cnfg['model'], item2idx['pad'])
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

