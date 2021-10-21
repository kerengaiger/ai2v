# -*- coding: utf-8 -*-

import numpy as np
import torch as t
import torch.nn as nn
from tqdm import tqdm

from torch import FloatTensor as FT
from sklearn.metrics.pairwise import cosine_similarity


class Bundler(nn.Module):

    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError


class Item2Vec(Bundler):

    def __init__(self, padding_idx, vocab_size=20000, emb_size=300):
        super(Item2Vec, self).__init__()
        self.name = 'i2v'
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.tvectors = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=padding_idx)
        self.cvectors = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=padding_idx)
        self.tvectors.weight = nn.Parameter(t.cat([FT(self.vocab_size - 1, self.emb_size).uniform_(
            -0.5 / self.emb_size, 0.5 / self.emb_size), t.zeros(1, self.emb_size)]))
        self.cvectors.weight = nn.Parameter(t.cat([FT(self.vocab_size - 1, self.emb_size).uniform_(
            -0.5 / self.emb_size, 0.5 / self.emb_size), t.zeros(1, self.emb_size)]))
        self.tvectors.weight.requires_grad = True
        self.cvectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_t(self, data):
        v = data.long()
        return self.tvectors(v)

    def forward_c(self, data):
        v = data.long()
        return self.cvectors(v)


class SGNS(nn.Module):

    def __init__(self, base_model, vocab_size=20000, n_negs=20, weights=None, loss_method='CCE', device='cpu'):
        super(SGNS, self).__init__()
        self.embedding = base_model
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
        self.loss_methos = loss_method
        self.device = device

    def forward(self, titems, citems):
        batch_size = titems.size()[0]
        context_size = citems.size()[1]
        if self.weights is not None:
            nitems = t.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nitems = FT(batch_size, self.n_negs).uniform_(0, self.vocab_size - 1).long()
        nitems = nitems.to(self.device)
        tvectors = self.embedding.forward_t(titems)
        cvectors = self.embedding.forward_c(citems)
        nvectors = self.embedding.forward_t(nitems).neg()

        all_tvectors = t.cat([tvectors.unsqueeze(1), nvectors], dim=1)
        if self.loss_method == 'CCE':
            loss = t.bmm(cvectors, all_tvectors.transpose(1, 2))
            loss = -loss.sigmoid().log().sum(2).sum(1).mean()
            return loss
        else:
            raise NotImplementedError

    def represent_user(self, user_itemids):
        context_vecs = self.embedding.cvectors.weight.data.cpu().numpy()
        user2vec = context_vecs[user_itemids, :].mean(axis=0)
        return user2vec

    def inference(self, user_itemids):
        user2vec = np.expand_dims(self.represent_user(user_itemids), axis=0)
        user_sim = cosine_similarity(user2vec, self.embedding.tvectors.weight.data.cpu().numpy()).squeeze()
        return user_sim

    def run_epoch(self, train_dl, epoch, sgns, optim):
        pbar = tqdm(train_dl)
        pbar.set_description("[Epoch {}]".format(epoch))
        train_losses = []

        for batch_titems, batch_citems in pbar:
            batch_titems, batch_citems = batch_titems.to(self.device), batch_citems.to(self.device)
            loss = sgns(batch_titems, batch_citems)

            train_losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(train_loss=loss.item())

        train_loss = np.array(train_losses).mean()
        print(f'train_loss: {train_loss}')
        return train_loss, sgns