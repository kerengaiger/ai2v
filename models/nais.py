# -*- coding: utf-8 -*-

import numpy as np
import torch as t
import torch.nn as nn
from tqdm import tqdm
import datetime

from torch import FloatTensor as FT


class Nais(nn.Module):
    def __init__(self, embedding_size, window_size, device, n_w, padding_idx, vocab_size, alpha):
        super(Nais, self).__init__()
        self.name = 'ai2v'
        self.vocab_size = vocab_size
        self.pad_idx = padding_idx
        self.emb_size = embedding_size
        self.window_size = window_size
        self.device = device
        self.alpha = alpha
        self.tvectors = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=padding_idx)
        self.cvectors = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=padding_idx)
        self.tvectors.weight = nn.Parameter(t.cat([FT(self.vocab_size - 1,
                                                      self.emb_size).uniform_(-0.5 / self.emb_size,
                                                                              0.5 / self.emb_size),
                                                   t.zeros(1, self.emb_size)]))
        self.cvectors.weight = nn.Parameter(t.cat([FT(self.vocab_size - 1,
                                                      self.emb_size).uniform_(-0.5 / self.emb_size,
                                                                              0.5 / self.emb_size),
                                                   t.zeros(1, self.emb_size)]))
        self.tvectors.weight.requires_grad = True
        self.cvectors.weight.requires_grad = True
        self.W = nn.Linear(2 * self.emb_size, n_w)
        self.h = nn.Linear(n_w, 1)

    def represent_users(self, queries, keys, values, mask_pad_ids=None):
        b_s, n_t_items = queries.shape[:2]

        p_repeat = queries.unsqueeze(2).repeat(1, 1, self.window_size, 1)
        q_repeat = keys.unsqueeze(1).repeat(1, n_t_items, 1, 1)
        concat = t.cat((p_repeat, q_repeat), -1)
        att = self.h(t.relu(self.W(concat))).squeeze()

        if mask_pad_ids is not None:
            attention_mask = mask_pad_ids.unsqueeze(1).repeat(1, n_t_items, 1)
            att = att.masked_fill(attention_mask, -np.inf)

        att = t.softmax(att, -1)

        out = t.matmul(att, values)  # (b_s, n_t_items, emb_size)
        norm = 1000 - mask_pad_ids.sum(axis=1)
        norm = norm.pow(self.alpha)
        out_norm = out / norm.unsqueeze(1).unsqueeze(2)
        return out_norm

    def forward(self, batch_titems, batch_citems, mask_pad_ids=None):
        v_l_j = self.forward_t(batch_titems)
        u_l_m = self.forward_c(batch_citems)

        sub_users = self.represent_users(v_l_j, u_l_m, u_l_m, mask_pad_ids)
        return sub_users

    def forward_t(self, data):
        v = data.long()
        return self.tvectors(v)

    def forward_c(self, data):
        v = data.long()
        return self.cvectors(v)


class SGNS(nn.Module):
    def __init__(self, base_model, vocab_size, n_negs, weights, loss_method, device):
        super(SGNS, self).__init__()
        self.nais = base_model
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.device = device
        self.weights = weights
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
        self.loss_method = loss_method

    def similarity(self, batch_sub_users, batch_tvecs):
        sim = t.diagonal(
            t.bmm(batch_sub_users, batch_tvecs.view(batch_tvecs.shape[0], batch_tvecs.shape[2], -1)),
            offset=0, dim1=-2, dim2=-1)
        return sim

    def inference(self, user_items):
        if len(user_items) < self.nais.window_size:
            pad_times = self.nais.window_size - len(user_items)
            user_items = [self.nais.pad_idx] * pad_times + user_items
        else:
            user_items = user_items[-self.nais.window_size:]
        num_items = self.nais.tvectors.weight.size()[0]
        citems = t.tensor([user_items])
        citems = citems.to(self.device)
        all_titems = t.tensor(range(num_items)).unsqueeze(0)
        all_titems = all_titems.to(self.device)
        mask_pad_ids = citems == self.nais.pad_idx
        sub_users = self.nais(all_titems, citems, mask_pad_ids=mask_pad_ids)
        all_tvecs = self.nais.forward_t(all_titems)
        sim = self.similarity(sub_users, all_tvecs, all_titems)
        return sim.squeeze(-1).squeeze(0).detach().cpu().numpy()

    def forward(self, batch_titems, batch_citems):
        if self.weights is not None:
            batch_nitems = t.multinomial(
                self.weights, batch_titems.size()[0] * self.n_negs, replacement=True).view(
                batch_titems.size()[0], -1)
        else:
            batch_nitems = FT(batch_titems.size()[0], self.n_negs).uniform_(0, self.vocab_size - 1).long()
        if next(self.parameters()).is_cuda:
            batch_nitems = batch_nitems.to(self.device)

        batch_titems = t.cat([batch_titems.reshape(-1, 1), batch_nitems], 1)
        mask_pad_ids = (batch_citems == self.nais.pad_idx)
        batch_sub_users = self.nais(batch_titems, batch_citems, mask_pad_ids)
        batch_tvecs = self.nais.forward_t(batch_titems)

        sim = self.similarity(batch_sub_users, batch_tvecs, batch_titems)

        if self.loss_method == 'CCE':  # This option is the default option.
            soft = sim.softmax(dim=1) + 1e-6
            return -soft[:, 0].log().sum()

        if self.loss_method == 'BCE':
            soft_pos = sim[:, 0].sigmoid() + 1e-6
            soft_neg = sim[:, 1:].neg().sigmoid() + 1e-6
            return (-soft_pos.log().sum()) + (-soft_neg.log().sum())

        if self.loss_method == 'Hinge':
            soft_pos = t.maximum((t.ones_like(sim[:, 0]) - sim[:, 0]), t.zeros_like(sim[:, 0])) + 1e-6
            soft_neg = t.maximum((t.ones_like(sim[:, 1:]) - (-sim[:, 1:])), t.zeros_like(sim[:, 1:])) + 1e-6
            return soft_pos.sum() + soft_neg.sum()

    def run_epoch(self, train_dl, epoch, sgns, optim):
        pbar = tqdm(train_dl)
        pbar.set_description("[Epoch {}]".format(epoch))
        train_loss = 0

        srt = datetime.datetime.now().replace(microsecond=0)
        for batch_users, batch_titems, batch_citems in pbar:
            batch_titems, batch_citems = batch_titems.to(self.device), batch_citems.to(self.device)
            loss = sgns(batch_titems, batch_citems)
            train_loss += loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(train_loss=loss.item())

        train_loss = train_loss / len(pbar)
        print(f'train_loss: {train_loss}')
        end = datetime.datetime.now().replace(microsecond=0)
        print('epoch time: ', end - srt)
        return train_loss, sgns
