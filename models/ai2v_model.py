# -*- coding: utf-8 -*-

import numpy as np
import torch as t
import torch.nn as nn
from tqdm import tqdm
import datetime

from torch import FloatTensor as FT


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, window_size, d_k, d_v, num_h):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = embedding_size
        self.window_size = window_size
        self.d_k = d_k
        self.d_v = d_v
        self.num_h = num_h
        self.Ac = nn.Linear(self.emb_size, self.num_h * self.d_k)
        self.At = nn.Linear(self.emb_size, self.num_h * self.d_k)
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.Bc = nn.Linear(self.emb_size, self.num_h * self.d_v)
        self.pos_bias = nn.Parameter(FT(self.window_size).uniform_(-0.5 / self.window_size,
                                                                   0.5 / self.window_size))
        self.pos_bias.requires_grad = True
        self.R = nn.Linear(self.num_h * self.d_v, self.emb_size)

    def forward(self, queries, keys, values, attention_mask=None):
        '''
        :param queries: Queries (b_s, n_t_items, emb_size)
        :param keys: Keys (b_s, n_c_items, emb_size)
        :param values: Values (b_s, n_c_items, d_model)
        :param attention_mask: Mask over attention values (b_s, num_h, n_t_items, n_c_items). True indicates masking.
        :return: batch_sub_user (b_s, n_t_items, emb_size)
        '''
        b_s, n_t_items = queries.shape[:2]
        n_c_items = keys.shape[1]
        q = self.At(queries).view(b_s, n_t_items, self.num_h, self.d_k).permute(0, 2, 1, 3)  # (b_s, num_h, n_t_items, d_k)
        k = self.Ac(keys).view(b_s, n_c_items, self.num_h, self.d_k).permute(0, 2, 3, 1)  # (b_s, num_h, d_k, n_c_items)
        v = self.Bc(values).view(b_s, n_c_items, self.num_h, self.d_v).permute(0, 2, 1, 3)  # (b_s, num_h, n_c_items, d_v)

        att = self.cos(q.unsqueeze(3), k.unsqueeze(2))
        if [param for param in self.parameters()][0].is_cuda:
            self.pos_bias.cuda()
        batch_pos_bias = self.pos_bias.repeat(b_s, self.num_h, n_t_items, 1)
        att = att + batch_pos_bias

        if attention_mask is not None:
            attention_mask = attention_mask.repeat()
            att = att.masked_fill(attention_mask, -np.inf)

        att = t.softmax(att, -1)
        out = t.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, n_t_items, self.h * self.d_v)  # (b_s, n_t_items, num_h*d_num_h)
        out = self.R(out)  # (b_s, n_t_items, emb_size)
        return out, att


class AttentiveItemToVec(nn.Module):
    def __init__(self, padding_idx, vocab_size, embedding_size, window_size, n_b=1):
        super(AttentiveItemToVec, self).__init__()
        self.name = 'ai2v'
        self.vocab_size = vocab_size
        self.emb_size = embedding_size
        self.pad_idx = padding_idx
        self.window_size = window_size
        self.n_b = n_b
        self.tvectors = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=padding_idx)
        self.cvectors = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=padding_idx)
        self.tvectors.weight = nn.Parameter(t.cat([FT(self.vocab_size - 1,
                                                      self.emb_size).uniform_(-0.5 / self.emb_size,
                                                                              0.5 / self.emb_size),
                                                   t.zeros(1, self.emb_size)]))
        self.cvectors.weight = nn.Parameter(t.cat([FT(self.vocab_size - 1,
                                                      self.emb_size).uniform_(-0.5 / self.emb_size,
                                                                              0.5 / self.emb_size),
                                                   t.zeros(1, self.embedding_size)]))
        self.tvectors.weight.requires_grad = True
        self.cvectors.weight.requires_grad = True
        self.W0 = nn.Linear(4 * self.embedding_size, self.embedding_size)
        self.W1 = nn.Linear(self.embedding_size, 1)
        self.relu = nn.ReLU()
        self.b_l_j = nn.Parameter(FT(self.vocab_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.b_l_j.requires_grad = True
        self.mha_layers = nn.ModuleList([MultiHeadAttention(embedding_size=embedding_size,
                                                            window_size=window_size,
                                                            d_k=60, d_v=60, num_h=1)
                                        for _ in range(self.n_b)])

    def forward(self, batch_titems, batch_citems, mask_pad_ids=None):
        v_l_j = self.forward_t(batch_titems)
        u_l_m = self.forward_c(batch_citems)
        c_vecs = self.Ac(u_l_m).unsqueeze(1)
        t_vecs = self.At(v_l_j).unsqueeze(2)

        sub_users_l = t_vecs
        for l in self.mha_layers:
            sub_users_l, _ = l(sub_users_l, c_vecs, c_vecs, attention_mask=mask_pad_ids)

        return sub_users_l

    def forward_t(self, data):
        v = data.long()
        return self.tvectors(v)

    def forward_c(self, data):
        v = data.long()
        return self.cvectors(v)


class SGNS(nn.Module):
    def __init__(self, base_model, vocab_size, n_negs, weights, loss_method, device):
        super(SGNS, self).__init__()
        self.ai2v = base_model
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        self.device = device
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
        self.loss_method = loss_method

    def similarity(self, batch_sub_users, batch_tvecs, batch_titem_ids):
        return self.ai2v.W1(self.ai2v.relu(self.ai2v.W0(t.cat([batch_sub_users, batch_tvecs,
                                                        t.mul(batch_sub_users, batch_tvecs),
                                                        (batch_sub_users - batch_tvecs).abs()], 2)))) + \
            self.ai2v.b_l_j[batch_titem_ids].unsqueeze(2)

    def inference(self, user_items):
        if len(user_items) < self.ai2v.window_size:
            pad_times = self.ai2v.window_size - len(user_items)
            user_items = [self.ai2v.pad_idx] * pad_times + user_items
        else:
            user_items = user_items[-self.ai2v.window_size:]
        num_items = self.ai2v.tvectors.weight.size()[0]
        citems = t.tensor([user_items])
        mask_pad_ids = citems == self.ai2v.pad_idx
        all_titems = t.tensor(range(num_items)).unsqueeze(0)
        if next(self.parameters()).is_cuda:
            citems = citems.cuda()
            all_titems = all_titems.cuda()
        sub_users = self.ai2v(all_titems, citems, mask_pad_ids=mask_pad_ids)
        all_tvecs = self.ai2v.Bt(self.ai2v.forward_t(all_titems))
        sim = self.similarity(sub_users, all_tvecs, all_titems)
        return sim.squeeze(-1).squeeze(0).detach().cpu().numpy()

    def forward(self, batch_titems, batch_citems, mask_pad_ids):
        if self.weights is not None:
            batch_nitems = t.multinomial(self.weights, batch_titems.size()[0] * self.n_negs, replacement=True).view(batch_titems.size()[0], -1)
        else:
            batch_nitems = FT(batch_titems.size()[0], self.n_negs).uniform_(0, self.vocab_size - 1).long()
        if next(self.parameters()).is_cuda:
            batch_nitems = batch_nitems.cuda()

        batch_titems = t.cat([batch_titems.reshape(-1, 1), batch_nitems], 1)

        batch_sub_users = self.ai2v(batch_titems, batch_citems, mask_pad_ids)
        batch_tvecs = self.ai2v.Bt(self.ai2v.forward_t(batch_titems))
        if [param for param in self.ai2v.parameters()][0].is_cuda:
            self.ai2v.b_l_j.cuda()

        sim = self.similarity(batch_sub_users, batch_tvecs, batch_titems)

        if self.loss_method == 'CCE':  # This option is the default option.
            soft = sim.softmax(dim=1) + 1e-6
            # print(-soft[:, 0].log().sum())
            return -soft[:, 0].log().sum()

        if self.loss_method == 'BCE':
            soft_pos = sim[:, 0].sigmoid() + 1e-6
            soft_neg = sim[:, 1:].neg().sigmoid() + 1e-6
            # print((-soft_pos.log().sum()) + (-soft_neg.log().sum()))
            return (-soft_pos.log().sum()) + (-soft_neg.log().sum())

        if self.loss_method == 'Hinge':
            soft_pos = t.maximum((t.ones_like(sim[:, 0]) - sim[:, 0]), t.zeros_like(sim[:, 0])) + 1e-6
            soft_neg = t.maximum((t.ones_like(sim[:, 1:]) - (-sim[:, 1:])), t.zeros_like(sim[:, 1:])) + 1e-6
            # print(soft_pos.sum() + soft_neg.sum())
            return soft_pos.sum() + soft_neg.sum()

    def run_epoch(self, train_dl, epoch, sgns, optim, pad_idx):
        pbar = tqdm(train_dl)
        pbar.set_description("[Epoch {}]".format(epoch))
        train_loss = 0

        srt = datetime.datetime.now().replace(microsecond=0)
        for batch_titems, batch_citems in pbar:
            batch_titems, batch_citems = batch_titems.to(self.device), batch_citems.to(self.device)
            mask_pad_ids = (batch_citems == pad_idx)
            loss = sgns(batch_titems, batch_citems, mask_pad_ids)
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
