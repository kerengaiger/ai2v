# -*- coding: utf-8 -*-

import datetime
from tqdm import tqdm
import numpy as np
import torch as t
import torch.nn as nn

from torch import FloatTensor as FT


class AttentiveItemToVec(nn.Module):
    def __init__(self, padding_idx, vocab_size, e_dim, num_heads,
                 num_blocks, dropout_rate):
        super(AttentiveItemToVec, self).__init__()
        self.name = 'ai2v'
        self.vocab_size = vocab_size
        self.e_dim = e_dim
        self.pad_idx = padding_idx
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.tvectors = nn.Embedding(self.vocab_size, self.e_dim, padding_idx=padding_idx)
        self.cvectors = nn.Embedding(self.vocab_size, self.e_dim, padding_idx=padding_idx)
        self.last_item_vectors = nn.Embedding(self.vocab_size, self.e_dim, padding_idx=padding_idx)
        self.tvectors.weight = nn.Parameter(t.cat([FT(self.vocab_size - 1,
                                                      self.e_dim).uniform_(-0.5 / self.e_dim,
                                                                           0.5 / self.e_dim),
                                                   t.zeros(1, self.e_dim)]))
        self.cvectors.weight = nn.Parameter(t.cat([FT(self.vocab_size - 1,
                                                      self.e_dim).uniform_(-0.5 / self.e_dim,
                                                                           0.5 / self.e_dim),
                                                   t.zeros(1, self.e_dim)]))
        self.last_item_vectors.weight = nn.Parameter(t.cat([FT(self.vocab_size - 1,
                                                               self.e_dim).uniform_(-0.5 / self.e_dim,
                                                                                    0.5 / self.e_dim),
                                                     t.zeros(1, self.e_dim)]))
        self.tvectors.weight.requires_grad = True
        self.cvectors.weight.requires_grad = True
        self.last_item_vectors.weight.requires_grad = True
        self.Bt = nn.Linear(self.e_dim, self.e_dim)
        self.W0 = nn.Linear(4 * self.e_dim, self.e_dim)
        self.W1 = nn.Linear(self.e_dim, 1)
        self.relu = nn.ReLU()
        self.b_l_j = nn.Parameter(FT(self.vocab_size).uniform_(-0.5 / self.e_dim, 0.5 / self.e_dim))
        self.b_l_j.requires_grad = True
        self.attention_layers = t.nn.ModuleList()
        self.attention_layernorms = t.nn.ModuleList()  # to be Q for self-attention

        for _ in range(num_blocks):
            new_attn_layernorm = t.nn.LayerNorm(e_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = t.nn.MultiheadAttention(e_dim,
                                                     num_heads,
                                                     dropout_rate,
                                                     batch_first=True)
            self.attention_layers.append(new_attn_layer)

    def forward(self, batch_titems, batch_citems, mask_pad_ids=None, inference=False):
        v_l_j = self.forward_t(batch_titems)
        u_l_m = self.forward_c(batch_citems)
        Q = v_l_j
        for i in range(len(self.attention_layers)):
            if not inference:
                Q = self.attention_layernorms[i](Q)
                outputs, attention_weights = self.attention_layers[i](Q, u_l_m, u_l_m, key_padding_mask=mask_pad_ids)
            else:
                Q = self.attention_layernorms[i](v_l_j)
                outputs, attention_weights = self.attention_layers[i](Q, u_l_m, u_l_m)
        return outputs, attention_weights

    def forward_t(self, data):
        v = data.long()
        return self.tvectors(v)

    def forward_c(self, data):
        v = data.long()
        return self.cvectors(v)


class SGNS(nn.Module):

    def __init__(self, base_model, vocab_size=20000, n_negs=10, weights=None, loss_method='CCE', device='cpu'):
        super(SGNS, self).__init__()
        self.ai2v = base_model
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
        self.loss_method = loss_method
        self.device = device

    def similarity(self, batch_sub_users, batch_tvecs, batch_titem_ids):
        return self.ai2v.W1(self.ai2v.relu(self.ai2v.W0(t.cat([batch_sub_users, batch_tvecs,
                                                        t.mul(batch_sub_users, batch_tvecs),
                                                        (batch_sub_users - batch_tvecs).abs()], 2)))) + \
            self.ai2v.b_l_j[batch_titem_ids].unsqueeze(2)

    def inference(self, user_items):
        num_items = self.ai2v.tvectors.weight.size()[0]
        citems = t.tensor([user_items])
        all_titems = t.tensor(range(num_items)).unsqueeze(0)

        citems, all_titems = citems.to(self.device), all_titems.to(self.device)

        sub_user, _ = self.ai2v(all_titems, citems, mask_pad_ids=None, inference=True)
        last_item_emb = self.ai2v.last_item_vectors(citems[:, -1].long())
        sub_user = sub_user + last_item_emb
        all_tvecs = self.ai2v.Bt(self.ai2v.forward_t(all_titems))
        sim = self.similarity(sub_user, all_tvecs, all_titems)
        return sim.squeeze(-1).squeeze(0).detach().cpu().numpy()

    def forward(self, batch_titems, batch_citems, mask_pad_ids):
        if self.weights is not None:
            batch_nitems = t.multinomial(self.weights, batch_titems.size()[0] * self.n_negs, replacement=True).view(batch_titems.size()[0], -1)
        else:
            batch_nitems = FT(batch_titems.size()[0], self.n_negs).uniform_(0, self.vocab_size - 1).long()
        batch_nitems = batch_nitems.to(self.device)

        batch_titems = t.cat([batch_titems.reshape(-1, 1), batch_nitems], 1)
        batch_last_items = self.ai2v.last_item_vectors(batch_citems[:, -1])
        batch_last_items = batch_last_items.unsqueeze(1).repeat(1, batch_titems.shape[1], 1)
        batch_sub_users, _ = self.ai2v(batch_titems, batch_citems, mask_pad_ids)
        batch_sub_users = batch_sub_users + batch_last_items
        batch_tvecs = self.ai2v.Bt(self.ai2v.forward_t(batch_titems))
        if [param for param in self.ai2v.parameters()][0].is_cuda:
            self.ai2v.b_l_j.cuda()

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

        else:
            raise NotImplementedError

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
