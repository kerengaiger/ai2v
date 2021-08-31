# -*- coding: utf-8 -*-

import numpy as np
import torch as t
import torch.nn as nn

from torch import FloatTensor as FT


class AttentiveItemToVec(nn.Module):
    def __init__(self, padding_idx, vocab_size, embedding_size, d_alpha=60, N=1):
        super(AttentiveItemToVec, self).__init__()
        self.name = 'ai2v'
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.d_alpha = d_alpha
        self.pad_idx = padding_idx
        self.tvectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.cvectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.tvectors.weight = nn.Parameter(t.cat([FT(self.vocab_size - 1,
                                                      self.embedding_size).uniform_(-0.5 / self.embedding_size,
                                                                                    0.5 / self.embedding_size),
                                                   t.zeros(1, self.embedding_size)]))
        self.cvectors.weight = nn.Parameter(t.cat([FT(self.vocab_size - 1,
                                                      self.embedding_size).uniform_(-0.5 / self.embedding_size,
                                                                                    0.5 / self.embedding_size),
                                                   t.zeros(1, self.embedding_size)]))
        self.tvectors.weight.requires_grad = True
        self.cvectors.weight.requires_grad = True
        self.Ac = nn.Linear(self.embedding_size, self.d_alpha)
        self.At = nn.Linear(self.embedding_size, self.d_alpha)
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.softmax = nn.Softmax(dim=-1)
        self.Bc = nn.Linear(self.embedding_size, self.embedding_size)
        self.Bt = nn.Linear(self.embedding_size, self.embedding_size)
        self.R = nn.Linear(self.embedding_size, N * self.embedding_size)
        self.W0 = nn.Linear(4 * self.embedding_size, self.embedding_size)
        self.W1 = nn.Linear(self.embedding_size, 1)
        self.relu = nn.ReLU()
        self.b_l_j = nn.Parameter(FT(self.vocab_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.b_l_j.requires_grad = True

    def calc_attention(self, batch_titems, batch_citems):
        v_l_j = self.forward_t(batch_titems)
        u_l_m = self.forward_c(batch_citems)
        c_vecs = self.Ac(u_l_m).unsqueeze(1)
        t_vecs = self.At(v_l_j).unsqueeze(2)
        cosine_sim = self.cos(t_vecs, c_vecs)
        attention_weights = self.softmax(cosine_sim)
        return attention_weights

    def forward(self, batch_titems, batch_citems, mask_pad_ids=None, inference=False):
        v_l_j = self.forward_t(batch_titems)
        u_l_m = self.forward_c(batch_citems)
        c_vecs = self.Ac(u_l_m).unsqueeze(1)
        t_vecs = self.At(v_l_j).unsqueeze(2)

        cosine_sim = self.cos(t_vecs, c_vecs)
        if not inference:
            cosine_sim[t.cat([mask_pad_ids] * batch_titems.shape[1], 1).view(
                batch_titems.shape[0], batch_titems.shape[1], -1)] = -np.inf

        attention_weights = self.softmax(cosine_sim)

        weighted_u_l_m = t.mul(attention_weights.unsqueeze(-1), self.Bc(u_l_m).unsqueeze(1))

        alpha_j_1 = weighted_u_l_m.sum(2)
        z_j_1 = self.R(alpha_j_1)

        return z_j_1

    def forward_t(self, data):
        v = data.long()
        return self.tvectors(v)

    def forward_c(self, data):
        v = data.long()
        return self.cvectors(v)


class SGNS(nn.Module):

    def __init__(self, ai2v, vocab_size=20000, n_negs=10, weights=None, loss_method='CCE'):
        super(SGNS, self).__init__()
        self.ai2v = ai2v
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
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
        num_items = self.ai2v.tvectors.weight.size()[0]
        citems = t.tensor([user_items])
        all_titems = t.tensor(range(num_items)).unsqueeze(0)
        if next(self.parameters()).is_cuda:
            citems = citems.cuda()
            all_titems = all_titems.cuda()
        sub_users = self.ai2v(all_titems, citems, mask_pad_ids=None, inference=True)
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
            print(-soft[:, 0].log().sum())
            return -soft[:, 0].log().sum()

        if self.loss_method == 'BCE':
            soft_pos = sim[:, 0].sigmoid() + 1e-6
            soft_neg = sim[:, 1:].neg().sigmoid() + 1e-6
            print((-soft_pos.log().sum()) + (-soft_neg.log().sum()))
            return (-soft_pos.log().sum()) + (-soft_neg.log().sum())

        if self.loss_method == 'Hinge':
            soft_pos = t.maximum((t.ones_like(sim[:, 0]) - sim[:, 0]), t.zeros_like(sim[:, 0])) + 1e-6
            soft_neg = t.maximum((t.ones_like(sim[:, 1:]) - (-sim[:, 1:])), t.zeros_like(sim[:, 1:])) + 1e-6
            print(soft_pos.sum() + soft_neg.sum())
            return soft_pos.sum() + soft_neg.sum()
