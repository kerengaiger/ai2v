# -*- coding: utf-8 -*-

import numpy as np
import torch as t
import torch.nn as nn

from torch import FloatTensor as FT


class AttentiveItemToVec(nn.Module):
    def __init__(self, padding_idx, vocab_size, embedding_size, d_alpha=40, N=1):
        super(AttentiveItemToVec, self).__init__()
        self.name = 'ai2v'
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.d_alpha = d_alpha
        self.pad_idx = padding_idx
        self.tvectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.cvectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.tvectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size),
                                                   FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size,
                                                                                                         0.5 / self.embedding_size)]))
        self.cvectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size),
                                                   FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size,
                                                                                                         0.5 / self.embedding_size)]))
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

    def forward(self, batch_titems, batch_citems, batch_pad_ids):
        v_l_j = self.forward_t(batch_titems)
        u_l_m = self.forward_c(batch_citems)
        c_vecs = self.Ac(u_l_m).unsqueeze(1)
        t_vecs = self.At(v_l_j).unsqueeze(2)

        cosine_sim = self.cos(t_vecs, c_vecs)
        batch_pad_ids = (batch_pad_ids[0].repeat_interleave(batch_titems.shape[1]),
                         t.cat([t.tensor(range(batch_titems.shape[1]))] * batch_pad_ids[0].shape[0]),
                         batch_pad_ids[1].repeat_interleave(batch_titems.shape[1]))

        cosine_sim[batch_pad_ids] = -np.inf

        attention_weights = self.softmax(cosine_sim)
        weighted_u_l_m = t.mul(attention_weights.unsqueeze(-1), self.Bc(u_l_m).unsqueeze(1))
        alpha_j_1 = weighted_u_l_m.sum(2)
        z_j_1 = self.R(alpha_j_1)

        return z_j_1

    def forward_t(self, data):
        v = data.long()
        v = v.cuda() if self.tvectors.weight.is_cuda else v
        return self.tvectors(v)

    def forward_c(self, data):
        v = data.long()
        v = v.cuda() if self.cvectors.weight.is_cuda else v
        return self.cvectors(v)


class SGNS(nn.Module):

    def __init__(self, ai2v, vocab_size=20000, n_negs=10, weights=None):
        super(SGNS, self).__init__()
        self.ai2v = ai2v
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)

    def similarity(self, batch_sub_users, batch_tvecs, batch_titem_ids):
        return self.ai2v.W1(self.ai2v.relu(self.ai2v.W0(t.cat([batch_sub_users, batch_tvecs,
                                                        t.mul(batch_sub_users, batch_tvecs),
                                                        batch_sub_users - batch_tvecs], 2)))) + \
            self.ai2v.b_l_j[batch_titem_ids].unsqueeze(2)

    def represent_user(self, citems, titem):
        pad_ids = (citems == self.ai2v.pad_idx).nonzero(as_tuple=True)
        return self.ai2v(titem, citems, pad_ids)

    def inference(self, user_items):
        num_items = self.ai2v.tvectors.weight.size()[0]
        citems = t.tensor([user_items])
        all_titems = t.tensor(range(num_items))
        v_l_j = self.ai2v.forward_t(all_titems)
        u_l_m = self.ai2v.forward_c(citems)
        c_vecs = self.ai2v.Ac(u_l_m)
        t_vecs = self.ai2v.At(v_l_j)
        cosine_sim = self.ai2v.cos(t_vecs.unsqueeze(1), c_vecs)
        attention_weights = self.ai2v.softmax(cosine_sim)
        weighted_u_l_m = t.mul(attention_weights.unsqueeze(-1), self.ai2v.Bc(u_l_m))
        alpha_j_1 = weighted_u_l_m.sum(1)
        z_j_1 = self.ai2v.R(alpha_j_1)
        all_tvecs = self.ai2v.Bt(self.ai2v.forward_t(all_titems))
        sim = self.ai2v.W1(self.ai2v.relu(self.ai2v.W0(t.cat([z_j_1,
                                                              all_tvecs,
                                                              t.mul(z_j_1, all_tvecs),
                                                              z_j_1 - all_tvecs], 1)))) + \
            self.ai2v.b_l_j[all_titems].unsqueeze(1)
        return sim.squeeze().detach().cpu().numpy()

    def forward(self, batch_titems, batch_citems, batch_pad_ids):
        batch_size = batch_titems.size()[0]
        if self.weights is not None:
            batch_nitems = t.multinomial(self.weights, batch_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            batch_nitems = FT(batch_size, self.n_negs).uniform_(0, self.vocab_size - 1).long()

        batch_titems = t.cat([batch_titems.reshape(-1, 1), batch_nitems], 1)
        batch_sub_users = self.ai2v(batch_titems, batch_citems, batch_pad_ids)
        batch_tvecs = self.ai2v.Bt(self.ai2v.forward_t(batch_titems))

        if [param for param in self.ai2v.parameters()][0].is_cuda:
            self.ai2v.b_l_j.cuda()

        sim = self.similarity(batch_sub_users, batch_tvecs, batch_titems)

        return -sim.squeeze(-1).softmax(dim=1)[:, 0].log().sum()
