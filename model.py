# -*- coding: utf-8 -*-

import numpy as np
import torch as t
import torch.nn as nn

from torch import LongTensor as LT
from torch import FloatTensor as FT


class Bundler(nn.Module):

    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError


class Item2Vec(Bundler):

    def __init__(self, vocab_size=20000, embedding_size=300, padding_idx=0):
        super(Item2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ivectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        # v = LT(data)
        v = data.long()
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        # v = LT(data)
        v = data.long()
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)


class AttentiveItemToVec(nn.Module):
    def __init__(self, vocab_size=20000, embedding_size=300, d_alpha=40, N=1, padding_idx=0):
        super(AttentiveItemToVec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.d_alpha = d_alpha
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
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.softmax = nn.Softmax(dim=1)
        self.Bc = nn.Linear(self.embedding_size, self.embedding_size)
        self.Bt = nn.Linear(self.embedding_size, self.embedding_size)
        self.R = nn.Linear(self.embedding_size, N * self.embedding_size)
        self.W0 = nn.Linear(self.embedding_size, 4 * self.embedding_size)
        self.W1 = nn.Linear(1, self.embedding_size)
        self.relu = nn.ReLU()
        self.b_l_j = nn.Parameter

    def forward(self, batch_titems, batch_citems):
        v_l_j = self.forward_t(batch_titems)
        u_l_m = self.forward_c(batch_citems)
        print(u_l_m)
        print(v_l_j.shape)
        print(u_l_m.shape)
        c_vecs = self.Ac(u_l_m)
        t_vecs = self.At(v_l_j).unsqueeze(1)
        print(t_vecs.shape)
        print(c_vecs.shape)
        cosine_sim = self.cos(t_vecs, c_vecs)
        print(cosine_sim.shape)
        attention_weights = self.softmax(cosine_sim)
        weighted_u_l_m = t.mul(attention_weights.unsqueeze(2), self.Bc(u_l_m))
        alpha_j_1 = weighted_u_l_m.sum(1)
        print(alpha_j_1.shape)
        z_j_1 = self.R(alpha_j_1)
        print(z_j_1.shape)
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

    def __init__(self, ai2v, vocab_size=20000, n_negs=20, weights=None):
        super(SGNS, self).__init__()
        self.ai2v = ai2v
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)

    def forward(self, batch_titems, batch_citems):
        batch_size = batch_titems.size()[0]
        context_size = batch_citems.size()[1]
        if self.weights is not None:
            nitems = t.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nitems = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()

        # call to forward sub_user and forward_t and calculate the similarity between them, and the similarity between
        # sub user and all the negative target items. then user softmax
        batch_sub_users = self.ai2v(batch_titems, batch_citems)
        print('batch_sub users', batch_sub_users.shape)
        batch_t_vecs = self.ai2v.Bt(self.ai2v.forward_t(batch_titems))
        print(batch_t_vecs.shape)

        #
        #
        #
        # ivectors = self.embedding.forward_i(iitem).unsqueeze(2)
        # print('ivectors', ivectors.shape)
        # ovectors = self.embedding.forward_o(oitems)
        # print('ovectors', ovectors.shape)
        # nvectors = self.embedding.forward_o(nitems).neg()
        # # print('nvectors', nvectors.shape)
        # # print('mult o and i', t.bmm(ovectors, ivectors).shape)
        # oloss = t.bmm(ovectors, ivectors).squeeze(dim=-1).sigmoid().log()
        # # print('oloss', oloss.shape)
        # assert oloss.shape[0] == batch_size, 'oloss vector shape is different than batch size'
        # nloss = t.bmm(nvectors, ivectors).squeeze(dim=-1).sigmoid().log().view(-1, context_size, self.n_negs).sum(2)
        # # print('mult n and i', t.bmm(nvectors, ivectors).shape)
        # # print('nloss', nloss.shape)
        # assert nloss.shape[0] == batch_size, 'nloss vector shape is different than batch size'
        # loss = oloss + nloss
        # loss = loss.sum(1).mean()
        # return -loss
