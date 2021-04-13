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

    def __init__(self, padding_idx, vocab_size=20000, embedding_size=300):
        super(Item2Vec, self).__init__()
        self.name = 'i2v'
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.tvectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.cvectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ivectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_t(self, data):
        # v = LT(data)
        v = data.long()
        v = v.cuda() if self.tvectors.weight.is_cuda else v
        return self.tvectors(v)

    def forward_c(self, data):
        # v = LT(data)
        v = data.long()
        v = v.cuda() if self.cvectors.weight.is_cuda else v
        return self.cvectors(v)


class SGNS(nn.Module):

    def __init__(self, embedding, vocab_size=20000, n_negs=20, weights=None):
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)

    def forward(self, titems, citems):
        batch_size = titems.size()[0]
        context_size = citems.size()[1]
        if self.weights is not None:
            nitems = t.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nitems = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()
        tvectors = self.embedding.forward_t(titems).unsqueeze(2)
        # print('ivectors', ivectors.shape)
        # print('ivectors', ivectors)

        # print('ivectors', ivectors.shape)
        cvectors = self.embedding.forward_c(citems)
        # print('ovectors', ovectors.shape)
        # print('ovectors', ovectors)

        # print('ovectors', ovectors.shape)
        nvectors = self.embedding.forward_t(nitems).neg()
        # print('nvectors', nvectors.shape)
        # print('mult o and i', t.bmm(ovectors, ivectors).shape)
        tloss = t.bmm(cvectors, tvectors).squeeze(dim=-1).sigmoid().log()
        # print('oloss', oloss.shape)
        # print('oloss', oloss.shape)
        assert tloss.shape[0] == batch_size, 'tloss vector shape is different than batch size'
        nloss = t.bmm(nvectors, tvectors).squeeze(dim=-1).sigmoid().log().view(-1, context_size, self.n_negs).sum(2)
        # print('mult n and i', t.bmm(nvectors, ivectors).shape)
        # print('nloss', nloss.shape)
        assert nloss.shape[0] == batch_size, 'nloss vector shape is different than batch size'
        loss = tloss + nloss
        loss = loss.sum(1).mean()
        return -loss