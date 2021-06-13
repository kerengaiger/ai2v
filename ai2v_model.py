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
        # self.cos_fin = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.W0 = nn.Linear(4 * self.embedding_size, self.embedding_size)
        self.W1 = nn.Linear(self.embedding_size, 1)
        self.relu = nn.ReLU()
        self.b_l_j = nn.Parameter(FT(self.vocab_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.b_l_j.requires_grad = True
        # self.dropout = nn.Dropout(0.1)

    def forward(self, batch_titems, batch_citems, batch_pad_ids=None, inference=False):
        device = t.device("cuda:0" if next(self.parameters()).is_cuda else "cpu")
        v_l_j = self.forward_t(batch_titems)
        u_l_m = self.forward_c(batch_citems)
        c_vecs = self.Ac(u_l_m).unsqueeze(1)
        t_vecs = self.At(v_l_j).unsqueeze(2)

        cosine_sim = self.cos(t_vecs, c_vecs)
        if not inference:
            tens = batch_pad_ids.repeat_interleave(batch_titems.shape[1], dim=1)
            batch_pad_ids = t.cat([tens[:1],
                                   t.cat([t.tensor(range(batch_titems.shape[1]),
                                                   device=device)] * batch_pad_ids.shape[1]).unsqueeze(0),
                                   tens[1:]], 0)
            cosine_sim[batch_pad_ids] = -np.inf

        attention_weights = self.softmax(cosine_sim)

        weighted_u_l_m = t.mul(attention_weights.unsqueeze(-1), self.Bc(u_l_m).unsqueeze(1))

        alpha_j_1 = weighted_u_l_m.sum(2)
        z_j_1 = self.R(alpha_j_1)

        return z_j_1

    def forward_t(self, data):
        v = data.long()
        # v = v.cuda() if self.tvectors.weight.is_cuda else v
        return self.tvectors(v)

    def forward_c(self, data):
        v = data.long()
        # v = v.cuda() if self.cvectors.weight.is_cuda else v
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
                                                        (batch_sub_users - batch_tvecs).abs()], 2)))) + \
            self.ai2v.b_l_j[batch_titem_ids].unsqueeze(2)
        # return self.ai2v.cos_fin(batch_sub_users, batch_tvecs)

    def inference(self, user_items):
        num_items = self.ai2v.tvectors.weight.size()[0]
        citems = t.tensor([user_items])
        all_titems = t.tensor(range(num_items)).unsqueeze(0)
        sub_users = self.ai2v(all_titems, citems, batch_pad_ids=None, inference=True)
        all_tvecs = self.ai2v.Bt(self.ai2v.forward_t(all_titems))
        sim = self.similarity(sub_users, all_tvecs,all_titems)
        return sim.squeeze(-1).squeeze(0).detach().cpu().numpy()

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

        soft = sim.softmax(dim=1) + 1e-6
        return -soft[:, 0].log().sum()

