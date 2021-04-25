# -*- coding: utf-8 -*-

import numpy as np
import torch as t
import torch.nn as nn

from torch import FloatTensor as FT


class AttentiveItemToVec(nn.Module):
    def __init__(self, vocab_size=20000, embedding_size=300, d_alpha=40, N=1, padding_idx=0):
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
        # print('v_l_j', v_l_j.shape)
        # print('u_l_m', u_l_m.shape)
        c_vecs = self.Ac(u_l_m).unsqueeze(1)
        t_vecs = self.At(v_l_j).unsqueeze(2)
        # print('c_vecs', c_vecs.shape)
        # print('t_vecs', t_vecs.shape)
        # print((c_vecs == 0).nonzero(), 'c_vecs zeros')
        # print(c_vecs.max(), 'c_vecs max')
        # print((t_vecs == 0).nonzero(), 't_vecs zeros')
        # print(t_vecs.max(), 't_vecs max')
        cosine_sim = self.cos(t_vecs, c_vecs)
        batch_pad_ids = (batch_pad_ids[0].repeat_interleave(batch_titems.shape[1]),
                         t.cat([t.tensor(range(batch_titems.shape[1]))] * batch_pad_ids[0].shape[0]),
                         batch_pad_ids[1].repeat_interleave(batch_titems.shape[1]))

        # print('cosine sim', cosine_sim.shape)
        cosine_sim[batch_pad_ids] = -np.inf

        # print((cosine_sim == 0).nonzero(), 'cosine_sim zeros')
        # print(cosine_sim.max(), 'cosine_sim max')
        attention_weights = self.softmax(cosine_sim)
        # print('attention weights', attention_weights)

        # print((attention_weights == 0).nonzero(), 'attention_weights zeros')
        # print(attention_weights.max(), 'attention_weights max')
        # print('Bc(u_l_m)', self.Bc(u_l_m))
        weighted_u_l_m = t.mul(attention_weights.unsqueeze(-1), self.Bc(u_l_m).unsqueeze(1))
        # print('weighted_u_l_m', weighted_u_l_m)

        # print((weighted_u_l_m == 0).nonzero(), 'weighted_u_l_m zeros')
        # print(weighted_u_l_m.max(), 'weighted_u_l_m max')
        alpha_j_1 = weighted_u_l_m.sum(2)
        # print('alpha_j_1', alpha_j_1)
        # print((alpha_j_1 == 0).nonzero(), 'alpha_j_1 zeros')
        # print(alpha_j_1.max(), 'alpha_j_1 max')
        z_j_1 = self.R(alpha_j_1)
        # print((z_j_1 == 0).nonzero(), 'z_j_1 zeros')
        # print(z_j_1.max(), 'z_j_1 max')
        # print('z_j_1', z_j_1)

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
        citems = t.cat(num_items * [t.tensor([user_items])])
        all_titems = t.tensor(range(num_items))
        sub_user = self.represent_user(citems, all_titems)
        all_tvecs = self.ai2v.Bt(self.ai2v.forward_t(all_titems))
        sim = self.similarity(sub_user, all_tvecs, all_titems)
        return sim.squeeze().detach().cpu().numpy()

    def forward(self, batch_titems, batch_citems, batch_pad_ids):
        batch_size = batch_titems.size()[0]
        if self.weights is not None:
            batch_nitems = t.multinomial(self.weights, batch_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            batch_nitems = FT(batch_size, self.n_negs).uniform_(0, self.vocab_size - 1).long()

        batch_titems = t.cat([batch_titems.reshape(-1, 1), batch_nitems], 1)
        batch_sub_users = self.ai2v(batch_titems, batch_citems, batch_pad_ids)
        # print('batch_sub_users', batch_sub_users.shape)
        # print((batch_sub_users == 0).nonzero(), 'batch_sub_users zeros')
        # print(batch_sub_users.max(), 'batch_sub_users max')
        batch_tvecs = self.ai2v.Bt(self.ai2v.forward_t(batch_titems))
        # print('batch_tvecs', batch_tvecs)
        # print((batch_tvecs == 0).nonzero(), 'batch_tvecs zeros')
        # print(batch_tvecs.max(), 'batch_tvecs max')

        if [param for param in self.ai2v.parameters()][0].is_cuda:
            self.ai2v.b_l_j.cuda()

        # print(self.ai2v.b_l_j.max(), 'b_l_j max')
        # print(self.ai2v.b_l_j, 'b_l_j')
        sim = self.similarity(batch_sub_users, batch_tvecs, batch_titems)
        # print('sim', sim.shape)
        # print(sim.max(), 'sim max')

        return -sim.squeeze(-1).softmax(dim=1)[:, 0].log().sum()
