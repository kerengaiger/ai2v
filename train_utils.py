import pathlib
import pickle
import random

import numpy as np
import torch as t
from torch.utils.data import Dataset


class UserBatchIncrementDataset(Dataset):
    def __init__(self, datapath, pad_idx, window_size, ws=None):
        data = pickle.load(datapath.open('rb'))
        self.pad_idx = pad_idx
        self.window_size = window_size

        if ws is not None:
            data_ws = []
            for citems, titem in data:
                if random.random() > ws[titem]:
                    data_ws.append((citems, titem))
            data = data_ws
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        len_samp = len(self.data[idx][0])
        pad_times = self.window_size - len_samp
        citems = self.data[idx][0] + [self.pad_idx] * pad_times
        titem = self.data[idx][1]
        # citems = self.data[idx][0]
        return titem, np.array(citems)


def configure_weights(cnfg, idx2item):
    ic = pickle.load(pathlib.Path(cnfg['data_dir'], 'ic.dat').open('rb'))

    ifr = np.array([ic[item] for item in idx2item])
    ifr = ifr / ifr.sum()

    assert (ifr > 0).all(), 'Items with invalid count appear.'
    istt = 1 - np.sqrt(cnfg['ss_t'] / ifr)
    istt = np.clip(istt, 0, 1)
    weights = istt if cnfg['weights'] else None
    return weights


def save_model(cnfg, model, sgns):
    tvectors = model.tvectors.weight.data.cpu().numpy()
    cvectors = model.cvectors.weight.data.cpu().numpy()

    pickle.dump(tvectors, open(pathlib.Path(cnfg['save_dir'], 'idx2tvec.dat'), 'wb'))
    pickle.dump(cvectors, open(pathlib.Path(cnfg['save_dir'], 'idx2cvec.dat'), 'wb'))
    t.save(sgns, pathlib.Path(cnfg['save_dir'], 'model.pt'))
