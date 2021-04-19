import pathlib
import pickle
import random

import numpy as np
import torch as t
from torch.utils.data import Dataset


class UserBatchIncrementDataset(Dataset):
    def __init__(self, datapath, pad_idx, window_size, ws=None):
        data = pickle.load(datapath.open('rb'))
        if ws is not None:
            data_ws = []
            for citems, titem in data:
                if random.random() > ws[titem]:
                    data_ws.append((citems, titem))
            data = data_ws

        padded_data = []
        for sub_user in data:
            padded_data.append((sub_user[0] + [pad_idx for _ in range(window_size - len(sub_user[0]))], sub_user[1]))

        self.data = padded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        citems, titem = self.data[idx]
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
    t.save(sgns, pathlib.Path(cnfg['save_dir'], cnfg['model'] + '_best.pt'))



