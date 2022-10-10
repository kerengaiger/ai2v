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
        self.u_id_pos = 0
        self.citems_pos = 1
        self.titem_pos = 2

        if ws is not None:
            data_ws = []
            for u_id, citems, titem in data:
                if random.random() > ws[titem]:
                    data_ws.append((u_id, citems, titem))
            data = data_ws
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        len_samp = len(self.data[idx][self.citems_pos])
        if self.window_size > len_samp:
            pad_times = self.window_size - len_samp
            citems = [self.pad_idx] * pad_times + self.data[idx][self.citems_pos]
        else:
            citems = self.data[idx][self.citems_pos][-self.window_size:]
        titem = self.data[idx][self.titem_pos]
        user_id = self.data[idx][self.u_id_pos]
        return user_id, titem, np.array(citems)


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


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    t.random.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
