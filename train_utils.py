import pathlib
import pickle
import random

import numpy as np
import torch as t
from torch.optim import Adagrad
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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
    if model.name == 'i2v':
        ivectors = model.ivectors.weight.data.cpu().numpy()
        ovectors = model.ovectors.weight.data.cpu().numpy()
    else:
        ivectors = model.tvectors.weight.data.cpu().numpy()
        ovectors = model.cvectors.weight.data.cpu().numpy()

    pickle.dump(ivectors, open(pathlib.Path(cnfg['save_dir'], 'idx2ivec.dat'), 'wb'))
    pickle.dump(ovectors, open(pathlib.Path(cnfg['save_dir'], 'idx2ovec.dat'), 'wb'))
    t.save(sgns, pathlib.Path(cnfg['save_dir'], 'best_model.pt'))
