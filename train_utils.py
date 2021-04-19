import pathlib
import pickle

import numpy as np
import torch as t


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
