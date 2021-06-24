import torch as t
from torch.utils.data import DataLoader
import pickle
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import pathlib

from scipy.stats import ttest_ind

from train_utils import UserBatchIncrementDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=20, help="k for hr and mrr")
    parser.add_argument('--window_size', type=int, default=1010, help="window size to pad all the rest")
    parser.add_argument('--data_dir', type=str, default='./data/', help="directory of all input data files")
    parser.add_argument('--model', type=str, default='./output/i2v_mix_batch__best.pt', help="best model trained")
    parser.add_argument('--test', type=str, default='test.dat', help="test set for evaluation")
    parser.add_argument('--batch_size', type=int, default=2000, help="batch size to iterate with when predicting")
    parser.add_argument('--hr_out', type=str, default='./output/hr_out.csv', help="hit at K for each test row")
    parser.add_argument('--mrr_out', type=str, default='./output/mrr_out.csv', help="hit at K for each test row")
    return parser.parse_args()


# def mrr_k(model, test_path, k, out_file, pad_idx, batch_size, window_size):
#     test_dataset = UserBatchIncrementDataset(test_path, pad_idx, window_size)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
#     pbar = tqdm(test_loader)
#
#     rec_rank = 0
#     for batch_titems, batch_citems in pbar:
#         all_titems = t.tensor(np.repeat(np.array([range(model.ai2v.tvectors.weight.size()[0])]), batch_size, axis=0))
#         if next(model.parameters()).is_cuda:
#             batch_citems, batch_titems = batch_citems.cuda(), batch_titems.cuda()
#             all_titems = all_titems.cuda()
#         mask_pad_ids = (batch_citems == pad_idx)
#         batch_sub_users = model.ai2v(all_titems, batch_citems, mask_pad_ids)
#         all_tvecs = model.ai2v.Bt(model.ai2v.forward_t(all_titems))
#         sim = model.similarity(batch_sub_users, all_tvecs, all_titems).squeeze()
#         items_ranked = t.argsort(sim, descending=True)[:, :k]
#         rec_rank += 1 / (items_ranked.eq(batch_titems.view(-1, 1)).nonzero()[:, 1] + 1)
#
#     return rec_rank / len(pbar)

def mrr_k(model, eval_set, k, out_file):
    rec_rank = 0

    pbar = tqdm(eval_set)

    with open(out_file, 'w') as file:
        for i, (user_itemids, target_item) in enumerate(pbar):
            items_ranked = model.inference(user_itemids).argsort()
            top_k_items = items_ranked[-k:][::-1]
            if target_item in top_k_items:
                rec_rank += 1 / (np.where(top_k_items == target_item)[0][0] + 1)
                file.write(f'{str(i)}, {target_item}, {rec_rank}')
                file.write('\n')
            else:
                file.write(f'{str(i)}, {target_item}, 0')
                file.write('\n')
    mrp_k = rec_rank / len(eval_set)
    return mrp_k


# def hr_k(model, test_path, k, out_file, pad_idx, batch_size, window_size):
#     test_dataset = UserBatchIncrementDataset(test_path, pad_idx, window_size)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
#     pbar = tqdm(test_loader)
#
#     in_top_k = 0
#     for batch_titems, batch_citems in pbar:
#         all_titems = t.tensor(np.repeat(np.array([range(model.ai2v.tvectors.weight.size()[0])]), batch_size, axis=0))
#         if next(model.parameters()).is_cuda:
#             batch_citems, batch_titems = batch_citems.cuda(), batch_titems.cuda()
#             all_titems = all_titems.cuda()
#         mask_pad_ids = (batch_citems == pad_idx)
#         batch_sub_users = model.ai2v(all_titems, batch_citems, mask_pad_ids)
#         all_tvecs = model.ai2v.Bt(model.ai2v.forward_t(all_titems))
#         sim = model.similarity(batch_sub_users, all_tvecs, all_titems).squeeze()
#         items_ranked = t.argsort(sim, descending=True)[:, :k]
#         in_top_k += items_ranked.eq(batch_titems.view(-1, 1)).sum()
#
#     return in_top_k / len(pbar)

def hr_k(model, eval_set, k, out_file):
    pbar = tqdm(eval_set)
    in_top_k = 0
    with open(out_file, 'w') as file:
        for i, (user_itemids, target_item) in enumerate(pbar):
            items_ranked = model.inference(user_itemids).argsort()
            top_k_items = items_ranked[-k:][::-1]
            if target_item in top_k_items:
                in_top_k += 1
                file.write(f'{str(i)}, {target_item}, 1')
                file.write('\n')
            else:
                file.write(f'{str(i)}, {target_item}, 0')
                file.write('\n')
    return in_top_k / len(eval_set)


def test_p_value(ai2v_file, i2v_file):
    '''
    :param ai2v_file: csv file containing u_id, item_id and the result of the tested metric, applied on ai2v model
    :param i2v_file: csv file containing u_id, item_id and the result of the tested metric, applied on i2v model
    :return: p_value of the paired_ttest between metric means of two models
    '''
    ai2v_pop = pd.read_csv(ai2v_file, header=None, names=['u_id', 'i_id', 'met'])
    i2v_pop = pd.read_csv(i2v_file, header=None, names=['u_id', 'i_id', 'met'])
    return ttest_ind(ai2v_pop['met'], i2v_pop['met'])[1]


def main():
    args = parse_args()
    model = t.load(args.model)
    model = t.nn.DataParallel(model)
    eval_set = pickle.load(open(args.test, 'rb'))
    item2idx = pickle.load(pathlib.Path(args.data_dir, 'item2idx.dat').open('rb'))
    # print(f'hit ratio at {args.k}:', hr_k(model.module, pathlib.Path(args.data_dir, args.test), args.k, args.hr_out,
    #                                       item2idx['pad'], args.batch_size, args.window_size))
    print(f'hit ratio at {args.k}:', hr_k(model.module, eval_set, args.k, args.hr_out))
    # print(f'mrr at {args.k}:', mrr_k(model.module, pathlib.Path(args.data_dir, args.test), args.k, args.mrr_out,
    #                                  item2idx['pad'], args.batch_size, args.window_size))
    print(f'mrr at {args.k}:', mrr_k(model.module, eval_set, args.k, args.mrr_out))


if __name__ == '__main__':
    main()
