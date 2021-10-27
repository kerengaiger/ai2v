import torch as t
import pickle
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import torch

from scipy.stats import ttest_ind

import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=20, help="k for hr and mrr")
    parser.add_argument('--data_dir', type=str, default='./data/', help="directory of all input data files")
    parser.add_argument('--output_dir', type=str, default='./output/', help="output directory")
    parser.add_argument('--model', type=str, default='model.pt', help="best model trained")
    parser.add_argument('--rank', action='store_true', help="output ranked items list")
    parser.add_argument('--test', type=str, default='test.dat', help="test set for evaluation")
    parser.add_argument('--test_raw', type=str, default='test_raw.csv', help="file containing raw usr and item test ids")
    parser.add_argument('--preds_out', type=str, default='preds_out.csv', help="ranked list of items for test users")
    parser.add_argument('--rank_out', type=str, default='rank_out.pkl', help="ranked list of items for test users")
    parser.add_argument('--hr_out', type=str, default='hr_out.csv', help="hit at K for each test row")
    parser.add_argument('--mrr_out', type=str, default='mrr_out.csv', help="hit at K for each test row")
    parser.add_argument('--mpr_out', type=str, default='mpr_out.csv', help="percentile for each test row")
    return parser.parse_args()


def mrr_k(preds_df, k, out_file):
    preds_df['rr_k'] = 1 / preds_df['pred_loc']
    preds_df.loc[preds_df['pred_loc'] > k, 'rr_k'] = 0
    preds_df.to_csv(out_file, index=False)
    return preds_df['rr_k'].mean()


def hr_k(preds_df, k, out_file):
    preds_df['hit'] = 0
    preds_df.loc[preds_df['pred_loc'] <= k, 'hit'] = 1
    preds_df.to_csv(out_file, index=False)
    return preds_df['hit'].mean()


def mpr(preds_df, num_all_items):
    return 1 - (preds_df['pred_loc'] / num_all_items).mean()


def ndcg_k(preds_df, k):
    preds_df['ndcg_k'] = 1 / np.log2(1 + preds_df['pred_loc'])
    preds_df.loc[preds_df['pred_loc'] > k, 'ndcg_k'] = 0
    return preds_df['ndcg_k'].mean()


def predict(model, eval_set_lst, eval_set_df, out_file):
    pbar = tqdm(eval_set_lst)
    model.eval()
    eval_set_df['pred_loc'] = np.nan
    for i, (user_itemids, target_item) in enumerate(pbar):
        items_ranked = model.inference(user_itemids).argsort()
        all_items = items_ranked[:][::-1]
        loc = np.where(all_items == target_item)[0][0] + 1
        eval_set_df.loc[i, 'pred_loc'] = loc

    eval_set_df.to_csv(out_file, index=False)


def calc_attention(model, eval_set_lst, add_pos_bias, out_file):
    pbar = tqdm(eval_set_lst)
    lst = []
    for i, (user_items, target_item) in enumerate(pbar):
        if len(user_items) < model.ai2v.window_size:
            pad_times = model.ai2v.window_size - len(user_items)
            user_items = [model.ai2v.pad_idx] * pad_times + user_items
        else:
            user_items = user_items[-model.ai2v.window_size:]

        batch_titems = torch.tensor([target_item]).unsqueeze(0)
        batch_titems = batch_titems.to(model.device)
        batch_citems = torch.tensor([user_items])
        batch_citems = batch_citems.to(model.device)
        mask_pad_ids = batch_citems == model.ai2v.pad_idx
        if not add_pos_bias:
            model.ai2v.add_pos_bias = False
        else:
            model.ai2v.add_pos_bias = True
        _, attention_weights = model.ai2v(batch_titems, batch_citems, mask_pad_ids=mask_pad_ids)
        lst.append(attention_weights[0][0].cpu().detach().numpy())
    pickle.dump(lst, open(out_file, 'wb'))


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
    model = t.load(os.path.join(args.output_dir, args.model))
    eval_set_lst = pickle.load(open(os.path.join(args.data_dir, args.test), 'rb'))
    eval_set_df = pd.read_csv(os.path.join(args.data_dir, args.test_raw), names=['usr', 'itm'])
    predict(model, eval_set_lst, eval_set_df, os.path.join(args.output_dir, args.preds_out))
    preds_df = pd.read_csv(os.path.join(args.output_dir, args.preds_out))
    print(f'hit ratio at {args.k}:', hr_k(preds_df, args.k, os.path.join(args.output_dir, args.hr_out)))
    print(f'mrr at {args.k}:', mrr_k(preds_df, args.k, os.path.join(args.output_dir, args.hr_out)))
    print(f'ndcg at {args.k}:', ndcg_k(preds_df, args.k))
    print(f'mpr:', mpr(preds_df, model.vocab_size))


if __name__ == '__main__':
    main()
