import torch as t
import pickle
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from scipy.stats import ttest_ind


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=20, help="k for hr and mrr")
    parser.add_argument('--model', type=str, default='./output/i2v_mix_batch__best.pt', help="best model trained")
    parser.add_argument('--test', type=str, default='./data/test.dat', help="test set for evaluation")
    return parser.parse_args()


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


def hr_k(model, eval_set, k, out_file):
    in_top_k = 0

    pbar = tqdm(eval_set)

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
    eval_set = pickle.load(open(args.test, 'rb'))
    print(f'hit ratio at {args.k}:', hr_k(model, eval_set, args.k, args.hr_out))
    print(f'mrr at {args.k}:', mrr_k(model, eval_set, args.k, args.rr_out))


if __name__ == '__main__':
    main()