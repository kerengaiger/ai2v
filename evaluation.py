import torch as t
import pickle
import numpy as np
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=20, help="k for hr and mrr")
    parser.add_argument('--model', type=str, default='./output/i2v_best.pt', help="best model trained")
    parser.add_argument('--test', type=str, default='./data/test_ai2v_batch_u.dat', help="test set for evaluation")
    return parser.parse_args()


def mrr_k(model, eval_set, k):
    in_top_k, rec_rank = 0, 0

    pbar = tqdm(eval_set)

    for user_itemids, target_item in pbar:
        items_ranked = model.inference(user_itemids).argsort()
        top_k_items = items_ranked[-k:][::-1]
        if target_item in top_k_items:
            in_top_k += 1
            rec_rank += 1 / (np.where(top_k_items == target_item)[0][0] + 1)
    return rec_rank / in_top_k


def hr_k(model, eval_set, k):
    in_top_k = 0

    pbar = tqdm(eval_set)

    for user_itemids, target_item in pbar:
        items_ranked = model.inference(user_itemids).argsort()
        top_k_items = items_ranked[-k:][::-1]
        if target_item in top_k_items:
            in_top_k += 1
    return in_top_k / len(eval_set)


def main():
    print(t.cuda.current_device())
    device = t.cuda.current_device()
    print(t.cuda.memory_allocated(device))
    args = parse_args()
    model = t.load(args.model)
    eval_set = pickle.load(open(args.test, 'rb'))
    print(f'hit ratio at {args.k}:', hr_k(model, eval_set, args.k))
    print(f'mrr at {args.k}:', mrr_k(model, eval_set, args.k))


if __name__ == '__main__':
    main()
