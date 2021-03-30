import torch as t
import pickle
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity


def represent_user(user_itemids, model):
    context_vecs = model.embedding.ivectors.weight.data.cpu().numpy()
    user2vec = context_vecs[user_itemids, :].mean(axis=0)
    return user2vec


def mrr_k(model, eval_set, k):
    in_top_k, rec_rank = 0, 0
    for user_itemids, target_item in eval_set:
        items_ranked = inference(model, user_itemids)
        top_k_items = items_ranked.argsort()[-k:][::-1]
        if target_item in top_k_items:
            in_top_k += 1
            rec_rank += 1 / (np.where(top_k_items == target_item)[0][0] + 1)
    mrp_k = rec_rank / in_top_k
    return mrp_k


def inference(model, user_itemids):
    user2vec = represent_user(user_itemids, model)
    user_sim = cosine_similarity(user2vec, model.embedding.ovectors.weight.data.cpu().numpy()).squeeze()
    return user_sim.argsort()


def hr_k(model, eval_set, k):
    in_top_k = 0
    for user_itemids, target_item in eval_set:
        items_ranked = inference(model, user_itemids)
        top_k_items = items_ranked.argsort()[-k:][::-1]
        if target_item in top_k_items:
            in_top_k += 1
    return in_top_k / len(eval_set)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=20, help="k for hr and mrr")
    parser.add_argument('--model', type=str, default='./data/best_model.pt', help="best model trained")
    parser.add_argument('--test', type=str, default='./data/test.dat', help="test set for evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    model = t.load(args.model)
    eval_set = pickle.load(open(args.test, 'rb'))
    print(f'hit ratio at {args.k}:', hr_k(model, eval_set, args.k))


if __name__ == '__main__':
    main()
