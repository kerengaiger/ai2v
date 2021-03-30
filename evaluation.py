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
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./output/', help="model directory path")
    parser.add_argument('--train', type=str, default='train.dat', help="train file name")
    parser.add_argument('--valid', type=str, default='valid_avi.dat', help="validation users file name")