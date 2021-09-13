import pickle
import pandas as pd
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='movielens', help="model to train: i2v or ai2v")
    parser.add_argument('--data_dir', type=str, default='./corpus/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./output/', help="model directory path")
    return parser.parse_args()


args = parse_args()
ic_file = f'{args.data_dir}/full_ic.dat'
item_cnts = pickle.load(open(ic_file, 'rb'))
itm_cnts_df = pd.DataFrame([item_cnts.keys(), item_cnts.values()]).T
itm_cnts_df.columns=['item_id', 'cnt']

# create hits file of popularity model
test_file = f'{args.data_dir}/test_raw.csv'
test = pd.read_csv(test_file, names=['user', 'item'])
itms_pop_rank = np.array([int(itm) for itm in itm_cnts_df.sort_values(by='cnt', ascending=False)['item_id'].tolist() if itm != '<UNK>' and itm != 'pad'])
test['pred_loc'] = 0
test['pred_loc'] = test.apply(lambda row: np.where(itms_pop_rank == row['item'])[0][0], axis=1)
test.to_csv(f'{args.save_dir}/{args.dataset}_pop_model_preds.csv', index=False)



