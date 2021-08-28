# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:44:14 2019

@author: t-avcaci
"""

import numpy as np
import pandas as pd
import csv
from collections import Counter
from datetime import datetime
import argparse
import pickle
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--usr_pos', type=int, default=0, help="position of user in row")
    parser.add_argument('--item_pos', type=int, default=1, help="position of item in row")
    parser.add_argument('--rate_pos', type=int, default=2, help="position of rating in row")
    parser.add_argument('--date_pos', type=int, default=3, help="position of date in row")
    parser.add_argument('--positive_threshold', type=float, default=4.0, help="threshold to consider positive items")
    parser.add_argument('--stats_dir', type=str, default='./stats/', help="directory to save stats of created corpus")
    parser.add_argument('--input_file', type=str, default='./data/corpus_netflix.txt', help="input file")
    parser.add_argument('--line_sep', type=str, default=',', help="line separator")
    parser.add_argument('--min_usr_len', type=int, default=3, help="minimum number of items per user")
    parser.add_argument('--max_usr_len', type=int, default=2700, help="maximum number of items per user")
    parser.add_argument('--min_items_cnt', type=int, default=100, help="minimum numbers of users per item")
    parser.add_argument('--max_items_cnt', type=int, default=130000, help="maximum numbers of users per item")
    parser.add_argument('--final_usr_len', type=int, default=4, help="final minimum user length")
    parser.add_argument('--split_strategy', choices=['leave_one_out', 'users_split'], default='users_split',
                        help="way of splitting to train and test")
    parser.add_argument('--data_dir', type=str, default='./corpus/netflix/', help="data_dir")
    parser.add_argument('--out_full_corpus', type=str, default='full_corpus.txt', help="output file")
    parser.add_argument('--out_full_train', type=str, default='full_train.txt', help="output file")
    parser.add_argument('--out_test', type=str, default='test.txt', help="test lists")
    parser.add_argument('--out_test_raw', type=str, default='test_raw.csv', help="file containing raw usr and item test ids")
    parser.add_argument('--out_corpus_raw', type=str, default='corpus_raw.csv', help="file containing raw usr and item corpus ids")
    parser.add_argument('--out_train', type=str, default='train.txt', help="train lists")
    parser.add_argument('--out_valid', type=str, default='valid.txt', help="validation lists")
    return parser.parse_args()


class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.items = []

    def ArrangeItemList(self):
        self.items = [item_index[0] for item_index in sorted(self.items, key=lambda x: x[1])]


class Index(object):
    def __init__(self):
        self.item2index = {}
        self.index2item = {}


def IndexLabels(labels, mask_zero=False):
    label2index = {}
    index2label = {}
    for i, label in enumerate(labels):
        if mask_zero:
            i += 1
        label2index[label] = i
        index2label[i] = label
    return label2index, index2label


def CountFilter(counter, min_count=10, max_count=10000000):
    return [item for item, count in counter.most_common(len(counter)) if count > min_count and count < max_count]


def ComputeSplitIndices(num_instances, test_size=0.1):
    permutation = np.random.permutation(num_instances)
    split_index = int((1 - test_size) * num_instances)
    return permutation[:split_index], permutation[split_index:]


def split_usrs(usrs_lst, user2data, test_size=0.2):
    train_indices, test_indices = ComputeSplitIndices(len(usrs_lst), test_size=test_size)
    train_users = [usrs_lst[i] for i in train_indices]
    test_users = [usrs_lst[i] for i in test_indices]
    return train_users, [user2data[user].items for user in train_users], [user2data[user].items for user in test_users]


def split_usr_itms(itms_lsts):
    return [usr_itms[:-1] for usr_itms in itms_lsts], itms_lsts


def main():
    args = parse_args()
    user2data = {}
    with open(args.input_file) as rating_file:
        for i, line in enumerate(rating_file):
            if i % 5000000 == 0:
                print(i)
            if line != "\n":
                line = line.strip().split(args.line_sep)
                line = [i for i in line if i != '']
                user_id = line[args.usr_pos]
                if user_id not in user2data:
                    user2data[user_id] = User(user_id)
                user = user2data[user_id]
                # treat date format
                try:
                    date = int(line[args.date_pos])
                except:
                    date = int(datetime.strptime(line[args.date_pos], '%Y-%m-%d').timestamp())
                if float(line[args.rate_pos]) > args.positive_threshold:
                    user.items.append((line[args.item_pos], date))

    valid_users = []
    for user in list(user2data.values()):
        if args.min_usr_len < len(user.items) < args.max_usr_len:
            user.ArrangeItemList()
            valid_users.append(user.user_id)

    np.random.seed(0)

    item_counter = Counter()
    index = Index()

    for user in list(valid_users):
        user = user2data[user]
        item_counter.update(user.items)

    index.item2index, index.index2item = IndexLabels(CountFilter(item_counter, min_count=args.min_items_cnt,
                                                                 max_count=args.max_items_cnt), True)

    valid_users_filtered = []
    for user_id in list(valid_users):
        user = user2data[user_id]
        items = [item for item in user.items if item in index.item2index]
        if len(items) >= args.final_usr_len:
            valid_users_filtered.append(user_id)
            user2data[user_id].items = items

    valid_users = valid_users_filtered

    itms_lsts = [user2data[usr].items for usr in valid_users]
    unique_items = list(set([item for sublist in itms_lsts for item in sublist]))

    if args.split_strategy == 'users_split':
        full_train_users, full_train_item_lsts, test_item_lsts = split_usrs(valid_users, user2data)
        train_users, train_item_lsts, validation_item_lsts = split_usrs(full_train_users, user2data)
    elif args.split_strategy == 'leave_one_out':
        full_train_item_lsts, test_item_lsts = split_usr_itms(itms_lsts)
        pd.DataFrame({'usr': valid_users, 'itm': [usr[-1] for usr in itms_lsts]}).to_csv(
            os.path.join(args.data_dir, args.out_test_raw), header=False, index=False)
        users_train = [[usr] * len(itms_lst) for usr, itms_lst in zip(valid_users, itms_lsts)]
        users_train = [usr for sublist in users_train for usr in sublist]
        items_train = [item for sublist in itms_lsts for item in sublist]
        pd.DataFrame({'usr': users_train, 'itm': items_train}).to_csv(
            os.path.join(args.data_dir, args.out_corpus_raw), header=False, index=False)
        train_item_lsts, validation_item_lsts = split_usr_itms(full_train_item_lsts)
    else:
        print('Split strategy not valid')
        return

    print("Items#: ", len(unique_items))
    print("Full corpus users#:", len(valid_users))

    with open(os.path.join(args.stats_dir, args.input_file.split('/')[-1]), 'w', newline="") as x:
        csv.writer(x, delimiter=',').writerows([['# users', '# items', '# samples'],
                                                [str(len(valid_users)), str(len(unique_items)),
                                                 str(sum([len(usr) for usr in itms_lsts]))]])
    with open(os.path.join(args.data_dir, args.out_full_corpus), 'w', newline="") as x:
        csv.writer(x, delimiter=" ").writerows(itms_lsts)
    with open(os.path.join(args.data_dir, args.out_full_train), 'w', newline="") as x:
        csv.writer(x, delimiter=" ").writerows(full_train_item_lsts)
    with open(os.path.join(args.data_dir, args.out_test), 'w', newline="") as x:
        csv.writer(x, delimiter=" ").writerows(test_item_lsts)
    with open(os.path.join(args.data_dir, args.out_train), 'w', newline="") as x:
        csv.writer(x, delimiter=" ").writerows(train_item_lsts)
    with open(os.path.join(args.data_dir, args.out_valid), 'w', newline="") as x:
        csv.writer(x, delimiter=" ").writerows(validation_item_lsts)


if __name__ == '__main__':
    main()
