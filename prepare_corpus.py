# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:44:14 2019

@author: t-avcaci
"""

import numpy as np
import csv
from collections import Counter
from datetime import datetime
import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--item_pos', type=int, default=0, help="position of item in row")
    parser.add_argument('--usr_pos', type=int, default=1, help="position of user in row")
    parser.add_argument('--rate_pos', type=int, default=2, help="position of rating in row")
    parser.add_argument('--date_pos', type=int, default=3, help="position of date in row")
    parser.add_argument('--positive_threshold', type=float, default=4.0, help="threshold to consider positive items")
    parser.add_argument('--input_file', type=str, default='./data/corpus_netflix.txt', help="input file")
    parser.add_argument('--line_sep', type=str, default=',', help="line separator")
    parser.add_argument('--min_usr_len', type=int, default=3, help="minimum number of items per user")
    parser.add_argument('--max_usr_len', type=int, default=2700, help="maximum number of items per user")
    parser.add_argument('--min_items_cnt', type=int, default=100, help="minimum numbers of users per item")
    parser.add_argument('--max_items_cnt', type=int, default=130000, help="maximum numbers of users per item")
    parser.add_argument('--final_usr_len', type=int, default=130000, help="final minimum user length")
    parser.add_argument('--split_strategy', choices=['time_order', 'users_split'], help="way of splitting to train and test")
    parser.add_argument('--out_full_train', type=str, default='./data/corpus_netflix.txt', help="input file")
    parser.add_argument('--out_test', type=str, default='./data/test_corpus_netflix.txt', help="input file")
    parser.add_argument('--out_train', type=str, default='./data/train_corpus_netflix.txt', help="input file")
    parser.add_argument('--out_valid', type=str, default='./data/valid_corpus_netflix.txt', help="input file")
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


def split_usr_itms(usrs_lst, user2data, test_size=0.2):
    with open('train_usr_ids.pkl', 'wb') as f:
        pickle.dump(usrs_lst, f)
    train_usrs = usrs_lst[:int(len(usrs_lst) * test_size)]
    test_usrs = usrs_lst[int(len(usrs_lst) * test_size):]
    return train_usrs, [user2data[usr].items for usr in train_usrs], [user2data[usr].items for usr in test_usrs]


def main():
    args = parse_args()
    user2data = {}
    with open(args.input_file) as rating_file:
        for i, line in enumerate(rating_file):
            if i == 0:
                continue
            if i % 5000000 == 0:
                print(i)
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

    print('valid_users after pos: ', len(user2data.keys()))
    valid_users = []
    for user in list(user2data.values()):
        if args.min_usr_len < len(user.items) < args.max_usr_len:
            user.ArrangeItemList()
            valid_users.append(user.user_id)

    np.random.seed(0)
    print('valid_users after first filter: ', len(valid_users))

    item_counter = Counter()
    index = Index()

    for user in list(valid_users):
        user = user2data[user]
        item_counter.update(user.items)

    index.item2index, index.index2item = IndexLabels(CountFilter(item_counter, min_count=args.min_items_cnt,
                                                                 max_count=args.max_items_cnt), True)

    print('items cnt after filter: ', len(item_counter.keys()))
    valid_users_filtered = []
    for user_id in list(valid_users):
        user = user2data[user_id]
        items = [item for item in user.items if item in index.item2index]
        if len(items) > args.final_usr_len:
            valid_users_filtered.append(user_id)
    valid_users = valid_users_filtered
    print('users cnt after final filter: ', len(valid_users_filtered))

    if args.split_strategy == 'users_split':
        full_train_users, full_train_item_lsts, test_item_lsts = split_usrs(valid_users, user2data)
        train_users, train_item_lsts, validation_item_lsts = split_usrs(full_train_users, user2data)
    else:
        full_train_users, full_train_item_lsts, test_item_lsts = split_usr_itms(valid_users, user2data)
        train_users, train_item_lsts, validation_item_lsts = split_usrs(full_train_users, user2data)

    with open(args.out_full_train, 'w', newline="") as x:
        csv.writer(x, delimiter=" ").writerows(full_train_item_lsts)
    with open(args.out_test, 'w', newline="") as x:
        csv.writer(x, delimiter=" ").writerows(test_item_lsts)
    with open(args.out_train, 'w', newline="") as x:
        csv.writer(x, delimiter=" ").writerows(train_item_lsts)
    with open(args.out_valid, 'w', newline="") as x:
        csv.writer(x, delimiter=" ").writerows(validation_item_lsts)

    print("Items#: ", len(index.item2index))
    print("Full corpus users#:", len(valid_users))
    print("Train users#: ", len(train_users))


if __name__ == '__main__':
    main()
