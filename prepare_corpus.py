# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:44:14 2019

@author: t-avcaci
"""

import numpy as np
import time
import csv
from collections import Counter


class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.items = []

    def ArrangeItemList(self):
        self.items = [item_index[0] for item_index in sorted(self.items, key=lambda x: x[1])]


positive_threshold = 4.0
input_file = r'./data/netflix_corpus.csv'
line_sep = ','
min_usr_len = 1
max_usr_len = 60
min_items_cnt = 1
final_usr_len = 2
out_full_train = r'./data/corpus_netflix.txt'
out_test = r'./data/test_corpus_netflix.txt'
out_train = r'./data/train_corpus_netflix.txt'
out_valid = r'./data/valid_corpus_netflix.txt'

user2data = {}
t = time.clock()
with open(input_file) as rating_file:
    for i, line in enumerate(rating_file):
        if i == 0:
            continue
        if i % 5000000 == 0:
            print(i)
        line = line.strip().split(line_sep)
        line = [i for i in line if i != '']
        user_id = line[0]
        if user_id not in user2data:
            user2data[user_id] = User(user_id)
        user = user2data[user_id]

        if float(line[2]) > positive_threshold:
            user.items.append((line[1], int(line[-1])))

valid_users = []
for user in list(user2data.values()):
    if len(user.items) > min_usr_len and len(user.items) < max_usr_len:
        user.ArrangeItemList()
        valid_users.append(user.user_id)
print(len(valid_users))
print(time.clock() - t)


def IndexLabels(labels, mask_zero=False):
    label2index = {}
    index2label = {}
    for i, label in enumerate(labels):
        if mask_zero:
            i += 1
        label2index[label] = i
        index2label[i] = label
    return label2index, index2label


class Index(object):
    def __init__(self):
        self.item2index = {}
        self.index2item = {}


def MinCountFilter(counter, min_count=10):
    return [item for item, count in counter.most_common(len(counter)) if count > min_count]


np.random.seed(0)


def ComputeSplitIndices(num_instances, test_size=0.1):
    permutation = np.random.permutation(num_instances)
    split_index = int((1 - test_size) * num_instances)
    return permutation[:split_index], permutation[split_index:]


item_counter = Counter()
index = Index()

for user in list(valid_users):
    user = user2data[user]
    item_counter.update(user.items)

index.item2index, index.index2item = IndexLabels(MinCountFilter(item_counter, min_count=min_items_cnt), True)

valid_users_filtered = []
for user_id in list(valid_users):
    user = user2data[user_id]
    items = [item for item in user.items if item in index.item2index]
    if len(items) > final_usr_len:
        valid_users_filtered.append(user_id)
valid_users = valid_users_filtered


train_indices, test_indices = ComputeSplitIndices(len(valid_users), test_size=0.1)
train_users = [valid_users[i] for i in train_indices]
train_item_lists = [user2data[user].items for user in train_users]
test_users = [valid_users[i] for i in test_indices]
test_item_lists = [user2data[user].items for user in test_users]

with open(out_full_train, 'w', newline="") as x:
    csv.writer(x, delimiter=" ").writerows(train_item_lists)

with open(out_test, 'w', newline="") as x:
    csv.writer(x, delimiter=" ").writerows(test_item_lists)

train_indices, validation_indices = ComputeSplitIndices(len(train_indices), test_size=0.1)
train_users = [valid_users[i] for i in train_indices]
train_item_lists = [user2data[user].items for user in train_users]
validation_users = [valid_users[i] for i in validation_indices]
validation_item_lists = [user2data[user].items for user in validation_users]


with open(out_train, 'w', newline="") as x:
    csv.writer(x, delimiter=" ").writerows(train_item_lists)
with open(out_valid, 'w', newline="") as x:
    csv.writer(x, delimiter=" ").writerows(validation_item_lists)


print("Items#: ", len(index.item2index))
print("Full corpus users#:", len(valid_users))
print("Train users#: ", len(train_users))
print("validation users#: ", len(validation_users))
print("Test users#: ", len(test_users))
