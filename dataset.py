import os
import codecs
import pickle
import datetime
import numpy as np
import pandas as pd
import json
import csv


def random_split(lst, frac=0.2):
    np.random.seed(0)
    return np.random.choice(lst, int(frac * len(lst)))


def split_usr_itms(itms_lsts):
    return [usr_itms[:-1] for usr_itms in itms_lsts], itms_lsts


def filter(data_dict, min_count, max_count):
    return {k: data_dict[k] for k in data_dict.keys() if min_count < len(data_dict[k]) < max_count}


class Preprocess(object):

    def __init__(self, unk='<UNK>', pad='pad', raw_data_file='./data/', save_data_dir='./corpus/', line_sep=',',
                 pos_thresh=4, user_pos=0, item_pos=1, rate_pos=2, date_pos=3, min_usr_len=2, max_usr_len=1000,
                 min_items_cnt=5, max_items_cnt=50000, final_usr_len=4, split_strategy='leave_one_out'):
        self.unk = unk
        self.pad = pad
        self.wc = {}
        self.idx2item = list()
        self.item2idx = dict()
        self.user2idx = dict()
        self.vocab = set()
        self.line_sep = line_sep
        self.pos_thresh = pos_thresh
        self.user_pos = user_pos
        self.item_pos = item_pos
        self.rate_pos = rate_pos
        self.date_pos = date_pos
        self.min_usr_len = min_usr_len
        self.max_usr_len = max_usr_len
        self.min_items_cnt = min_items_cnt
        self.max_items_cnt = max_items_cnt
        self.final_usr_len = final_usr_len
        self.split_strategy = split_strategy
        self.raw_data_file = raw_data_file
        self.save_data_dir = save_data_dir

    def build(self, filepath, ic_out, vocab_out, idx2item_out, item2idx_out, idx2user_out, user2idx_out):
        print("building vocab...")
        step = 0
        self.idx2user = list()
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                step += 1
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000), end='\r')
                line = line.strip()
                if not line:
                    continue
                user = line.split()
                self.idx2user.append(user[0])
                for item in user[1:]:
                    self.wc[item] = self.wc.get(item, 0) + 1
        print("")
        # sorted list of items in a descent order of their frequency
        self.wc[self.unk] = 1
        self.wc[self.pad] = 1
        self.idx2item = sorted(self.wc, key=self.wc.get, reverse=True)
        self.item2idx = {self.idx2item[i_idx]: i_idx for i_idx, _ in enumerate(self.idx2item)}
        self.user2idx = {self.idx2user[u_idx]: u_idx for u_idx, _ in enumerate(self.idx2user)}
        self.vocab = set([item for item in self.item2idx])
        pickle.dump(self.wc, open(os.path.join(self.save_data_dir, ic_out), 'wb'))
        pickle.dump(self.vocab, open(os.path.join(self.save_data_dir, vocab_out), 'wb'))
        pickle.dump(self.idx2item, open(os.path.join(self.save_data_dir, idx2item_out), 'wb'))
        pickle.dump(self.item2idx, open(os.path.join(self.save_data_dir, item2idx_out), 'wb'))
        pickle.dump(self.idx2user, open(os.path.join(self.save_data_dir, idx2user_out), 'wb'))
        pickle.dump(self.user2idx, open(os.path.join(self.save_data_dir, user2idx_out), 'wb'))
        print("build done")

    def create_train_samp(self, user, item_target):
        sub_user = user[:item_target]
        target_item = user[item_target]
        return [self.item2idx[item] for item in sub_user], self.item2idx[target_item]

    def convert(self, filepath, savepath):
        print("converting corpus...")
        step = 0
        data = []
        usrs_len = []
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            num_users = 0
            for line in file:
                step += 1
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000), end='\r')
                line = line.strip()
                if not line:
                    continue
                user = []
                line_split = line.split()
                for item in line_split[1:]:
                    if item in self.vocab:
                        user.append(item)
                    else:
                        user.append(self.unk)
                usrs_len.append(len(user))
                num_users += 1
                for item_target in range(1, len(user)):
                    if self.split_strategy == 'leave_one_out' and item_target < (len(user) - 1):
                        # in case we want to have as validation/test only the last item with sub-user of all the
                        # rest, skip all other target items
                        continue
                    user_idx = self.user2idx[line_split[0]]
                    citems, titem = self.create_train_samp(user, item_target)
                    data.append((user_idx, citems, titem))

        print("")
        pickle.dump(data, open(savepath, 'wb'))
        print("conversion done")
        print("num of users:", num_users)
        print("max user:", max(usrs_len))

    def read_data(self, raw_file):
        user2data = {}
        item2data = {}
        with open(raw_file) as rating_file:
            for i, line in enumerate(rating_file):
                if i % 5000000 == 0:
                    print(i)
                if line != "\n":
                    line = line.strip().split(self.line_sep)
                    line = [i for i in line if i != '']
                    user_id = line[self.user_pos]
                    if user_id not in user2data:
                        user2data[user_id] = []
                    item_id = line[self.item_pos]
                    if item_id not in item2data:
                        item2data[item_id] = []
                    try:
                        date = int(line[self.date_pos])
                    except:
                        date = int(datetime.strptime(line[self.date_pos], '%Y-%m-%d').timestamp())
                    if float(line[self.rate_pos]) > self.pos_thresh:
                        user2data[user_id].append((line[self.item_pos], date))
                        item2data[item_id].append(line[self.user_pos])

        return user2data, item2data

    def split(self, users):
        if self.split_strategy == 'users_split':
            train_users = random_split(list(users.keys()))
            full_train, test = {user: users[user] for user in train_users}, \
                               {user: users[user] for user in list(users.keys()) if user not in train_users}
            train_users = random_split(list(full_train.keys()))
            train, valid = {user: full_train[user] for user in train_users}, \
                           {user: full_train[user] for user in list(users.keys()) if user not in train_users}

        elif self.split_strategy == 'leave_one_out':
            full_train, test = {user: users[user][:-1] for user in users}, users
            train, valid = {user: full_train[user][:-1] for user in full_train}, full_train
        else:
            print('Split strategy not valid')
            return

        return [[user] + full_train[user] for user in list(full_train.keys())], \
               [[user] + train[user] for user in list(train.keys())], \
               [[user] + valid[user] for user in list(valid.keys())], \
               [[user] + test[user] for user in list(test.keys())]

    def save_file(self, file_name, data):
        with open(os.path.join(self.save_data_dir, file_name), 'w', newline="") as x:
            csv.writer(x, delimiter=" ").writerows(data)


def generate_train_files(data_cnfg):
    with open(data_cnfg) as f:
        params = json.load(f)
    preprocess = Preprocess(**params)
    user2data, item2data = preprocess.read_data(preprocess.raw_data_file)
    # filter user and items
    user2data = filter(user2data, preprocess.min_usr_len, preprocess.max_usr_len)
    item2data = {item: [user for user in item2data[item] if user in user2data.keys()] for item in item2data.keys()}
    item2data = filter(item2data, preprocess.min_items_cnt, preprocess.max_items_cnt)
    user2data = {user: [item for item in user2data[user] if item[0] in item2data.keys()] for user in user2data.keys()}
    user2data = filter(user2data, preprocess.final_usr_len, preprocess.max_usr_len)
    # arrange users data by date
    user2data = {usr: [item_index[0] for item_index in sorted(user2data[usr], key=lambda x: x[1])] for usr in user2data.keys()}
    # generate processed raw files
    full_corpus = [[user] + user2data[user] for user in user2data.keys()]
    pd.DataFrame({'user': list(user2data.keys()), 'item': [user2data[usr][-1] for usr in user2data.keys()]}).to_csv(
        os.path.join(preprocess.save_data_dir, 'test_raw.csv'), header=False, index=False)
    full_train, train, valid, test = preprocess.split(user2data)
    preprocess.save_file('full_corpus.txt', full_corpus)
    preprocess.save_file('full_train.txt', full_train)
    preprocess.save_file('train.txt', train)
    preprocess.save_file('valid.txt', valid)
    preprocess.save_file('test.txt', test)

    # generate final train files
    preprocess.build(os.path.join(preprocess.save_data_dir, 'full_corpus.txt'), 'full_ic.dat', 'full_vocab.dat',
                     'full_idx2item.dat', 'full_item2idx.dat', 'full_idx2user.dat', 'full_user2idx.dat')
    preprocess.build(os.path.join(preprocess.save_data_dir, 'full_train.txt'), 'ic.dat', 'vocab.dat',
                     'idx2item.dat', 'item2idx.dat', 'idx2user.dat', 'user2idx.dat')
    print("Full train")
    preprocess.convert(os.path.join(preprocess.save_data_dir, 'full_train.txt'),
                       os.path.join(preprocess.save_data_dir, 'full_train.dat'))
    print("Test")
    preprocess.convert(os.path.join(preprocess.save_data_dir, 'test.txt'),
                       os.path.join(preprocess.save_data_dir, 'test.dat'))

    print("Train")
    preprocess.convert(os.path.join(preprocess.save_data_dir, 'train.txt'),
                       os.path.join(preprocess.save_data_dir, 'train.dat'))
    print("valid")
    preprocess.convert(os.path.join(preprocess.save_data_dir, 'valid.txt'),
                       os.path.join(preprocess.save_data_dir, 'valid.dat'))
