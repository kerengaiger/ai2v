# -*- coding: utf-8 -*-

import os
import codecs
import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--vocab', type=str, default='./data/corpus_avi.txt', help="corpus path for building vocab")
    parser.add_argument('--full_corpus', type=str, default='./data/corpus_avi.txt', help="corpus path")
    parser.add_argument('--test_corpus', type=str, default='./data/test_corpus_avi.txt', help="test corpus path")
    parser.add_argument('--build_train_valid', action='store_true', help="build part train and validation sets from provided paths")
    parser.add_argument('--train_corpus', type=str, default='./data/train_corpus_avi.txt', help="part train corpus path to build part train set, in case build_train_valid is True")
    parser.add_argument('--valid_corpus', type=str, default='./data/valid_corpus_avi.txt', help="validation corpys path to build validation set, in case build_train_valid is True")
    parser.add_argument('--full_train_file', type=str, default='./data/full_train_batch_u.dat', help="full train file name")
    parser.add_argument('--train_file', type=str, default='./data/train_batch_u.dat', help="train file name, in case of build_valid")
    parser.add_argument('--valid_file', type=str, default='./data/valid_batch_u.dat', help="validation file name, in case of build_valid")
    parser.add_argument('--test_file', type=str, default='./data/test_batch_u.dat', help="test file of sub users and target items")
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token")
    parser.add_argument('--max_vocab', type=int, default=20000, help="maximum number of vocab")
    return parser.parse_args()


class Preprocess(object):

    def __init__(self, window=5, unk='<UNK>', pad='pad', data_dir='./data/'):
        self.window = window
        self.unk = unk
        self.pad = pad
        self.data_dir = data_dir

    def build(self, filepath, max_vocab=20000):
        print("building vocab...")
        step = 0
        self.wc = {}
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                step += 1
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000), end='\r')
                line = line.strip()
                if not line:
                    continue
                user = line.split()
                for item in user:
                    self.wc[item] = self.wc.get(item, 0) + 1
        print("")
        self.wc[self.unk] = 1
        self.wc[self.pad] = 1
        # sorted list of items in a descent order of their frequency
        self.idx2item = sorted(self.wc, key=self.wc.get, reverse=True)[:max_vocab - 1]
        self.item2idx = {self.idx2item[idx]: idx for idx, _ in enumerate(self.idx2item)}
        self.vocab = set([item for item in self.item2idx])
        pickle.dump(self.wc, open(os.path.join(self.data_dir, 'ic.dat'), 'wb'))
        pickle.dump(self.vocab, open(os.path.join(self.data_dir, 'vocab.dat'), 'wb'))
        pickle.dump(self.idx2item, open(os.path.join(self.data_dir, 'idx2item.dat'), 'wb'))
        pickle.dump(self.item2idx, open(os.path.join(self.data_dir, 'item2idx.dat'), 'wb'))
        print("build done")

    def convert(self, filepath, savepath):
        print("converting corpus...")
        step = 0
        data = []
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
                for item in line.split():
                    if item in self.vocab:
                        user.append(item)
                    else:
                        user.append(self.unk)
                if len(user) < 2:
                    print('skip user')
                    continue
                num_users += 1
                data.append([self.item2idx[item] for item in user])

        print("")
        pickle.dump(data, open(savepath, 'wb'))
        print("conversion done")
        print("num of users:", num_users)

    def process_test(self, test_path, savepath):
        print("processing test...")
        step = 0
        data = []
        with codecs.open(test_path, 'r', encoding='utf-8') as file:
            num_users = 0
            for line in file:
                step += 1
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000), end='\r')
                line = line.strip()
                if not line:
                    continue
                user = []
                for item in line.split():
                    if item in self.vocab:
                        user.append(item)
                    else:
                        user.append(self.unk)
                if len(user) < 2:
                    print('skip user')
                    continue
                num_users += 1
                # create list of [sub-user: lst[item ids], target_item_id]
                for i in range(1, len(user)):
                    sub_user = user[:i]
                    target_item = user[i]
                    data.append(([self.item2idx[item] for item in sub_user], self.item2idx[target_item]))

        print("")
        pickle.dump(data, open(savepath, 'wb'))
        print("conversion done")
        print("num of users:", num_users)


def main():
    args = parse_args()
    preprocess = Preprocess(unk=args.unk, data_dir=args.data_dir)
    preprocess.build(args.vocab, max_vocab=args.max_vocab)
    preprocess.convert(args.full_corpus, args.full_train_file)
    preprocess.process_test(args.test_corpus, args.test_file)
    if args.build_train_valid:
        preprocess.convert(args.train_corpus, args.train_file)
        preprocess.convert(args.valid_corpus, args.valid_file)


if __name__ == '__main__':
    main()
