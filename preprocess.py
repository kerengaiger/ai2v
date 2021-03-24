# -*- coding: utf-8 -*-

import os
import codecs
import pickle
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--vocab', type=str, default='./data/corpus.txt', help="corpus path for building vocab")
    parser.add_argument('--full_corpus', type=str, default='./data/corpus.txt', help="corpus path")
    parser.add_argument('--train_corpus', type=str, default='./data/train_corpus.txt', help="train corpus path")
    parser.add_argument('--full_train_file', type=str, default='./data/full_train.dat', help="full train file name")
    parser.add_argument('--train_file', type=str, default='./data/train.dat', help="train file name")
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token")
    parser.add_argument('--max_vocab', type=int, default=20000, help="maximum number of vocab")
    return parser.parse_args()


class Preprocess(object):

    def __init__(self, window=5, unk='<UNK>', data_dir='./data/'):
        self.window = window
        self.unk = unk
        self.data_dir = data_dir

    def build(self, filepath, max_vocab=20000):
        print("building vocab...")
        step = 0
        self.wc = {self.unk: 1}
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
        # sorted list of items in a descent order of their frequency
        self.wc['pad'] = 1
        self.idx2item = [self.unk] + sorted(self.wc, key=self.wc.get, reverse=True)[:max_vocab - 1]
        # self.idx2item.append('pad')
        self.item2idx = {self.idx2item[idx]: idx for idx, _ in enumerate(self.idx2item)}
        # self.item2idx['pad'] = len(self.idx2item)
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


def main():
    args = parse_args()
    preprocess = Preprocess(window=args.window, unk=args.unk, data_dir=args.data_dir)
    preprocess.build(args.vocab, max_vocab=args.max_vocab)
    preprocess.convert(args.full_corpus, args.full_train_file)
    preprocess.convert(args.train_corpus, args.train_file)


if __name__ == '__main__':
    main()
