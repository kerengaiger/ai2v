import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import argparse
import pickle
import pathlib

from ax.service.managed_loop import optimize

from train_i2v import train_evaluate as train_evaluate_i2v
from train_i2v import train as train_i2v
from train_ai2v import train_evaluate as train_evaluate_ai2v
from train_ai2v import train as train_ai2v

I2V = 'i2v'
AI2V = 'ai2v'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ai2v', help="model to train: i2v or ai2v")
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./output/', help="model directory path")
    parser.add_argument('--train', type=str, default='train.dat', help="train file name")
    parser.add_argument('--valid', type=str, default='valid.dat', help="validation users file name")
    parser.add_argument('--test', type=str, default='test.dat', help="test users file name")
    parser.add_argument('--full_train', type=str, default='full_train.dat', help="full train file name")
    parser.add_argument('--vocab', type=str, default='vocab.dat', help="vocab file")
    parser.add_argument('--ic', type=str, default='ic.dat', help='items counts file')
    parser.add_argument('--item2idx', type=str, default='item2idx.dat', help='item2index mapping')
    parser.add_argument('--idx2item', default='idx2item.dat', help='index2item mapping')
    parser.add_argument('--max_epoch', type=int, default=50, help="max number of epochs")
    parser.add_argument('--patience', type=float, default=3, help="epochs to wait until early stopping")
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token")
    parser.add_argument('--trials', type=int, default=5, help="number of trials ")
    parser.add_argument('--k', type=int, default=20, help="k to use when calculating hr_k and mrr_k")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    parser.add_argument('--window_size', type=int, default=1000, help="window size")
    parser.add_argument('--log_dir', type=str, default='my_logdir', help="directory for tensorboard logs")
    parser.add_argument('--hr_out', type=str, default='hr_out.csv', help="out file name of hr for test set")
    parser.add_argument('--rr_out', type=str, default='rr_out.csv', help="out file name of rr for test set")
    parser.add_argument('--cnfg_out', type=str, default='best_cnfg.pkl', help="best configuration file name")

    return parser.parse_args()


def i2v_full_train(cnfg, epochs, args):
    cnfg['max_epoch'] = int(epochs)
    cnfg['train'] = args.full_train
    train_i2v(cnfg)


def ai2v_full_train(cnfg, epochs, args):
    cnfg['max_epoch'] = int(epochs)
    cnfg['train'] = args.full_train
    train_ai2v(cnfg)


def main():
    args = parse_args()
    if args.model == I2V:
        best_parameters, values, _experiment, _cur_model = optimize(
            parameters=[
                {"name": "lr", "type": "range", "value_type": "float", "bounds": [3e-2, 1e-1]},
                {"name": "ss_t", "type": "range", "value_type": "float", "bounds": [1e-5, 3e-3]},
                {"name": "e_dim", "type": "choice", "value_type": "int", "values": [12, 17, 20, 25, 30, 50, 100]},
                {"name": "n_negs", "type": "choice", "value_type": "int", "values": [7, 8]},
                {"name": "mini_batch", "type": "choice", "value_type": "int", "values": [128, 256, 500, 1000]},
                {"name": "weights", "type": "choice", "value_type": "bool", "values": [False, False]},
                {"name": "max_epoch", "type": "fixed", "value_type": "int", "value": args.max_epoch},
                {"name": "patience", "type": "fixed", "value_type": "int", "value": args.patience},
                {"name": "unk", "type": "fixed", "value_type": "str", "value": args.unk},
                {"name": "cuda", "type": "fixed", "value": args.cuda},
                {"name": "data_dir", "type": "fixed", "value_type": "str", "value": args.data_dir},
                {"name": "save_dir", "type": "fixed", "value_type": "str", "value": args.save_dir},
                {"name": "train", "type": "fixed", "value_type": "str", "value": args.train},
                {"name": "valid", "type": "fixed", "value_type": "str", "value": args.valid},
                {"name": "test", "type": "fixed", "value_type": "str", "value": args.test},
                {"name": "ic", "type": "fixed", "value_type": "str", "value": args.ic},
                {"name": "vocab", "type": "fixed", "value_type": "str", "value": args.vocab},
                {"name": "item2idx", "type": "fixed", "value_type": "str", "value": args.item2idx},
                {"name": "idx2item", "type": "fixed", "value_type": "str", "value": args.idx2item},
                {"name": "window_size", "type": "fixed", "value_type": "int", "value": args.window_size},
                {"name": "model", "type": "fixed", "value_type": "str", "value": args.model},
                {"name": "log_dir", "type": "fixed", "value_type": "str", "value": args.log_dir},
                {"name": "k", "type": "fixed", "value_type": "int", "value": args.k},
                {"name": "hr_out", "type": "fixed", "value_type": "str", "value": args.hr_out},
                {"name": "rr_out", "type": "fixed", "value_type": "str", "value": args.rr_out},
            ],
            evaluation_function=train_evaluate_i2v,
            minimize=True,
            objective_name='valid_loss',
            total_trials=args.trials
        )

        best_parameters['best_epoch'] = values[0]['early_stop_epoch']
        pickle.dump(best_parameters, open(pathlib.Path(args.save_dir, args.cnfg_out), "wb"))
        i2v_full_train(best_parameters, values[0]['early_stop_epoch'], args)

    else:
        best_parameters, values, _experiment, _cur_model = optimize(
            parameters=[
                {"name": "lr", "type": "range", "value_type": "float", "bounds": [5e-2, 1e-1]},
                {"name": "dropout_rate", "type": "range", "value_type": "float", "bounds": [0, 1]},
                {"name": "ss_t", "type": "range", "value_type": "float", "bounds": [1e-5, 3e-3]},
                {"name": "e_dim", "type": "choice", "value_type": "int", "values": [12, 17, 19, 20, 22, 25, 30, 50, 100]},
                {"name": "n_negs", "type": "choice", "value_type": "int", "values": [7, 8]},
                {"name": "mini_batch", "type": "choice", "value_type": "int", "values": [30, 32]},
                {"name": "num_blocks", "type": "choice", "value_type": "int", "values": [2, 3, 4]},
                {"name": "num_heads", "type": "choice", "value_type": "int", "values": [1]},
                {"name": "weights", "type": "choice", "value_type": "bool", "values": [False, False]},
                {"name": "max_epoch", "type": "fixed", "value_type": "int", "value": args.max_epoch},
                {"name": "patience", "type": "fixed", "value_type": "int", "value": args.patience},
                {"name": "unk", "type": "fixed", "value_type": "str", "value": args.unk},
                {"name": "cuda", "type": "fixed", "value": args.cuda},
                {"name": "data_dir", "type": "fixed", "value_type": "str", "value": args.data_dir},
                {"name": "save_dir", "type": "fixed", "value_type": "str", "value": args.save_dir},
                {"name": "train", "type": "fixed", "value_type": "str", "value": args.train},
                {"name": "valid", "type": "fixed", "value_type": "str", "value": args.valid},
                {"name": "test", "type": "fixed", "value_type": "str", "value": args.test},
                {"name": "ic", "type": "fixed", "value_type": "str", "value": args.ic},
                {"name": "vocab", "type": "fixed", "value_type": "str", "value": args.vocab},
                {"name": "item2idx", "type": "fixed", "value_type": "str", "value": args.item2idx},
                {"name": "idx2item", "type": "fixed", "value_type": "str", "value": args.idx2item},
                {"name": "window_size", "type": "fixed", "value_type": "int", "value": args.window_size},
                {"name": "model", "type": "fixed", "value_type": "str", "value": args.model},
                {"name": "log_dir", "type": "fixed", "value_type": "str", "value": args.log_dir},
                {"name": "k", "type": "fixed", "value_type": "int", "value": args.k},
                {"name": "hr_out", "type": "fixed", "value_type": "str", "value": args.hr_out},
                {"name": "rr_out", "type": "fixed", "value_type": "str", "value": args.rr_out},
            ],
            evaluation_function=train_evaluate_ai2v,
            minimize=True,
            objective_name='valid_loss',
            total_trials=args.trials
        )

        best_parameters['best_epoch'] = values[0]['early_stop_epoch']
        pickle.dump(best_parameters, open(pathlib.Path(args.save_dir, args.cnfg_out), "wb"))
        ai2v_full_train(best_parameters, values[0]['early_stop_epoch'], args)


if __name__ == '__main__':
    main()
