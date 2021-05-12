import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import argparse

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
    parser.add_argument('--train', type=str, default='train_batch_u.dat', help="train file name")
    parser.add_argument('--valid', type=str, default='valid_batch_u.dat', help="validation users file name")
    parser.add_argument('--test', type=str, default='test_batch_u.dat', help="test users file name")
    parser.add_argument('--full_train', type=str, default='full_train_batch_u.dat', help="full train file name")
    parser.add_argument('--max_epoch', type=int, default=50, help="max number of epochs")
    parser.add_argument('--patience', type=float, default=3, help="epochs to wait until early stopping")
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token")
    parser.add_argument('--trials', type=int, default=10, help="number of trials ")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    parser.add_argument('--max_batch_size', type=int, default=200, help="max number of training obs in batch")
    parser.add_argument('--log_dir', type=str, default='tensorboard/logs/mylogdir', help="logs dir for tensorboard")
    parser.add_argument('--k', type=int, default=20, help="k to calc hrr_k and mrr_k evaluation metrics")

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
                {"name": "model", "type": "fixed", "value_type": "str", "value": args.model},
                {"name": "lr", "type": "range", "value_type": "float", "bounds": [3e-2, 1e-1]},
                {"name": "ss_t", "type": "range", "value_type": "float", "bounds": [1e-5, 3e-3]},
                {"name": "e_dim", "type": "choice", "value_type": "int", "values": [12, 17, 20, 25, 30]},
                {"name": "n_negs", "type": "choice", "value_type": "int", "values": [7, 8]},
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
                {"name": "max_batch_size", "type": "fixed", "value_type": "int", "value": args.max_batch_size},
                {"name": "log_dir", "type": "fixed", "value_type": "str", "value": args.log_dir},
                {"name": "k", "type": "fixed", "value_type": "int", "value": args.k},
            ],
            evaluation_function=train_evaluate_i2v,
            minimize=True,
            objective_name='valid_loss',
            total_trials=args.trials
        )

        i2v_full_train(best_parameters, values[0]['early_stop_epoch'], args)

    else:
        best_parameters, values, _experiment, _cur_model = optimize(
            parameters=[
                {"name": "model", "type": "fixed", "value_type": "str", "value": args.model},
                {"name": "lr", "type": "range", "value_type": "float", "bounds": [3e-2, 1e-1]},
                {"name": "ss_t", "type": "range", "value_type": "float", "bounds": [1e-5, 3e-3]},
                {"name": "e_dim", "type": "choice", "value_type": "int", "values": [80, 100, 20]},
                {"name": "n_negs", "type": "choice", "value_type": "int", "values": [7, 8]},
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
                {"name": "max_batch_size", "type": "fixed", "value_type": "int", "value": args.max_batch_size},
                {"name": "log_dir", "type": "fixed", "value_type": "str", "value": args.log_dir},
                {"name": "k", "type": "fixed", "value_type": "int", "value": args.k},
            ],
            evaluation_function=train_evaluate_ai2v,
            minimize=True,
            objective_name='valid_loss',
            total_trials=args.trials
        )

        ai2v_full_train(best_parameters, values[0]['early_stop_epoch'], args)


if __name__ == '__main__':
    main()
