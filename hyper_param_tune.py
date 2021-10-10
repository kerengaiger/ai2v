import argparse
import pickle
import pathlib

import optuna

from train import train_evaluate, train


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
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token")
    parser.add_argument('--trials', type=int, default=50, help="number of trials ")
    parser.add_argument('--num_workers', type=int, default=0, help="num workers to load train_loader")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    parser.add_argument('--device', type=int, default=0, help="cude device to use")
    parser.add_argument('--window_size', type=int, default=1000, help="window size")
    parser.add_argument('--log_dir', type=str, default='my_logdir', help="directory for tensorboard logs")
    parser.add_argument('--cnfg_init', type=str, default=None, help="initial configuration to start study from")
    parser.add_argument('--cnfg_out', type=str, default='best_cnfg.pkl', help="best configuration file name")
    parser.add_argument('--loss_method', type=str, default='CCE', help="the loss method")
    parser.add_argument('--seed', type=int, default=2021, help="seed number")
    parser.add_argument('--n_h', type=int, default=1, help="number of heads in attention")
    parser.add_argument('--n_b', type=int, default=1, help="number of attention blocks")
    parser.add_argument('--add_last_item_emb', action='store_true', help="add last item embedding to the user representation")
    return parser.parse_args()


class Objective:

    def __init__(self):
        self.best_epoch = None

    def __call__(self, trial):
        cnfg = {}
        args = parse_args()
        args = vars(args)
        cnfg['lr'] = trial.suggest_loguniform("lr", 1e-5, 1e-1)
        cnfg['ss_t'] = 1e-4
        cnfg['emb_size'] = 50
        cnfg['n_negs'] = 7
        cnfg['mini_batch'] = trial.suggest_categorical("mini_batch", [32, 64, 128, 200, 256])
        cnfg['weights'] = trial.suggest_categorical("weights", [False, False])
        valid_loss, best_epoch = train_evaluate({**cnfg, **args}, trial)
        self.best_epoch = best_epoch
        return valid_loss

    def callback(self, study, trial):
        args = parse_args()
        if study.best_trial == trial:
            best_cnfg = trial.params
            best_cnfg['best_epoch'] = self.best_epoch
            best_cnfg['max_epoch'] = best_cnfg['best_epoch']
            best_cnfg = {**best_cnfg, **vars(args)}
            pickle.dump(best_cnfg, open(pathlib.Path(args.save_dir, args.cnfg_out), "wb"))


def main():
    args = parse_args()
    objective = Objective()

    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="minimize"
    )
    study.optimize(objective, n_trials=args.trials, callbacks=[objective.callback])

    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: ", best_trial.value)

    best_parameters = pickle.load(open(pathlib.Path(args.save_dir, args.cnfg_out), "rb"))
    best_parameters['max_epoch'] = best_parameters['best_epoch']
    best_parameters['train'] = args.full_train
    train(best_parameters)


if __name__ == '__main__':
    main()
