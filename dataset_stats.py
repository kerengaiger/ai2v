import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output_hist', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input, names=['usr', 'itm', 'rating', 'timestamp'])
    users = list(df['usr'].sort_values().unique())
    users_dict = {k: v for k, v in zip(users, range(len(users)))}
    df['usr_id'] = df['usr'].map(users_dict)
    items = list(df['itm'].sort_values().unique())
    items_dict = {k: v for k, v in zip(items, range(len(items)))}
    df['itm_id'] = df['itm'].map(items_dict)
    print('mean sequence:', df['usr_id'].itm_id.size().mean())
    users_sizes = df.groupby('usr_id').item.size().reset_index()
    users_sizes.groupby('itm_id').user.size().to_csv(args.output_hist)
    df.groupby('usr_id').item.size().hist(bins=1000)


if __name__ == '__main__':
    main()
