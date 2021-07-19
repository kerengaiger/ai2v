#!/bin/bash
# the running function:
# bash wrapper.sh movielens split

# split choices ['leave_one_out', 'users_split']

function prepare_database_corpus(){
  echo "preparing $1"

  # If to check what data base I want to run.
  if [ "$1" = "movielens" ]; then
    if [ "$2" = "llo" ]; then
      python prepare_corpus.py --input_file ./data/movielens_corpus.dat --line_sep :: --min_usr_len 2 --max_usr_len 60 \
      --min_items_cnt 10 --max_items_cnt 10000 --final_usr_len 4 --split_strategy "$2" --data_dir ./corpus/movielens/
    else
      python prepare_corpus.py --input_file ./data/movielens_corpus.dat --line_sep :: --min_usr_len 2 --max_usr_len 60 \
      --min_items_cnt 10 --max_items_cnt 10000 --final_usr_len 3 --split_strategy "$2" --data_dir ./corpus/movielens/
    fi
  fi

  if [ "$1" = "netflix" ]; then
    if [ "$2" = "llo" ]; then
      python prepare_corpus.py --input_file ./data/netflix_corpus.dat --line_sep , --min_usr_len 3 --max_usr_len 2700 \
      --min_items_cnt 100 --max_items_cnt 130000 --final_usr_len 4 --split_strategy "$2" --data_dir ./corpus/netflix/
    else
      python prepare_corpus.py --input_file ./data/netflix_corpus.dat --line_sep , --min_usr_len 3 --max_usr_len 2700 \
      --min_items_cnt 100 --max_items_cnt 130000 --final_usr_len 3 --split_strategy "$2" --data_dir ./corpus/netflix/
    fi
  fi

  if [ "$1" = "moviesdat" ]; then
    python prepare_corpus.py --input_file ./data/moviesdat_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 --min_items_cnt 5 --max_items_cnt 1000 --final_usr_len 2
  fi

  if [ "$1" = "yahoo" ]; then
    python prepare_corpus.py --input_file ./data/yahoo_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 \
    --min_items_cnt 5 --max_items_cnt 1000 --final_usr_len 2
  fi

  if [ "$1" = "goodbooks" ]; then
    python prepare_corpus.py --input_file ./data/goodbooks_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 \
    --min_items_cnt 5 --max_items_cnt 1000 --final_usr_len 2
  fi

  if [ "$1" = "booksrec" ]; then
    python prepare_corpus.py --input_file ./data/booksrec_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 \
    --min_items_cnt 5 --max_items_cnt 1000 --final_usr_len 2
  fi

  if [ "$1" = "animerec" ]; then
    python prepare_corpus.py --input_file ./data/animerec_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 \
    --min_items_cnt 5 --max_items_cnt 1000 --final_usr_len 2
  fi

  if [ "$1" = "animerec20" ]; then
    python prepare_corpus.py --input_file ./data/animerec20_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 \
    --min_items_cnt 5 --max_items_cnt 1000 --final_usr_len 2
  fi
}

function run_pipeline() {
  echo "pipeline for $1"

  python preprocess.py --vocab ./data/full_train.txt --full_corpus ./data/full_train.txt --test_corpus ./data/test.txt \
  --build_train_valid --train_corpus ./data/train.txt --valid_corpus ./data/valid.txt \
  --full_train_file ./data/full_train.dat --train_file "./data/train_$1_$2.dat" --valid_file ./data/valid_.dat \
  --test_file ./data/test.dat

  python hyper_param_tune.py --model ai2v --data_dir ./data/ --save_dir ./output/ --train train.dat --valid valid.dat \
  --test test.dat --full_train full_train.dat --max_epoch 50 --patience 3 --trials 5 --cuda --log_dir my_log_dir \
  --k 20 --hr_out ./output/hr_out.csv --rr_out ./output/rr_out.csv --cnfg_out ./output/best_cnfg.pkl

  python hyper_param_tune.py --model i2v --data_dir ./data/ --save_dir ./output/ --train train.dat --valid valid.dat \
  --test test.dat --full_train full_train.dat --max_epoch 50 --patience 3 --trials 5 --cuda --log_dir my_log_dir \
  --k 20 --hr_out ./output/hr_out.csv --rr_out ./output/rr_out.csv --cnfg_out ./output/best_cnfg.pkl

  python evaluation.py --k 20 --model ai2v --test ./data/test.dat

  python evaluation.py --k 20 --model i2v --test ./data/test.dat

  #python evaluation.py/test_p_value('./output/ht_ai2v.csv', './output/ht_i2v.csv')
}

if [ "$1" = "all" ]; then
  #databases_list=["movielens", "netflix", "moviesdat", "yahoo", "goodbooks", "booksrec", "animerec", "animerec20"]
  declare -a databases_list=("movielens" "netflix" "moviesdat" "yahoo" "goodbooks" "booksrec" "animerec" "animerec20")
else
  declare -a databases_list=($1)
fi

# A for loop to run on all the databases.
for database in "${databases_list[@]}"; do
  prepare_database_corpus $database
  run_pipeline $database
done