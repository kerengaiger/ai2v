#!/bin/bash
# the running function example:
# bash wrapper.sh movielens llo

function prepare_database_corpus(){
  echo "preparing $1"

  # If to check what data base I want to run.
  if [ "$1" = "movielens" ]; then
    if [ "$2" = "llo" ]; then
      python prepare_corpus.py --input_file ./data/movielens_corpus.csv --line_sep :: --min_usr_len 1 --max_usr_len 1000 \
      --min_items_cnt 10 --max_items_cnt 10000 --final_usr_len 4 --split_strategy leave_one_out --data_dir ./corpus/movielens_llo/
    else
      python prepare_corpus.py --input_file ./data/movielens_corpus.csv --line_sep :: --min_usr_len 1 --max_usr_len 1000 \
      --min_items_cnt 10 --max_items_cnt 10000 --final_usr_len 3 --split_strategy users_split --data_dir ./corpus/movielens/
    fi
  fi

  if [ "$1" = "netflix" ]; then
    if [ "$2" = "llo" ]; then
      python prepare_corpus.py --input_file ./data/netflix_corpus.csv --line_sep , --min_usr_len 3 --max_usr_len 1000 \
      --min_items_cnt 100 --max_items_cnt 130000 --final_usr_len 4 --split_strategy leave_one_out --data_dir ./corpus/netflix_llo/
    else
      python prepare_corpus.py --input_file ./data/netflix_corpus.csv --line_sep , --min_usr_len 3 --max_usr_len 1000 \
      --min_items_cnt 100 --max_items_cnt 130000 --final_usr_len 3 --split_strategy users_split --data_dir ./corpus/netflix/
    fi
  fi

  if [ "$1" = "moviesdat" ]; then
    if [ "$2" = "llo" ]; then
      python prepare_corpus.py --input_file ./data/moviesdat_corpus.csv --line_sep , --min_usr_len 100 --max_usr_len 1000 \
      --min_items_cnt 100 --max_items_cnt 100000 --final_usr_len 4 --split_strategy leave_one_out --data_dir ./corpus/moviesdat_llo/
    else
      python prepare_corpus.py --input_file ./data/moviesdat_corpus.csv --line_sep , --min_usr_len 100 --max_usr_len 1000 \
      --min_items_cnt 100 --max_items_cnt 100000 --final_usr_len 3 --split_strategy users_split --data_dir ./corpus/moviesdat/
    fi
  fi

  if [ "$1" = "yahoo" ]; then
    if [ "$2" = "llo" ]; then
      python prepare_corpus.py --input_file ./data/yahoo_all_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 \
      --min_items_cnt 10 --max_items_cnt 100000 --final_usr_len 4 --split_strategy leave_one_out --data_dir ./corpus/yahoo_llo/
    else
      python prepare_corpus.py --input_file ./data/yahoo_all_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 \
      --min_items_cnt 10 --max_items_cnt 100000 --final_usr_len 3 --split_strategy users_split --data_dir ./corpus/yahoo/
    fi
  fi

  if [ "$1" = "goodbooks" ]; then
    if [ "$2" = "llo" ]; then
      python prepare_corpus.py --input_file ./data/goodbooks_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 \
      --min_items_cnt 5 --max_items_cnt 10000 --final_usr_len 4 --split_strategy leave_one_out --data_dir ./corpus/goodbooks_llo/
    else
      python prepare_corpus.py --input_file ./data/goodbooks_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 \
      --min_items_cnt 5 --max_items_cnt 10000 --final_usr_len 3 --split_strategy users_split --data_dir ./corpus/goodbooks/
    fi
  fi

  if [ "$1" = "booksrec" ]; then
    if [ "$2" = "llo" ]; then
      python prepare_corpus.py --input_file ./data/booksrec_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 \
      --min_items_cnt 5 --max_items_cnt 10000 --final_usr_len 4 --split_strategy leave_one_out --data_dir ./corpus/booksrec_llo/
    else
      python prepare_corpus.py --input_file ./data/booksrec_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 \
      --min_items_cnt 5 --max_items_cnt 10000 --final_usr_len 3 --split_strategy users_split --data_dir ./corpus/booksrec/
    fi
  fi

  if [ "$1" = "animerec" ]; then
    if [ "$2" = "llo" ]; then
      python prepare_corpus.py --input_file ./data/animerec_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 \
      --min_items_cnt 10 --max_items_cnt 100000 --final_usr_len 4 --split_strategy leave_one_out --data_dir ./corpus/animerec_llo/
    else
      python prepare_corpus.py --input_file ./data/animerec_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 \
      --min_items_cnt 10 --max_items_cnt 100000 --final_usr_len 3 --split_strategy users_split --data_dir ./corpus/animerec/
    fi
  fi

  if [ "$1" = "animerec20" ]; then
    if [ "$2" = "llo" ]; then
      python prepare_corpus.py --input_file ./data/animerec20_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 \
      --min_items_cnt 15 --max_items_cnt 100000 --final_usr_len 4 --split_strategy leave_one_out --data_dir ./corpus/animerec20_llo/
    else
      python prepare_corpus.py --input_file ./data/animerec20_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 \
      --min_items_cnt 15 --max_items_cnt 100000 --final_usr_len 3 --split_strategy users_split --data_dir ./corpus/animerec20/
    fi
  fi

  if [ "$1" = "amazonbeauty" ]; then
    if [ "$2" = "llo" ]; then
      python prepare_corpus.py --input_file ./data/amazonbeauty_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 \
      --min_items_cnt 5 --max_items_cnt 50000 --final_usr_len 4 --split_strategy leave_one_out --data_dir ./corpus/amazonbeauty_llo/
    else
      python prepare_corpus.py --input_file ./data/amazonbeauty_corpus.csv --line_sep , --min_usr_len 2 --max_usr_len 1000 \
      --min_items_cnt 5 --max_items_cnt 50000 --final_usr_len 3 --split_strategy users_split --data_dir ./corpus/amazonbeauty/
    fi
  fi
}

function run_pipeline() {
  echo "pipeline for $1 with $2"
  # split choices ['leave_one_out', 'users_split']

  if [ "$2" = "llo" ]; then
    python preprocess.py --data_dir "./corpus/$1_llo/" --build_train_valid --split_strategy leave_one_out

    python hyper_param_tune.py --model ai2v --data_dir "./corpus/$1_llo/" --save_dir "./output/$1_ai2v_llo/" \
    --cuda --window_size 1000 --log_dir "tensorboard/logdir/$1_llo_ai2v"
    python hyper_param_tune.py --model i2v --data_dir "./corpus/$1_llo/" --save_dir "./output/$1_i2v_llo/" \
    --cuda --window_size 1000 --log_dir "tensorboard/logdir/$1_llo_i2v"

    # python evaluation.py --k 20 --data_dir "./data/$1_llo/" --output_dir "./output/$1_ai2v_llo/"
    # python evaluation.py --k 20 --data_dir "./data/$1_llo/" --output_dir "./output/$1_i2v_llo/"
    # python evaluation.py --k 10 --data_dir "./data/$1_llo/" --output_dir "./output/$1_ai2v_llo/"
    # python evaluation.py --k 10 --data_dir "./data/$1_llo/" --output_dir "./output/$1_i2v_llo/"
    # python evaluation.py --k 5 --data_dir "./data/$1_llo/" --output_dir "./output/$1_ai2v_llo/"
    # python evaluation.py --k 5 --data_dir "./data/$1_llo/" --output_dir "./output/$1_i2v_llo/"
  else
    python preprocess.py --data_dir "./data/$1_user_split/" --build_train_valid --split_strategy users_split

    python hyper_param_tune.py --model ai2v --data_dir "./data/$1_user_split/" --save_dir "./output/$1_ai2v_user_split/" \
    --cuda --window_size 1000 --log_dir "tensorboard/logdir/$1_user_split_ai2v"
    python hyper_param_tune.py --model i2v --data_dir "./data/$1_user_split/" --save_dir "./output/$1_i2v_user_split/" \
    --cuda --window_size 1000 --log_dir "tensorboard/logdir/$1_user_split_i2v"

    python evaluation.py --k 20 --data_dir "./data/$1_user_split/" --output_dir "./output/$1_ai2v_user_split/"
    # python evaluation.py --k 20 --data_dir "./data/$1_user_split/" --output_dir "./output/$1_i2v_user_split/"
    python evaluation.py --k 10 --data_dir "./data/$1_user_split/" --output_dir "./output/$1_ai2v_user_split/"
    # python evaluation.py --k 10 --data_dir "./data/$1_user_split/" --output_dir "./output/$1_i2v_user_split/"
    python evaluation.py --k 5 --data_dir "./data/$1_user_split/" --output_dir "./output/$1_ai2v_user_split/"
    # python evaluation.py --k 5 --data_dir "./data/$1_user_split/" --output_dir "./output/$1_i2v_user_split/"
  fi
  #python evaluation.py/test_p_value('./output/ht_ai2v.csv', './output/ht_i2v.csv')
}

if [ "$1" = "all" ]; then
  declare -a databases_list=("moviesdat" "goodbooks" "amazonbeauty")
else
  declare -a databases_list=($1)
fi

# A for loop to run on all the databases.
for database in "${databases_list[@]}"; do
  prepare_database_corpus $database $2
  run_pipeline $database $2
done