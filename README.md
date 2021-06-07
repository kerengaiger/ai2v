# Item2Vec and Attentiv Item2Vec for items recommendations

This repository contain implementations of item2vec and attentive item2vec models presented in the paper:
https://arxiv.org/ftp/arxiv/papers/2002/2002.06205.pdf

## Item2Vec
For each item 𝑖, I2V learns latent context and target vectors 𝑢𝑖, 𝑣𝑖 which are estimated via implicit 
factorization of items co-occurrences matrix.

## Attentive Item2Vec
AI2V extracts attentive context-target representations each
designed to capture different user behavior w.r.t the potential
target item recommendation. These attentive context-target
representations are then fed into a secondary network that
chooses the most relevant user characteristics and constructs
a final attentive user vector. This vector constitutes the
dynamic representation 

### Usage
#### preprocess.py
Gets train, valid and test txt files containing users as list of items. One should also provide the file 
to create the vocabulary from. Returns .dat files of training samples for full train(train + validation, 
used to train the final model), train, validation and test.

##### Running example
```bash
python preprocess.py --vocab ./data/full_train.txt --full_corpus ./data/full_train.txt --test_corpus ./data/test.txt 
--build_train_valid --train_corpus ./data/train.txt --valid_corpus ./data/valid.txt --full_train_file ./data/full_train.dat 
--train_file ./data/train.dat --valid_file ./data/valid_.dat --test_file ./data/test.dat
```
#### hyper_param_tune.py
Performs a hyper parameters search when training the chosen model(I2V/AI2V). After the best configuration has chosen,
the model is being trained with it, its final weights are being saved into output files and the 
[mean recipocal rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) and [hit ratio](https://en.wikipedia.org/wiki/Hit_rate)
are evaluated on the given test set. Training procedure is being tracked with Tensorboard  

##### User Arguments
* model - ai2v / i2v
* data_dir - the directory of all data files
* save_dir - the directory of all output files
* train - the train file path
* valid - validation file path
* test - test file path
* full_train - full train(train+valid) file path
* max_epoch - maximum number of epochs to run
* patience - number of epochs the user tolerates a validation loss increase before early stopping
* trials - number of trials to test in the optimization search
* log_dir - the name of directory for tensorboard logs
* k - the k used in hr_k and mrp_k on test set
* hr_out - .csv file containing the hit/ miss for each test samples
* rr_out - .csv file containing the recipocal rank for each test sample
* cnfg_out - .pkl file containing the best configuration 

##### Running example
```bash
python hyper_param_tune.py --model ai2v --data_dir ./data/ --save_dir ./output/ --train train.dat --valid valid.dat 
--test test.dat --full_train full_train.dat --max_epoch 50 --patience 3 --trials 5 --cuda 
--log_dir my_log_dir --k 20 --hr_out ./output/hr_out.csv --rr_out ./output/rr_out.csv --cnfg_out ./output/best_cnfg.pkl
```

#### train_i2v.py, train_ai2v.py
Train ai2v or i2v models with a chosen configuration over a certain number of epochs.

##### Running example
```bash
python train_ai2v.py --data_dir ./data/ --save_dir ./output/ --train train.dat 
--test test.dat --patience 3  --cuda --log_dir my_log_dir --k 20 --hr_out ./output/hr_out.csv 
--rr_out ./output/rr_out.csv --cnfg_out ./output/best_cnfg.pkl --max_epoch 50
```

#### evaluation.py
Evaluates model performance using hit ratio and mean recipocal rank metrics.
##### Running example
```bash
python evaluation.py --k 20 --model ai2v --test ./data/test.dat
```

This module also contain a function for calculating the p_value when assessing 
the difference between the two model results(in terms of mean recipocal rank and 
the hit ratio). Inputs files must be in a .csv formal, no header, 
in the format of user_id,item_id, metric_values

```python
from evaluation import test_p_value
test_p_value('./output/ht_ai2v.csv', './output/ht_i2v.csv')
```