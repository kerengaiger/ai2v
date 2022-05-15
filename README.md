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
Raw input files must include user_id, item_id, date and rate_score. We trained the models on Movielens 1M,
Netflix, Yahoo, Goodbooks, Moviesdat, Amazon Books and Amazon Beuty datasets. Preprocess parameters we used are
configured at the json files in ```cnfg/``` folder. In case one would like to test the models on a new dataset, 
he needs to create a proper json file.
#### Training files generation
```python
from dataset import generate_train_files
generate_train_files('path/to/json/cnfg/file')
```
#### Train
Train one of the models based on a configuration file of hyper parameter values.
```bash
python train.py --data_dir path/to/data/dir --data_cnfg path/to/json/cnfg --save_dir /path/to/model/saves/ \
--cuda --device 0 --best_cnfg path/to/params/cnfg

```
#### Tune
Tune one of the models hyper parameters on a specific dataset. This module finds the best configuration,
trains the model on the full training set and saves the final model file (based on an early stop process).
One can change the number of heads and attention blocks using the n_h and n_b args respectively. In order
to learn the items' positional bias, configure --add_pos_bias. 
```bash
python hyper_param_tune.py --model ai2v --data_dir path/to/data/dir --data_cnfg path/to/json/cnfg \
--save_dir /path/to/model/saves/ --trials 50 --num_workers 4 --cuda --device 0  \
--log_dir path/to/tensorboard/logs/dir --cnfg_init /path/to/initial/cnfg --cnfg_out path/to/best/cnfg/save \
--loss_method BCE --n_h 1 --n_b 1 --add_pos_bias
```
#### Evaluate
Calculate the HR@K, MRR@K, NDCG@K and MPR@K metrics given a model,a test set and a certain K value.
```bash
python evaluation.py --k 20 --data_dir path/to/data/dir --output_dir /path/to/model/saves/ \
--hr_out path/to/hr/out/file --mrr_out path/to/mrr/out/file
```