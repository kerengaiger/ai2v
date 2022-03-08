from models.ai2v_model import AttentiveItemToVec as ai2v
from models.ai2v_model import SGNS as sgns_ai2v
from models.i2v_model import Item2Vec as i2v
from models.i2v_model import SGNS as sgns_i2v

i2v_cnfg_keys = ['padding_idx', 'vocab_size', 'emb_size']
ai2v_cnfg_keys = ['padding_idx', 'vocab_size', 'emb_size', 'window_size', 'num_users', 'device', 'n_b', 'n_h',
                  'add_pos_bias']
