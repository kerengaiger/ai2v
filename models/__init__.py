from models.ai2v_model import AttentiveItemToVec as ai2v
from models.ai2v_model import SGNS as sgns_ai2v
from models.i2v_model import Item2Vec as i2v
from models.i2v_model import SGNS as sgns_i2v

ai2v_cnfg_keys = ['padding_idx', 'vocab_size', 'embedding_size']
i2v_cnfg_keys = ['padding_idx', 'vocab_size', 'embedding_size', 'num_heads',
                 'num_blocks', 'dropout_rate']