# TRAINING CONFIG
MASK_RATIO_LOW = 0.3
MASK_RATIO_HIGH = 0.7
BIT = 6
REINIT_THRESHOLD = 7
REINIT_TRAIN_EPOCH = 200
TOTAL_TRAIN_EPOCH = 250

# SOLID CONFIG
SOLID_TRAIN_PATH = 'data/solid/train_deduplicate.pkl'
SOLID_TEST_PATH = 'data/solid/test.pkl'
SOLID_VAL_PATH = 'data/solid/val.pkl'
SOLID_FULL_PATH = 'data/solid/train.pkl'
SOLID_PARAM_SEQ = 6
MAX_SOLID = 5
SOLID_CODEBOOK_DIM = 10000 # or optionally use 5000

# PROFILE CONFIG
PROFILE_TRAIN_PATH = 'data/profile/train.pkl'
PROFILE_VAL_PATH = 'data/profile/val.pkl'
PROFILE_TEST_PATH = 'data/profile/test.pkl'
PROFILE_EVL_PATH = 'data/profile/evl.pkl'
PROFILE_PARAM_SEQ = 4
MAX_PROFILE = 20
PROFILE_CODEBOOK_DIM = 5000

# LOOP CONFIG
LOOP_TRAIN_PATH = 'data/loop/train.pkl'
LOOP_VAL_PATH = 'data/loop/val.pkl'
LOOP_TEST_PATH = 'data/loop/test.pkl'
LOOP_FULL_PATH = 'data/loop/train.pkl'
LOOP_PARAM_PAD = 2
LOOP_PARAM_SEQ = 2
MAX_LOOP = 80
LOOP_CODEBOOK_DIM = 5000

# NETWORK CONFIG
ENCODER_CONFIG = {
    'hidden_dim': 512,
    'embed_dim': 256,
    'num_layers': 4,
    'num_heads': 8,
    'dropout_rate': 0.1
}
DECODER_CONFIG = {
    'hidden_dim': 512,
    'embed_dim': 256,
    'num_layers': 4,
    'num_heads': 8,
    'dropout_rate': 0.1
}

