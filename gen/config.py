###############
# Data Config #
###############
CAD_BIT = 6
EXT_SEQ=11 
MAX_BOX=5
MAX_EXT = MAX_BOX*EXT_SEQ+1
SKETCH_R = 1
EXTRUDE_R = 1
BBOX_RANGE = 1
CUBOID_RANGE = 1
MAX_CAD = 200
MAX_CODE = 35
SKETCH_PAD = 3
EXT_PAD = 2
CODE_PAD = 3

################
# Train Config #
################
UNCOND_TRAIN_EPOCH = 350
COND_TRAIN_EPOCH = 250
CAD_TRAIN_PATH = 'data/model/train_deduplicate.pkl'
PROFILE_TRAIN_PATH = 'data/profile/train.pkl'
PROFILE_VAL_PATH = 'data/profile/val.pkl'
PROFILE_TEST_PATH = 'data/profile/test.pkl'
LOOP_TRAIN_PATH = 'data/loop/train.pkl'
LOOP_VAL_PATH = 'data/loop/val.pkl'
LOOP_TEST_PATH = 'data/loop/test.pkl'

TRAIN_PROFILE_CODE = 'train/profile.pkl'
TRAIN_LOOP_CODE = 'train/loop.pkl'
VAL_PROFILE_CODE = 'val/profile.pkl'
VAL_LOOP_CODE = 'val/loop.pkl'
TEST_PROFILE_CODE = 'test/profile.pkl'
TEST_LOOP_CODE = 'test/loop.pkl'

ENCODER_CONFIG = {
    'hidden_dim': 512,
    'embed_dim': 256,
    'num_layers': 6,
    'num_heads': 8,
    'dropout_rate': 0.1
}
DECODER_CONFIG = {
    'hidden_dim': 512,
    'embed_dim': 256,
    'num_layers': 6,
    'num_heads': 8,
    'dropout_rate': 0.1
}
CODE_CONFIG = {
    'hidden_dim': 512,
    'embed_dim': 256,
    'num_layers': 6,
    'num_heads': 8,
    'dropout_rate': 0.1
}
AUG_RANGE = 2 # or 1
MASK_RATIO_LOW = 0.0
MASK_RATIO_HIGH = 1.0

#################
# Sample Config #
#################
code_top_p_sample = 0.98
code_top_p_eval = 1.0
cad_top_p_sample = 0.5  # nucleus sampling has better visual quality
cad_top_p_eval = 1.0 # simply sample based on the distribution
RANDOM_SAMPLE_TOTAL = 1000  # visualize 1000 randomly generated samples
RANDOM_SAMPLE_BS = 32
RANDOM_EVAL_TOTAL = 15000 # need more generated data for evaluation purpose
RANDOM_EVAL_BS = 1024
