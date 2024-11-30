codebook训练
python codebook/train.py --output proj_log/profile --batchsize 256 --format profile --device 0
python codebook/train.py --output proj_log/loop --batchsize 1024 --format loop --device 0

根据训练好的模型提取出profile.pkl和loop.pkl
python codebook/extract_code.py --checkpoint proj_log/profile --format profile --epoch 250 --device 0
python codebook/extract_code.py --checkpoint proj_log/loop --format loop --epoch 250 --device 0

预测模型训练
python gen/door_window_train.py --output proj_log/gen_full --batchsize 128 --profile_code  profile.pkl --loop_code loop.pkl --mode cond --device 0

模型生成，
python gen/door_window_gen.py --output result/ac_test --weight proj_log/gen_full --profile_code  profile.pkl --loop_code loop.pkl --mode cond --device 0

使用tensorboard查看loss
tensorboard --logdir=./proj_log/gen_full
