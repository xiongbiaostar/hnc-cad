

# Hierarchical Neural Coding for Controllable CAD Model Generation (ICML 2023)

[![arXiv](https://img.shields.io/badge/📃-arXiv%20-red.svg)](https://arxiv.org/abs/2307.00149)
[![webpage](https://img.shields.io/badge/🌐-Website%20-blue.svg)](https://hnc-cad.github.io) 
[![Youtube](https://img.shields.io/badge/📽️-Video%20-orchid.svg)](https://www.youtube.com/watch?v=1XVUJIKioO4)

*[Xiang Xu](https://samxuxiang.github.io/), [Pradeep Kumar Jayaraman](https://www.research.autodesk.com/people/pradeep-kumar-jayaraman/), [Joseph G. Lambourne](https://www.research.autodesk.com/people/joseph-george-lambourne/), [Karl D.D. Willis](https://www.karlddwillis.com/), [Yasutaka Furukawa](https://www.cs.sfu.ca/~furukawa/)*

![alt HNCode](resources/teaser.png)

> We present a novel generative model for
Computer Aided Design (CAD) that 1) represents high-level design concepts of a CAD model as a
three-level hierarchical tree of neural codes, from global part arrangement down to local curve geometry; and 2) controls the generation of CAD models by specifying the target design using a code tree. Our method supports diverse and higher-quality generation; novel user controls while specifying design intent; and autocompleting a partial CAD model under construction.


## Requirements

### Environment
- Linux
- Python 3.8
- CUDA >= 11.4
- GPU with 24 GB ram recommended

### Dependencies
- PyTorch >= 1.10
- Install pythonocc following the instruction [here](https://github.com/tpaviot/pythonocc-core) (use mamba if conda is too slow).
- Install other dependencies with ```pip install -r requirements.txt```

We also provide the [docker image](https://hub.docker.com/r/samxuxiang/skexgen). Note: only tested on CUDA 11.4. 

## codebook训练

这部分代码在文件夹codebook下面。分别训练profile和loop的codebook，读取data/loop/train.pkl数据。config.py中对应模型的参数设置。

```
python codebook/train.py --output proj_log/profile --batchsize 256 --format profile --device 0
python codebook/train.py --output proj_log/loop --batchsize 256 --format loop --device 0
```

根据训练好的模型提取出profile.pkl和loop.pkl

```
python codebook/extract_code.py --checkpoint proj_log/profile --format profile --epoch 250 --device 0
python codebook/extract_code.py --checkpoint proj_log/loop --format loop --epoch 250 --device 0
```



## 预测模型训练

这部分代码在文件夹gen下面，主要函数是cond_train.py和cond_generation.分别进行训练和预测。数据处理在dataset.py中CADData类。config.py中COND_TRAIN_EPOCH设置训练epoch和其他参数。

```
python gen/cond_train.py --output proj_log/gen_full --batchsize 256 --profile_code  profile.pkl --loop_code loop.pkl --mode cond --device 0
```

读取训练好模型预测，

```
python gen/cond_generatino.py --output result/ac_test --weight proj_log/gen_full --profile_code  profile.pkl --loop_code loop.pkl --mode cond --device 0
```





