import torch
import numpy as np
import pickle
import random
from config import *


class SolidData(torch.utils.data.Dataset):
    """
    Dataset for solid
    """

    def __init__(self, path):
        # Load vector data
        print('Loading Data...')
        with open(path, 'rb') as f:  # train_deduplicate.pkl
            dataset = pickle.load(f)

        # Filter data for training
        self.data = []
        for data in dataset:
            cuboids = data['solid']
            if len(cuboids) <= MAX_SOLID:  # MAX_SOLID=5
                op = cuboids[:, 0]  # boolean operation              #(N)
                num_box = len(cuboids)  # number of boxes            #num_box = N    操作类型extrude
                xyz_min, xyz_max = self.get_corners(cuboids[:,
                                                    1:])  # all corners [x_min, x_max, y_min, y_max, z_min, z_max]   xyz_min(N,3)  xyz_max(N,3)
                vec_data = {}
                vec_data['cmd'] = op  # N
                vec_data['param'] = np.concatenate([xyz_min, xyz_max], 1)  # (N,6)
                vec_data['num'] = num_box  # N
                vec_data['name'] = data['name'].split('/')[-1]
                self.data.append(vec_data)  # 处理数据，用xyz,最小点和xyz距离保存点记录边界框。

        # print(f'Post-Filter: {len(self.data)}, Keep Ratio: {100*len(self.data)/len(dataset):.2f}%')

    def get_corners(self, boxes):
        x_min, x_max, y_min, y_max, z_min, z_max = boxes.T
        xyz_min = np.concatenate([x_min[:, np.newaxis], y_min[:, np.newaxis], z_min[:, np.newaxis]],
                                 1)  # [[x,y,z] ... [x,y,z]],  Nx3
        x_span, y_span, z_span = x_max - x_min, y_max - y_min, z_max - z_min
        xyz_span = np.concatenate([x_span[:, np.newaxis], y_span[:, np.newaxis], z_span[:, np.newaxis]],
                                  1)  # Nx3   # xyz_min
        return xyz_min, xyz_span

    def __len__(self):
        return len(self.data)

    def pad_data(self, cmd, param):
        keys = np.ones(len(cmd))
        padding = np.zeros(MAX_SOLID - len(cmd)).astype(int)
        cmd_pad = np.concatenate([cmd, padding], axis=0)  # 对于长度不够max_solid的进行填充
        mask = 1 - np.concatenate([keys, padding]) == 1
        padding = np.zeros((MAX_SOLID - len(param), SOLID_PARAM_SEQ)).astype(
            int)  # SOLID_PARAM_SEQ=6,solid表示参数要求6，对于剩余部分进行填充。
        param_pad = np.concatenate([param, padding], axis=0)
        return cmd_pad, param_pad, mask  # mask对填充部分进行掩码。

    def __getitem__(self, index):
        vec_data = self.data[index]  # vec_data = (N,6)
        num_box = vec_data['num']  # N

        cmd_pad, param_pad, seq_mask = self.pad_data(vec_data['cmd'],
                                                     vec_data['param'])  # cmd_pad=(5)   param_pad=(5,6)  seq_mask=(5)

        # Random masking (per token)
        num_token = num_box * SOLID_PARAM_SEQ  #
        masked_ratio = random.uniform(MASK_RATIO_LOW, MASK_RATIO_HIGH)
        len_keep = np.clip(round(num_token * (1 - masked_ratio)), a_min=1, a_max=num_token - 1)
        noise = np.random.random(num_token)  # noise in [0, 1]
        ids_shuffle = np.argsort(noise)  # ascend: small is keep, large is remove     np.argsort返回元素从小到大排序后的索引值。
        ids_keep = ids_shuffle[:len_keep]
        ids_masked = list(set(ids_shuffle) - set(ids_keep))  # set()接受list，不包含重复元素。 ids_masked保存没有被keep下来的值。

        ignore_mask = np.repeat(np.copy(seq_mask), SOLID_PARAM_SEQ)
        for masked in ids_masked:
            ignore_mask[masked] = True
        ignore_mask = ignore_mask.reshape(-1, SOLID_PARAM_SEQ)
        ignore_mask[num_box:] = False  # ignore对于没有被keep下来的值设置掩码。
        return param_pad, seq_mask, ignore_mask, vec_data['name']  # param_pad=(5,6) seq_mask=(5), ignore_mask=(5,6)


class ProfileData(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # Load vector data
        print('Loading Data...')
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        # Filter data for training
        self.data = []
        for data in dataset:
            bboxs = np.array(data['profile'])
            num_bbox = len(bboxs)

            if num_bbox <= MAX_PROFILE:
                corners = self.get_corners(bboxs)  # all four corner coordinates
                vec_data = {}
                vec_data['coords'] = corners
                vec_data['num'] = num_bbox
                vec_data['name'] = data['uid']
                self.data.append(vec_data)

        # print(f'Post-Filter: {len(self.data)}, Keep Ratio: {100*len(self.data)/len(dataset):.2f}%')

    def get_corners(self, boxes):
        x_min, y_min, x_max, y_max = boxes.T
        x_min, x_max = np.minimum(x_min, x_max), np.maximum(x_min, x_max)
        y_min, y_max = np.minimum(y_min, y_max), np.maximum(y_min, y_max)
        x_span, y_span = x_max - x_min, y_max - y_min
        xywh = np.concatenate(
            [x_min[:, np.newaxis], y_min[:, np.newaxis], x_span[:, np.newaxis], y_span[:, np.newaxis]], 1)
        return xywh

    def __len__(self):
        return len(self.data)

    def pad_coord(self, tokens):
        keys = np.ones(len(tokens))
        padding = np.zeros((MAX_PROFILE - len(tokens))).astype(int)
        seq_mask = 1 - np.concatenate([keys, padding]) == 1
        padding = np.zeros((MAX_PROFILE - len(tokens), PROFILE_PARAM_SEQ)).astype(int)
        tokens = np.concatenate([tokens, padding], axis=0)
        return tokens, seq_mask

    def __getitem__(self, index):
        vec_data = self.data[index]
        num_bbox = vec_data['num']
        corners = vec_data['coords']
        coord_pad, seq_mask = self.pad_coord(corners)

        # Random masking (corner)
        num_token = num_bbox * PROFILE_PARAM_SEQ
        masked_ratio = random.uniform(MASK_RATIO_LOW, MASK_RATIO_HIGH)
        len_keep = np.clip(round(num_token * (1 - masked_ratio)), a_min=1, a_max=num_token - 1)
        noise = np.random.random(num_token)  # noise in [0, 1]
        ids_shuffle = np.argsort(noise)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:len_keep]
        ids_masked = list(set(ids_shuffle) - set(ids_keep))

        ignore_mask = np.repeat(np.copy(seq_mask), PROFILE_PARAM_SEQ)
        for masked in ids_masked:
            ignore_mask[masked] = True
        ignore_mask = ignore_mask.reshape(-1, PROFILE_PARAM_SEQ)
        ignore_mask[num_bbox:] = False

        return coord_pad, seq_mask, ignore_mask, vec_data['name']


class LoopData(torch.utils.data.Dataset):
    """ Single loop dataset """

    def __init__(self, path):
        with open(path, 'rb') as f:
            dataset = pickle.load(f)

        # Filter data
        self.data = []
        for data in dataset:
            param_full = data['param']

            tokens = []
            for pp in param_full:
                tokens.append(pp)
                tokens.append(np.array([-1, -1]))
            # EOS
            tokens.append(np.array([-2, -2]))
            tokens = np.vstack(tokens)
            tokens = tokens + LOOP_PARAM_PAD

            if len(tokens) > MAX_LOOP:
                continue
            vec_data = {}
            vec_data['coord'] = tokens
            vec_data['name'] = data['uid']
            self.data.append(vec_data)

        # print(f'Post-Filter: {len(self.data)}, Keep Ratio: {100*len(self.data)/len(dataset):.2f}%')

    def __len__(self):
        return len(self.data)

    def pad_pixel(self, tokens):
        keys = np.ones(len(tokens))
        padding = np.zeros((MAX_LOOP - len(tokens))).astype(int)
        seq_mask = 1 - np.concatenate([keys, padding]) == 1
        tokens = np.concatenate([tokens, padding], axis=0)
        return tokens, seq_mask

    def pad_token(self, tokens):
        keys = np.ones(len(tokens))
        padding = np.zeros(MAX_LOOP - len(tokens)).astype(int)
        seq_mask = 1 - np.concatenate([keys, padding]) == 1
        padding = np.zeros((MAX_LOOP - len(tokens), 2)).astype(int)
        tokens = np.concatenate([tokens, padding], axis=0)
        return tokens, seq_mask

    def __getitem__(self, index):
        vec_data = self.data[index]
        coords = vec_data['coord']
        num_param = len(coords)

        coords, seq_mask = self.pad_token(coords)

        # Random masking (per token)
        num_token = num_param * LOOP_PARAM_SEQ
        masked_ratio = random.uniform(MASK_RATIO_LOW, MASK_RATIO_HIGH)
        len_keep = np.clip(round(num_token * (1 - masked_ratio)), a_min=1, a_max=num_token - 1)
        noise = np.random.random(num_token)  # noise in [0, 1]
        ids_shuffle = np.argsort(noise)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:len_keep]
        ids_masked = list(set(ids_shuffle) - set(ids_keep))

        ignore_mask = np.repeat(np.copy(seq_mask), LOOP_PARAM_SEQ)
        for masked in ids_masked:
            ignore_mask[masked] = True
        ignore_mask = ignore_mask.reshape(-1, LOOP_PARAM_SEQ)
        ignore_mask[num_param:] = False


        return coords, seq_mask, ignore_mask, vec_data['name'] 
