import torch
import numpy as np
import pickle
from config import *
from tqdm import tqdm
import random


def normalize_points(boundaries, quantization_unit=2 ** CAD_BIT):
    """
    对所有房间角点进行统一归一化和量化（[0, quantization_unit - 1]），
    最后一个点（类别）保留原值，不参与归一化。
    """
    all_points = []

    # 收集所有非类别点
    for room in boundaries:
        if len(room) < 2:
            return []
        points = room[:-1]
        all_points.append(points)
    if not all_points:
        return []
    # 拼接所有点为一个大矩阵
    all_points = np.concatenate(all_points, axis=0)

    min_val = all_points.min()
    max_val = all_points.max()

    if min_val < 0 or max_val > (2 ** CAD_BIT - 1) or min_val >= max_val:
        return []

    # 全局归一化
    def quantize(p):
        return np.clip(((p - min_val) / (max_val - min_val) * (quantization_unit - 1)).astype(np.int32),
                       0, quantization_unit - 1)

    # 处理每个房间
    processed_boundaries = []
    for room in boundaries:
        points = room[:-1]
        room_type = room[-1]
        quantized_points = quantize(points)
        processed = np.vstack([quantized_points, room_type])  # 拼接回来
        processed_boundaries.append(processed)

    return processed_boundaries


class CADData(torch.utils.data.Dataset):
    """ CAD dataset """

    def __init__(self, room_path, boundary_path, profile_path, loop_path, mode, ori_param, is_training=True, ):
        # Load data
        with open(room_path, 'rb') as f:  # profile/train.py
            room_data = pickle.load(f)

        with open(boundary_path, 'rb') as f:  # loop/train.py
            boundaries_data = pickle.load(f)

        with open(profile_path, 'rb') as f:  # profile.pkl
            profile_data = pickle.load(f)
        self.profile_code = profile_data['content']

        with open(loop_path, 'rb') as f:  # loop.pkl
            loop_data = pickle.load(f)
        self.loop_code = loop_data['content']

        self.profile_unique_num = profile_data['unique_num']
        self.loop_unique_num = loop_data['unique_num']
        self.mode = mode
        self.is_training = is_training
        self.ori_param = ori_param

        # Find matching codes
        self.data = []
        boundaries_dict = {item['uid']: item['param'] for item in boundaries_data}
        print("Loading dataset...")
        for room in tqdm(room_data):
            if is_training:
                sketchProfileCode = []
                sketchLoopCode = []
                valid = True

                profile_uid = room['uid']
                if profile_uid not in self.profile_code:
                    valid = False
                    continue
                profile_code = self.profile_code[profile_uid] + self.loop_unique_num  # profile code index
                sketchProfileCode.append(profile_code)

                # LOOP code
                loop_codes = []
                boundaries = []
                num_loop = len(room['profile'])
                for idx_loop in range(num_loop):
                    loop_uid = profile_uid + '_' + str(idx_loop)
                    if loop_uid not in self.loop_code:
                        valid = False
                        continue
                    if loop_uid in boundaries_dict:
                        param = boundaries_dict[loop_uid]
                        boundaries.append(param)
                    loop_code = self.loop_code[loop_uid]  # Loop code index
                    loop_codes.append(loop_code)
                sketchLoopCode.append(loop_codes)
                boundaries = normalize_points(boundaries)
                if boundaries == []:
                    continue

                if not valid:
                    continue
            else:
                sketchProfileCode = []
                sketchLoopCode = []
                boundaries = []
                profile_uid = room['uid']
                num_loop = len(room['profile'])
                for idx_loop in range(num_loop):
                    loop_uid = profile_uid + '_' + str(idx_loop)
                    if loop_uid in boundaries_dict:
                        param = boundaries_dict[loop_uid]
                        boundaries.append(param)
                boundaries = normalize_points(boundaries)
                if boundaries == []:
                    continue

            types = [profile[0] for profile in room['profile']]
            # Global cad parameters
            pixel_full, coord_full = self.param2pix(boundaries)

            # Hierarchical codes (improved)
            total_code = []
            for bbox_code, loops in zip(sketchProfileCode, sketchLoopCode):
                total_code += [-1]  # loop
                total_code += [bbox_code]
                total_code += [-2]  # bbox
                total_code += loops
            total_code += [-3]  # END of cuboid
            total_code = np.array(total_code) + CODE_PAD

            if len(pixel_full) > MAX_CAD or len(total_code) > MAX_CODE:
                continue

            # Pad data
            pixels, sketch_mask = self.pad_pixel(pixel_full)
            coords = self.pad_coord(coord_full)
            total_code, code_mask = self.pad_code(total_code)
            total_types, types_mask = self.pad_type(types)

            vec_data = {}
            vec_data['pixel'] = pixels
            vec_data['coord'] = coords
            vec_data['sketch_mask'] = sketch_mask
            vec_data['code'] = total_code
            vec_data['code_mask'] = code_mask
            vec_data['param'] = boundaries
            vec_data['types'] = total_types
            vec_data['types_mask'] = types_mask
            vec_data['name'] = profile_uid

            self.data.append(vec_data)

    def param2pix_par(self, boundaries):
        pixel_full = []
        coord_full = []

        for i, boundary in enumerate(boundaries):
            # Sketch
            coords = []
            pixels = []

            for param in boundary[1:]:
                coords.append(param)

            coords.append(np.array([-1, -1]))

            for xy in coords:
                if xy[0] < 0:
                    pixels.append(xy[0])
                else:
                    pixels.append(xy[1] * (2 ** CAD_BIT) + xy[0])

            pixel_full.append(pixels)
            coord_full.append(coords)

        coord_full.append(np.array([-2, -2]))  # profile结束标志
        pixel_full += [-2]

        coord_full = np.vstack(coord_full) + SKETCH_PAD
        pixel_full = np.hstack(pixel_full) + SKETCH_PAD

        return pixel_full, coord_full

    def param2pix(self, boundaries):
        pixel_full = []
        coord_full = []

        for i, boundary in enumerate(boundaries):
            # Sketch
            coords = []
            pixels = []

            for param in boundary[1:]:
                coords.append(param)

            coords.append(np.array([-1, -1]))

            for xy in coords:
                if xy[0] < 0:
                    pixels.append(xy[0])
                else:
                    pixels.append(xy[1] * (2 ** CAD_BIT) + xy[0])

            pixel_full.append(pixels)
            coord_full.append(coords)

        coord_full.append(np.array([-2, -2]))  # profile结束标志
        pixel_full += [-2]

        coord_full = np.vstack(coord_full) + SKETCH_PAD
        pixel_full = np.hstack(pixel_full) + SKETCH_PAD

        return pixel_full, coord_full

    def pad_pixel(self, tokens):
        keys = np.ones(len(tokens))
        padding = np.zeros((MAX_CAD - len(tokens))).astype(int)
        seq_mask = 1 - np.concatenate([keys, padding]) == 1
        tokens = np.concatenate([tokens, padding], axis=0)
        return tokens, seq_mask

    def pad_coord(self, tokens):
        padding = np.zeros((MAX_CAD - len(tokens), 2)).astype(int)
        tokens = np.concatenate([tokens, padding], axis=0)
        return tokens

    def pad_code(self, total_code):
        keys = np.ones(len(total_code))
        padding = np.zeros(MAX_CODE - len(total_code)).astype(int)
        total_code = np.concatenate([total_code, padding], axis=0)
        seq_mask = 1 - np.concatenate([keys, padding]) == 1
        return total_code, seq_mask

    def pad_type(self, types):
        keys = np.ones(len(types))
        padding = np.zeros(MAX_TYPE - len(types)).astype(int)
        total_types = np.concatenate([types, padding], axis=0)
        seq_mask = 1 - np.concatenate([keys, padding]) == 1
        return total_types, seq_mask

    def pad_ext(self, tokens):
        keys = np.ones(len(tokens))
        padding = np.zeros((MAX_EXT - len(tokens))).astype(int)
        seq_mask = 1 - np.concatenate([keys, padding]) == 1
        tokens = np.concatenate([tokens, padding], axis=0)
        return tokens, seq_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vec_data = self.data[index]
        sketch_mask = vec_data['sketch_mask']
        code = vec_data['code']
        code_mask = vec_data['code_mask']
        param = vec_data['param']
        types = vec_data['types']
        types_mask = vec_data['types_mask']

        # if self.ori_param is False:
        #     # Random masking
        #     num_token = len(param)
        #     masked_ratio = random.uniform(MASK_RATIO_LOW, MASK_RATIO_HIGH)
        #     len_keep = np.clip(round(num_token * (1 - masked_ratio)), a_min=1, a_max=num_token - 1)
        #     noise = np.random.random(num_token)  # noise in [0, 1]
        #     ids_shuffle = np.argsort(noise)  # ascend: small is keep, large is remove
        #     ids_keep = sorted(ids_shuffle[:len_keep])
        # else:
        ids_keep = [0]  # keep first one and autocomplete the rest

        # Partial SE
        param_partial = [param[id] for id in ids_keep]
        pixel_partial, coord_partial= self.param2pix_par(param_partial)
        pixels_par, sketch_mask_par = self.pad_pixel(pixel_partial)
        coords_par = self.pad_coord(coord_partial)
        pixels = vec_data['pixel']
        coords = vec_data['coord']
        if self.ori_param:
            return pixels_par, coords_par, sketch_mask_par, pixels, coords,  sketch_mask, code, code_mask, types, types_mask, vec_data['name'], vec_data['param']
        else:
            return pixels_par, coords_par,  sketch_mask_par, pixels, coords,  sketch_mask, code, code_mask, types, types_mask, vec_data['name']



class CodeData(torch.utils.data.Dataset):
    """ Code Tree dataset """
    def __init__(self, cad_path, solid_path, profile_path, loop_path):   
        # Load data
        with open(cad_path, 'rb') as f:
            cad_data = pickle.load(f)

        with open(solid_path, 'rb') as f:
            solid_data = pickle.load(f)
        self.solid_code = solid_data['content']
        
        with open(profile_path, 'rb') as f:
            profile_data = pickle.load(f)
        self.profile_code = profile_data['content']

        with open(loop_path, 'rb') as f:
            loop_data = pickle.load(f)
        self.loop_code = loop_data['content']

        self.solid_unique_num = solid_data['unique_num']
        self.profile_unique_num = profile_data['unique_num']
        self.loop_unique_num = loop_data['unique_num']

        # Find matching codes
        self.data = []
        print('Loading data...')
        for cad in tqdm(cad_data):
            # Solid code
            solid_uid = cad['name'].split('/')[-1]
            if solid_uid not in self.solid_code:
                continue 
            solid_code = self.solid_code[solid_uid] + self.loop_unique_num + self.profile_unique_num  # solid code index
            num_se = len(cad['cad_ext'])
                       
            sketchProfileCode = []
            sketchLoopCode = []
            valid = True

            for idx_se in range(num_se):
                # Profile code
                profile_uid = solid_uid+'_'+str(idx_se)                  
                if profile_uid not in self.profile_code:
                    valid = False 
                    continue
                profile_code = self.profile_code[profile_uid] + self.loop_unique_num  # profile code index 
                sketchProfileCode.append(profile_code)

                # LOOP code
                loop_codes = []
                num_loop = len(np.where(cad['cad_cmd'][idx_se]==3)[0])
                for idx_loop in range(num_loop):
                    loop_uid = profile_uid+'_'+str(idx_loop) 
                    if loop_uid not in self.loop_code:
                        valid=False
                        continue
                    loop_code = self.loop_code[loop_uid]  # Loop code index
                    loop_codes.append(loop_code)
                sketchLoopCode.append(loop_codes)

            if not valid:
                continue
          
            # Global cad parameters
            pixel_full, _, _ = self.param2pix(cad)

            # Hierarchical codes (improved)
            total_code = []
            for bbox_code, loops in zip(sketchProfileCode, sketchLoopCode):
                total_code += [-1] # loop
                total_code += [bbox_code]
                total_code += [-2] # bbox
                total_code += loops
            total_code+=[-3] # solid
            total_code += [solid_code]
            total_code+=[-4] # END of cuboid
            total_code = np.array(total_code) + CODE_PAD

            # # Hierarchical codes (breadth)
            # total_code=[-1] # solid
            # total_code += [solid_code]
            # total_code += [-2] # bbox
            # for bbox_code, loops in zip(sketchProfileCode, sketchLoopCode):
            #     total_code += [bbox_code]
            # for bbox_code, loops in zip(sketchProfileCode, sketchLoopCode):
            #     total_code += [-3] # loop
            #     total_code += loops
            # total_code+=[-4] # END of cuboid
            # total_code = np.array(total_code) + CODE_PAD

            if len(total_code) > MAX_CODE or len(pixel_full) > MAX_CAD:
                continue
            total_code = self.pad_code(total_code)
            self.data.append(total_code)

        self.unq_code = np.unique(np.vstack(self.data), return_counts=False, axis=0) # code distribution is uniform
        return

    
    def param2pix(self, cad):
        pixel_full = []
        coord_full = []
        ext_full = []

        for cmd, param, ext in zip(cad['cad_cmd'], cad['cad_param'], cad['cad_ext']):
            # Extrude
            ext_full.append(ext)
            ext_full.append(np.array([-1]))  # Add -1 for normal cad

            # Sketch
            coords = []
            pixels = []
            for cc, pp in zip(cmd, param):
                if cc == 6: # circle 
                    coords.append(pp[0:2])
                    coords.append(pp[2:4])
                    coords.append(pp[4:6])
                    coords.append(pp[6:8])
                    coords.append(np.array([-1,-1]))
                elif cc == 5: # arc
                    coords.append(pp[0:2])
                    coords.append(pp[2:4])
                    coords.append(np.array([-1,-1]))
                elif cc == 4: # line
                    coords.append(pp[0:2])
                    coords.append(np.array([-1,-1]))
                elif cc == 3: # EoL 
                    coords.append(np.array([-2,-2]))
                elif cc == 2: # EoF 
                    coords.append(np.array([-3,-3]))
                elif cc == 1: # EoS 
                    coords.append(np.array([-4,-4]))

            for xy in coords:
                if xy[0] < 0: 
                    pixels.append(xy[0])
                else:
                    pixels.append(xy[1]*(2**CAD_BIT)+xy[0])

            pixel_full.append(pixels)
            coord_full.append(coords)

        ext_full.append(np.array([-2]))
        coord_full.append(np.array([-5,-5]))
        pixel_full += [-5]        
        
        ext_full = np.hstack(ext_full) + EXT_PAD
        coord_full = np.vstack(coord_full) + SKETCH_PAD
        pixel_full = np.hstack(pixel_full) + SKETCH_PAD
        
        return pixel_full, coord_full, ext_full


    def pad_code(self, total_code):
        padding = np.zeros(MAX_CODE-len(total_code)).astype(int)  
        total_code = np.concatenate([total_code, padding], axis=0)
        return total_code

       
    def __len__(self):
        return len(self.unq_code)


    def __getitem__(self, index):
        code = self.unq_code[index]
        code_mask = np.zeros(MAX_CODE)==0
        code_mask[:np.where(code==0)[0][0]+1] = False
        return code, code_mask