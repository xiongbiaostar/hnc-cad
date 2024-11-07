import torch
import numpy as np
import pickle
from config import *
from tqdm import tqdm
import random


class CADData(torch.utils.data.Dataset):
    """ CAD dataset """

    def __init__(self, room_path, profile_path, loop_path, mode, is_training=True):
        # Load data
        with open(room_path, 'rb') as f:  # profile/train.py
            room_data = pickle.load(f)

        with open(room_path, 'rb') as f:  # loop/train.py
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

        # Find matching codes
        self.data = []
        print("Loading dataset...")
        for room in tqdm(room_data):
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
            num_loop = len(room_data['profile'])
            for idx_loop in range(num_loop):
                loop_uid = profile_uid + '_' + str(idx_loop)
                if loop_uid not in self.loop_code:
                    valid = False
                    continue
                loop_code = self.loop_code[loop_uid]  # Loop code index
                loop_codes.append(loop_code)
            sketchLoopCode.append(loop_codes)

            if not valid:
                continue

            # Global cad parameters
            pixel_full, coord_full = self.param2pix(boundaries_data)

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

            vec_data = {}
            vec_data['pixel'] = pixels
            vec_data['coord'] = coords
            vec_data['sketch_mask'] = sketch_mask
            vec_data['code'] = total_code
            vec_data['code_mask'] = code_mask
            vec_data['param'] = boundaries_data

            self.data.append(vec_data)

    def param2pix_par(self, boundaries):
        pixel_full = []
        coord_full = []

        # Sketch
        coords = []
        pixels = []
        for param in boundaries['param']:
            coords.append([param])
            coords.append(np.array([-1, -1]))
        coords.append(np.array([-2, -2]))  # loop结束标志
        coords.append(np.array([-3, -3]))  # profile结束标志

        for xy in coords:
            if xy[0] < 0:
                pixels.append(xy[0])
            else:
                pixels.append(xy[1] * (2 ** CAD_BIT) + xy[0])

            pixel_full.append(pixels)
            coord_full.append(coords)

        pixel_full += [-3]

        coord_full = np.vstack(coord_full) + SKETCH_PAD
        pixel_full = np.hstack(pixel_full) + SKETCH_PAD

        return pixel_full, coord_full

    def param2pix(self, boundaries):
        pixel_full = []
        coord_full = []

        # Sketch
        coords = []
        pixels = []
        for param in boundaries['param']:
            coords.append([param])
            coords.append(np.array([-1, -1]))
        coords.append(np.array([-2, -2]))  # loop结束标志
        coords.append(np.array([-3, -3]))  # profile结束标志

        for xy in coords:
            if xy[0] < 0:
                pixels.append(xy[0])
            else:
                pixels.append(xy[1] * (2 ** CAD_BIT) + xy[0])

            pixel_full.append(pixels)
            coord_full.append(coords)

        pixel_full += [-3]

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

        if self.mode == 'uncond':  # unconditional
            # XY augmentation
            aug_xys = []
            for xy in vec_data['coord']:
                if xy[0] < SKETCH_PAD:
                    aug_xys.append(xy - SKETCH_PAD)  # special END tokens
                else:
                    new_xy = xy - SKETCH_PAD
                    new_xy[0] = new_xy[0] + random.randint(-AUG_RANGE, +AUG_RANGE)
                    new_xy[1] = new_xy[1] + random.randint(-AUG_RANGE, +AUG_RANGE)
                    new_xy = np.clip(new_xy, a_min=0, a_max=2 ** CAD_BIT - 1)
                    aug_xys.append(new_xy)
            coords_aug = np.vstack(aug_xys) + SKETCH_PAD

            # PIX augmentation
            aug_pix = []
            for xy in aug_xys:
                if xy[0] >= 0 and xy[1] >= 0:
                    aug_pix.append(xy[1] * (2 ** CAD_BIT) + xy[0])
                else:
                    aug_pix.append(xy[0])
            pixels_aug = np.hstack(aug_pix) + SKETCH_PAD

            pixels_aug, _ = self.pad_pixel(pixels_aug)
            coords_aug = self.pad_coord(coords_aug)
            pixels = vec_data['pixel']
            coords = vec_data['coord']

            return pixels, coords, sketch_mask, pixels_aug, coords_aug, code, code_mask

        else:  # conditional
            assert self.mode == 'cond'
            param = vec_data['param']

            if self.is_training:
                # Random masking
                num_token = len(param)
                masked_ratio = random.uniform(MASK_RATIO_LOW, MASK_RATIO_HIGH)
                len_keep = np.clip(round(num_token * (1 - masked_ratio)), a_min=1, a_max=num_token - 1)
                noise = np.random.random(num_token)  # noise in [0, 1]
                ids_shuffle = np.argsort(noise)  # ascend: small is keep, large is remove
                ids_keep = sorted(ids_shuffle[:len_keep])
            else:
                ids_keep = [0]  # keep first one and autocomplete the rest

            # Partial SE
            param_partial = [param[id] for id in ids_keep]
            pixel_partial, coord_partial = self.param2pix_par(param_partial)
            pixels_par, sketch_mask_par = self.pad_pixel(pixel_partial)
            coords_par = self.pad_coord(coord_partial)
            pixels = vec_data['pixel']
            coords = vec_data['coord']

            return pixels_par, coords_par, sketch_mask_par, pixels, coords, sketch_mask, code, code_mask


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