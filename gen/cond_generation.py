import os
import torch
import argparse
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from config import *
from hashlib import sha256
import numpy as np
from dataset import CADData
from utils import CADparser, write_obj_sample
from model.encoder import SketchEncoder, ExtEncoder
from model.decoder import SketchDecoder, ExtDecoder, CodeDecoder


def raster_cad(coord, ext):
    parser = CADparser(CAD_BIT)
    parsed_data = parser.perform(coord, ext)
    return parsed_data


def pad_code(total_code):
    keys = np.ones(len(total_code))
    padding = np.zeros(MAX_CODE - len(total_code)).astype(int)
    total_code = np.concatenate([total_code, padding], axis=0)
    seq_mask = 1 - np.concatenate([keys, padding]) == 1
    return total_code, seq_mask


def hash_sketch(sketch, ext):
    hash_str = sha256(np.ascontiguousarray(sketch).flatten()).hexdigest() + '_' + \
               sha256(np.ascontiguousarray(ext).flatten()).hexdigest()
    return hash_str


def pix2param(coord_full, SKETCH_PAD):
    coord_full = coord_full - SKETCH_PAD

    params = []
    param = []
    for i in range(0, len(coord_full)-1):
        if np.array_equal(coord_full[i], np.array([-1, -1])):
            continue
        if np.array_equal(coord_full[i], np.array([-2, -2])):
            params.append(param)
            param = []
            continue
        param.append(coord_full[i])

    return params

def draw_polygon(ax, points_list, colors):
    for index, points in enumerate(points_list):
        room_points = points
        color = colors[index % len(colors)]

        # 将多边形点添加到 Patch
        polygon = Polygon(room_points, closed=True, facecolor=color, alpha=0.3, edgecolor=color)
        ax.add_patch(polygon)

        for j in range(len(room_points)):
            next_index = (j + 1) % len(room_points)
            ax.plot([room_points[j][0], room_points[next_index][0]],
                    [room_points[j][1], room_points[next_index][1]],
                    'o-', color=color)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

def plot(boundaries, param_ori, param_pred_sel1, param_pred_sel2, save_folder, name, colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    draw_polygon(axs[0, 0], boundaries, colors)
    draw_polygon(axs[0, 1], param_ori, colors)
    draw_polygon(axs[1, 0], param_pred_sel1, colors)
    draw_polygon(axs[1, 1], param_pred_sel2, colors)

    axs[0, 0].set_title('GT')
    axs[0, 1].set_title('Outer contour')
    axs[1, 0].set_title('Predict 1')
    axs[1, 1].set_title('Predict 2')

    save_path = os.path.join(save_folder, f"room_{int(name[0])}.png")
    plt.savefig(save_path)
    plt.close()

@torch.inference_mode()
def sample(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    dataset = CADData(PROFILE_TEST_PATH,
                      LOOP_TEST_PATH,
                      args.profile_code,
                      args.loop_code,
                      args.mode,
                      is_training=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=False,
                                             batch_size=1,
                                             num_workers=1)
    code_size = dataset.profile_unique_num + dataset.loop_unique_num

    # Load model weights
    sketch_enc = SketchEncoder()
    sketch_enc.load_state_dict(torch.load(os.path.join(args.weight, 'sketch_enc_epoch_750.pt')))
    sketch_enc.cuda().eval()

    sketch_dec = SketchDecoder(args.mode, num_code=code_size)
    sketch_dec.load_state_dict(torch.load(os.path.join(args.weight, 'sketch_dec_epoch_750.pt')))
    sketch_dec.cuda().eval()

    code_dec = CodeDecoder(args.mode, code_size)
    code_dec.load_state_dict(torch.load(os.path.join(args.weight, 'code_dec_epoch_750.pt')))
    code_dec.cuda().eval()

    # Random sampling
    code_bsz = 10  # every partial input samples this many neural codes
    count = 0
    for pixel_p, coord_p, sketch_mask_p, _, _, _, _, _,name, boundaries in dataloader:
        if count > 200: break  # only visualize the first 50 examples

        pixel_p = pixel_p.cuda()
        coord_p = coord_p.cuda()
        sketch_mask_p = sketch_mask_p.cuda()

        # encode partial CAD model
        sketch_latent = sketch_enc(pixel_p, coord_p, sketch_mask_p)

        # generate the neural code tree
        code_sample = code_dec.sample(n_samples = code_bsz,
                                      latent_z = sketch_latent.repeat(code_bsz, 1, 1), 
                                      latent_mask = sketch_mask_p.repeat(code_bsz, 1), 
                                      top_k = 0,
                                      top_p = 0.95)

        # filter code, only keep unique code
        # if len(code_sample) < 3:
        #     continue
        code_unique = {}
        for ii in range(len(code_sample)):
            if len(torch.where(code_sample[ii] == 0)[0]) == 0:
                continue
            code = (code_sample[ii][:torch.where(code_sample[ii] == 0)[0][0] + 1]).detach().cpu().numpy()
            code_uid = code.tobytes()
            if code_uid not in code_unique:
                code_unique[code_uid] = code

        total_code = []
        total_code_mask = []
        for _, code in code_unique.items():
            _code_, _code_mask_ = dataset.pad_code(code)
            total_code.append(_code_)
            total_code_mask.append(_code_mask_)

        total_code = np.vstack(np.vstack(total_code))
        total_code_mask = np.vstack(total_code_mask)
        total_code = torch.LongTensor(total_code).cuda()
        total_code_mask = torch.BoolTensor(total_code_mask).cuda()

        # generate the full CAD model
        sketch_latent = sketch_latent.repeat(len(total_code), 1, 1)
        sketch_mask_p = sketch_mask_p.repeat(len(total_code), 1)
        xy_samples, _code_, _code_mask_, _latent_z_, _latent_mask_ = sketch_dec.sample(total_code, total_code_mask,pixel_p, coord_p, sketch_latent, sketch_mask_p,top_k=1, top_p=0)
        if len(xy_samples) >= 2:
            param_pred_sel1 = pix2param(xy_samples[0], SKETCH_PAD)
            param_pred_sel2 = pix2param(xy_samples[1], SKETCH_PAD)
        else:
            param_pred_sel1 = param_pred_sel2 = pix2param(xy_samples[0], SKETCH_PAD)
        GT_boundaries = [tensor.squeeze(0).numpy().tolist() for tensor in boundaries]
        coord_ori = coord_p.cpu().numpy()[0]
        param_ori = pix2param(coord_ori, SKETCH_PAD)
        plot(GT_boundaries, param_ori, param_pred_sel1, param_pred_sel2, result_folder, name)
        count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, help="Pretrained CAD model", required=True)
    parser.add_argument("--output", type=str, help="Output folder to save the data", required=True)
    parser.add_argument("--device", type=str, help="CUDA Device Index", required=True)
    parser.add_argument("--mode", type=str, required=True, help="eval | sample")
    parser.add_argument("--profile_code", type=str, required=True)
    parser.add_argument("--loop_code", type=str, required=True)
    args = parser.parse_args()

    result_folder = args.output
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    sample(args)