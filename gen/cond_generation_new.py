import os
import torch
import argparse
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as MatplotlibPolygon
from config import *
from hashlib import sha256
import numpy as np
from dataset import CADData
from utils import CADparser, write_obj_sample
from model.encoder import SketchEncoder, ExtEncoder
from model.decoder import SketchDecoder, ExtDecoder, CodeDecoder
from shapely.geometry import Polygon
from cleanfid import fid
import cv2

IMAGE_SIZE = 128
real_dir = "real_images_aug_fivres"
fake_dir = "fake_images_aug_fivres"
os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

cover_count = 0
cover_allcount = 0

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

# def coord2param(coord_full, SKETCH_PAD):
#     coord_full = coord_full - SKETCH_PAD
#
#     params = []
#     param = []
#     for i in range(0, len(coord_full)):
#         if np.array_equal(coord_full[i], np.array([-9, -9])):
#             break
#         if np.array_equal(coord_full[i], np.array([-8, -8])):
#             params.append(param)
#             param = []
#             continue
#         if np.all(coord_full[i] <= np.array([-1, -1])) and np.all(coord_full[i] >= np.array([-7, -7])):
#             param.append(coord_full[i] + 7)
#             continue
#         param.append(coord_full[i])
#     return params

def coord2param(coord_full, type_full, SKETCH_PAD):
    coord_full = coord_full - SKETCH_PAD
    type_full = type_full - TYPE_PAD
    typelen = len(type_full) - 1
    params = []
    param = []
    index = 0
    param.append(np.array([type_full[index], type_full[index]]))
    index += 1
    for i in range(0, len(coord_full)):
        if np.array_equal(coord_full[i], np.array([-2, -2])):
            break
        if np.array_equal(coord_full[i], np.array([-1, -1])):
            params.append(param)
            param = []
            param.append(np.array([type_full[min(index, typelen)], type_full[min(index, typelen)]]))
            index += 1
            continue
        param.append(coord_full[i])
    return params

def draw_polygon(ax, points_list, colors):
    for index, points in enumerate(points_list):
        room_points = points[1:]
        color = colors[points[0][0]]

        # 将多边形点添加到 Patch
        polygon = MatplotlibPolygon(room_points, closed=True, facecolor=color, alpha=0.3, edgecolor=color)
        ax.add_patch(polygon)

        for j in range(len(room_points)):
            next_index = (j + 1) % len(room_points)
            ax.plot([room_points[j][0], room_points[next_index][0]],
                    [room_points[j][1], room_points[next_index][1]],
                    'o-', color=color)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

def draw_polygon_GT_image(points_list, save_folder, colors=['b', 'g', 'r', 'c', 'm', 'y', 'k']):
    # 创建一个新的图和轴
    fig, ax = plt.subplots()

    for index, points in enumerate(points_list):
        room_points = points[1:]
        color = colors[points[0][0]-1]

        # 将多边形点添加到 Patch
        polygon = MatplotlibPolygon(room_points, closed=True, facecolor=color, alpha=0.3, edgecolor=color)
        ax.add_patch(polygon)

        for j in range(len(room_points)):
            next_index = (j + 1) % len(room_points)
            ax.plot([room_points[j][0], room_points[next_index][0]],
                    [room_points[j][1], room_points[next_index][1]],
                    'o-', color=color)

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    # 保存图像到指定路径
    plt.savefig(save_folder)
    plt.close()


def draw_polygon_image( points_list, save_folder,colors=['b', 'g', 'r', 'c', 'm', 'y', 'k']):
    fig, ax = plt.subplots()
    for index, points in enumerate(points_list):
        room_points = points[1:]
        color = colors[points[0][0]]

        # 将多边形点添加到 Patch
        polygon = MatplotlibPolygon(room_points, closed=True, facecolor=color, alpha=0.3, edgecolor=color)
        ax.add_patch(polygon)

        for j in range(len(room_points)):
            next_index = (j + 1) % len(room_points)
            ax.plot([room_points[j][0], room_points[next_index][0]],
                    [room_points[j][1], room_points[next_index][1]],
                    'o-', color=color)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    plt.savefig(save_folder)
    plt.close()

def draw_polygon_GT(ax, points_list, colors):
    for index, points in enumerate(points_list):
        room_points = points[1:]
        color = colors[points[0][0] - 1]

        # 将多边形点添加到 Patch
        polygon = MatplotlibPolygon(room_points, closed=True, facecolor=color, alpha=0.3, edgecolor=color)
        ax.add_patch(polygon)

        for j in range(len(room_points)):
            next_index = (j + 1) % len(room_points)
            ax.plot([room_points[j][0], room_points[next_index][0]],
                    [room_points[j][1], room_points[next_index][1]],
                    'o-', color=color)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

def plot(boundaries, param_ori, param_pred_sel1, param_pred_sel2, save_folder, name, colors=['b', 'g', 'r', 'c', 'm', 'y', 'k']):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    draw_polygon_GT(axs[0, 0], boundaries, colors)
    draw_polygon(axs[0, 1], param_ori, colors)
    draw_polygon(axs[1, 0], param_pred_sel1, colors)
    draw_polygon(axs[1, 1], param_pred_sel2, colors)

    axs[0, 0].set_title('GT')
    axs[0, 1].set_title('Outer contour')
    axs[1, 0].set_title('Predict 1')
    axs[1, 1].set_title('Predict 2')

    color_dict = {'b': 'exterior wall', 'g': 'living room', 'r': 'bedroom', 'c': 'kitchen', 'm': 'bathroom', 'y': 'balcony', 'k': 'Storage'}
    for color, label in color_dict.items():
        axs[0, 1].scatter([], [], c=color, label=label)
    axs[0, 1].legend(loc='upper right', bbox_to_anchor=(1.05, 1), title="Legend")

    try:
        coverage_sel1, total_overlap_area_sel1, total_outside_area_sel1 = calculate_coverage(param_pred_sel1)
        coverage_sel2, total_overlap_area_sel2, total_outside_area_sel2 = calculate_coverage(param_pred_sel2)
        global cover_count, cover_allcount
        if coverage_sel1 == 1 and  total_overlap_area_sel1 == 0 and  total_outside_area_sel1 == 0:
             cover_count += 1
        if coverage_sel2 == 1 and total_overlap_area_sel2 == 0 and total_outside_area_sel2 == 0:
             cover_count += 1
        cover_allcount += 2
        # 在子图上添加文本
        axs[1, 0].text(0.05, 0.9,
                       f"Coverage: {coverage_sel1:.4f}\noverlap between rooms: {total_overlap_area_sel1:.4f}\nbeyond the exterior wall: {total_outside_area_sel1:.4f}",
                       ha='left', va='top', transform=axs[1, 0].transAxes)
        axs[1, 1].text(0.05, 0.9,
                       f"Coverage: {coverage_sel2:.4f}\noverlap between rooms: {total_overlap_area_sel2:.4f}\nbeyond the exterior wall: {total_outside_area_sel2:.4f}",
                       ha='left', va='top', transform=axs[1, 1].transAxes)
    except Exception as e:
        print(e)

    save_path = os.path.join(save_folder, f"room_{int(name[0])}.png")
    plt.savefig(save_path)
    plt.close()

def calculate_coverage(data):

    wall_polygon = Polygon(data[0][1:])
    wall_area = wall_polygon.area

    room_polygons = [Polygon(room[1:]) for room in data[1:]]
    #房间和外墙之间交集
    room_intersections = [wall_polygon.intersection(room).area for room in room_polygons]
    total_room_area = sum(room_intersections)

    #房间和外墙之间差集
    outside_areas = [room.difference(wall_polygon).area for room in room_polygons]
    total_outside_area = sum(outside_areas)

    #判断房间之间重叠
    overlap_areas = []
    for i in range(1, len(data)):
        for j in range(i + 1, len(data)):
            overlap = Polygon(data[i][1:]).intersection(Polygon(data[j][1:])).area
            overlap_areas.append(overlap)
    total_overlap_area = sum(overlap_areas)

    coverage = (total_room_area - total_overlap_area) / wall_area
    overlap_percent = total_overlap_area / wall_area
    outside_percent = total_outside_area / wall_area

    return coverage, overlap_percent, outside_percent

@torch.inference_mode()
def sample(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    dataset = CADData(PROFILE_TEST_PATH,
                      LOOP_TEST_PATH,
                      args.profile_code,
                      args.loop_code,
                      args.mode,
                      ori_param=True,
                      is_training=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=False,
                                             batch_size=1,
                                             num_workers=1)
    code_size = dataset.profile_unique_num + dataset.loop_unique_num

    # Load model weights
    sketch_enc = SketchEncoder()
    sketch_enc.load_state_dict(torch.load(os.path.join(args.weight, 'sketch_enc_epoch_1000.pt')))
    sketch_enc.cuda().eval()

    sketch_dec = SketchDecoder(args.mode, num_code=code_size)
    sketch_dec.load_state_dict(torch.load(os.path.join(args.weight, 'sketch_dec_epoch_1000.pt')))
    sketch_dec.cuda().eval()

    code_dec = CodeDecoder(args.mode, code_size)
    code_dec.load_state_dict(torch.load(os.path.join(args.weight, 'code_dec_epoch_1000.pt')))
    code_dec.cuda().eval()

    # Random sampling
    code_bsz = 10  # every partial input samples this many neural codes
    count = 0
    for pixel_p, coord_p, sketch_mask_p, _, _, _, _, _, _, _, name, boundaries in dataloader:
        #if count > 400: break  # only visualize the first 50 examples
        try:
            pixel_p = pixel_p.cuda()
            coord_p = coord_p.cuda()
            sketch_mask_p = sketch_mask_p.cuda()

            # encode partial CAD model
            sketch_latent = sketch_enc(pixel_p, coord_p, sketch_mask_p)

            # generate the neural code tree
            code_sample, type_samples = code_dec.sample(n_samples = code_bsz,
                                          latent_z = sketch_latent.repeat(code_bsz, 1, 1),
                                          latent_mask = sketch_mask_p.repeat(code_bsz, 1),
                                          top_k = 0,
                                          top_p = 0.95)

            # filter code, only keep unique code
            # if len(code_sample) < 3:
            #     continue
            code_unique = {}
            type_unique = []
            for ii in range(len(code_sample)):
                if len(torch.where(code_sample[ii] == 0)[0]) == 0:
                    continue
                code = (code_sample[ii][:torch.where(code_sample[ii] == 0)[0][0] + 1]).detach().cpu().numpy()
                type = (type_samples[ii][:torch.where(code_sample[ii] == 0)[0][0] - 3]).detach().cpu().numpy()
                code_uid = code.tobytes()
                if code_uid not in code_unique:
                    code_unique[code_uid] = code
                    type_unique.append(type)

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
            xy_samples,  _code_, _code_mask_, _latent_z_, _latent_mask_ = sketch_dec.sample(total_code, total_code_mask,pixel_p, coord_p,  sketch_latent, sketch_mask_p,top_k=1, top_p=0)
            if len(xy_samples) >= 5:
                param_pred_sel1 = coord2param(xy_samples[0], type_unique[0], SKETCH_PAD)
                param_pred_sel2 = coord2param(xy_samples[1], type_unique[1], SKETCH_PAD)
                param_pred_sel3 = coord2param(xy_samples[2], type_unique[2], SKETCH_PAD)
                param_pred_sel4 = coord2param(xy_samples[3], type_unique[3], SKETCH_PAD)
                param_pred_sel5 = coord2param(xy_samples[4], type_unique[4], SKETCH_PAD)
            else:
                continue
                #param_pred_sel1 = param_pred_sel2 = coord2param(xy_samples[0], type_unique[0], SKETCH_PAD)
            GT_boundaries = [tensor.squeeze(0).numpy().tolist() for tensor in boundaries]
            coord_ori = coord_p.cpu().numpy()[0]

            param_ori = coord2param(coord_ori, np.array([1]), SKETCH_PAD)
            #plot(GT_boundaries, param_ori, param_pred_sel1, param_pred_sel2, result_folder, name)
            param_pred_sel = [param_pred_sel1, param_pred_sel2, param_pred_sel3, param_pred_sel4, param_pred_sel5]
            for pred in param_pred_sel:
                global cover_count, cover_allcount
                coverage_sel, total_overlap_area_sel, total_outside_area_sel = calculate_coverage(pred)
                if coverage_sel == 1 and total_overlap_area_sel == 0 and total_outside_area_sel == 0:
                    cover_count += 1
                cover_allcount += 1
            #coverage_sel1, total_overlap_area_sel1, total_outside_area_sel1 = calculate_coverage(param_pred_sel1)
            #coverage_sel2, total_outside_area_sel2, total_overlap_area_sel2 = calculate_coverage(param_pred_sel2)
            # print("predict1:", f"覆盖度: {coverage_sel1:.4f}", f"房间之间的重叠度: {total_overlap_area_sel1:.4f}", f"房间超出外墙部分: {total_outside_area_sel1:.4f}")
            # print("predict2:", f"覆盖度: {coverage_sel2:.4f}", f"房间之间的重叠度: {total_overlap_area_sel2:.4f}", f"房间超出外墙部分: {total_outside_area_sel2:.4f}")

            draw_polygon_GT_image(GT_boundaries, os.path.join(real_dir, f"{int(name[0])}_gt.png"))
            draw_polygon_image(param_pred_sel1, os.path.join(fake_dir, f"{int(name[0])}_pred1.png"))
            draw_polygon_image(param_pred_sel2, os.path.join(fake_dir, f"{int(name[0])}_pred2.png"))
            draw_polygon_image(param_pred_sel3, os.path.join(fake_dir, f"{int(name[0])}_pred3.png"))
            draw_polygon_image(param_pred_sel4, os.path.join(fake_dir, f"{int(name[0])}_pred4.png"))
            draw_polygon_image(param_pred_sel5, os.path.join(fake_dir, f"{int(name[0])}_pred5.png"))

            count += 1
            print("cover_count =", cover_count)
            print("cover_allcount = ", cover_allcount)
            print("rating =", cover_count/cover_allcount)
        except Exception as e:
            print(e)
    fid_score = fid.compute_fid("real_images_aug_fivres", "fake_images_aug_fivres")
    print("FID Score:", fid_score)

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