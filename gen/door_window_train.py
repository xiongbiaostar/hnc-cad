import os
import torch
import argparse
from tqdm import tqdm
from config import *
from dataset import CADData
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model.network import schedule_with_warmup
from model.encoder import SketchEncoder, ExtEncoder
from model.decoder import SketchDecoder, ExtDecoder, CodeDecoder
def train(args):
    # gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:0")
    # Initialize dataset loader
    dataset_train = CADData(CAD_TRAIN_PATH, BOUNDARY_TRAIN_PATH, args.profile_code, args.loop_code, args.mode, is_training=True)

    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                             shuffle=True,
                                             batch_size=args.batchsize,
                                             num_workers=6)

    # dataset_val = CADData(CAD_VAL_PATH, BOUNDARY_VAL_PATH, args.profile_code, args.loop_code, args.mode, is_training=False)
    #
    # dataloader_val = torch.utils.data.DataLoader(dataset_val,
    #                                          shuffle=False,
    #                                          batch_size=args.batchsize,
    #                                          num_workers=6)


    code_size = dataset_train.profile_unique_num + dataset_train.loop_unique_num
    # Initialize models
    sketch_dec = SketchDecoder(args.mode, num_code=code_size)
    sketch_dec = nn.DataParallel(sketch_dec)
    sketch_dec = sketch_dec.to(device).train()
    sketch_enc = SketchEncoder()
    sketch_enc = nn.DataParallel(sketch_enc)
    sketch_enc = sketch_enc.to(device).train()
    code_dec = CodeDecoder(args.mode, code_size)
    code_dec = nn.DataParallel(code_dec)
    code_dec.to(device).train()

    params = list(sketch_enc.parameters()) + list(sketch_dec.parameters()) + list(code_dec.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    scheduler = schedule_with_warmup(optimizer, 2000)
    writer = SummaryWriter(log_dir=args.output)
    # Main training loop
    print('Start training...')
    for epoch in range(COND_TRAIN_EPOCH):
        #train
        sketch_dec.train()
        sketch_enc.train()
        code_dec.train()

        progress_bar = tqdm(total=len(dataloader_train))
        progress_bar.set_description(f"Epoch {epoch}")
        train_loss = 0
        for pixel_p, coord_p, sketch_mask_p, pixel, coord, sketch_mask, code, code_mask, _, _ in dataloader_train:
            pixel_p = pixel_p.to(device)
            coord_p = coord_p.to(device)
            sketch_mask_p = sketch_mask_p.to(device)
            pixel = pixel.to(device)
            coord = coord.to(device)
            sketch_mask = sketch_mask.to(device)
            code = code.to(device)
            code_mask = code_mask.to(device)
            # Partial Token Encoder
            latent_sketch = sketch_enc(pixel_p, coord_p, sketch_mask_p)
            # Pass through sketch decoder
            sketch_logits = sketch_dec(pixel[:, :-1], coord[:, :-1, :], code, code_mask, latent_sketch, sketch_mask_p)
            # Pass through extrude decoder
            # Pass through code decoder
            code_logits = code_dec(code[:, :-1], latent_sketch, sketch_mask_p)
            # Compute losses
            valid_mask = (~sketch_mask).reshape(-1)
            sketch_pred = sketch_logits.reshape(-1, sketch_logits.shape[-1])
            sketch_gt = pixel.reshape(-1)
            sketch_loss = F.cross_entropy(sketch_pred[valid_mask], sketch_gt[valid_mask])
            valid_mask = (~code_mask).reshape(-1)
            code_pred = code_logits.reshape(-1, code_logits.shape[-1])
            code_gt = code.reshape(-1)
            code_loss = F.cross_entropy(code_pred[valid_mask], code_gt[valid_mask])
            train_loss = sketch_loss + code_loss
            # Update model
            optimizer.zero_grad()
            train_loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()  # linear warm up to 1e-3
            progress_bar.update(1)
        progress_bar.close()

        # eval
        sketch_enc.eval()
        sketch_dec.eval()
        code_dec.eval()

        # progress_bar = tqdm(total=len(dataloader_val))
        # progress_bar.set_description('Evaluating')
        val_loss = 0
        # batch_count = 0
        # with torch.no_grad():
        #     for pixel_p, coord_p, sketch_mask_p, pixel, coord, sketch_mask, code, code_mask, _, _ in dataloader_val:
        #         pixel_p = pixel_p.to(device)
        #         coord_p = coord_p.to(device)
        #         sketch_mask_p = sketch_mask_p.to(device)
        #         pixel = pixel.to(device)
        #         coord = coord.to(device)
        #         sketch_mask = sketch_mask.to(device)
        #         code = code.to(device)
        #         code_mask = code_mask.to(device)
        #         # Partial Token Encoder
        #         latent_sketch = sketch_enc(pixel_p, coord_p, sketch_mask_p)
        #         # Pass through sketch decoder
        #         sketch_logits = sketch_dec(pixel[:, :-1], coord[:, :-1, :], code, code_mask, latent_sketch,
        #                                    sketch_mask_p)
        #         # Pass through code decoder
        #         code_logits = code_dec(code[:, :-1], latent_sketch, sketch_mask_p)
        #         # Compute losses
        #         valid_mask = (~sketch_mask).reshape(-1)
        #         sketch_pred = sketch_logits.reshape(-1, sketch_logits.shape[-1])
        #         sketch_gt = pixel.reshape(-1)
        #         sketch_loss = F.cross_entropy(sketch_pred[valid_mask], sketch_gt[valid_mask])
        #         valid_mask = (~code_mask).reshape(-1)
        #         code_pred = code_logits.reshape(-1, code_logits.shape[-1])
        #         code_gt = code.reshape(-1)
        #         code_loss = F.cross_entropy(code_pred[valid_mask], code_gt[valid_mask])
        #         val_loss += (sketch_loss + code_loss).item()
        #         batch_count += 1
        #         progress_bar.update(1)
        # progress_bar.close()
        # val_loss = val_loss / batch_count

        # logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        #writer.add_scalar("Loss/val", val_loss, epoch)
        print(f"Epoch {epoch}: Loss/train = {train_loss:.4f}, Loss/val = {val_loss:.4f}")
        writer.flush()

        # # save model after n epoch
        if (epoch + 1) % 10 == 0:
            torch.save(sketch_dec.module.state_dict(),
                       os.path.join(args.output, 'sketch_dec_epoch_' + str(epoch + 1) + '.pt'))
            torch.save(sketch_enc.module.state_dict(),
                       os.path.join(args.output, 'sketch_enc_epoch_' + str(epoch + 1) + '.pt'))
            torch.save(code_dec.module.state_dict(),
                       os.path.join(args.output, 'code_dec_epoch_' + str(epoch + 1) + '.pt'))
    writer.close()

if __name__ == "__main_s_":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Output folder to save the data", required=True)
    parser.add_argument("--batchsize", type=int, help="Training batchsize", required=True)
    parser.add_argument("--device", type=str, help="CUDA device", required=True)
    parser.add_argument("--profile_code", type=str, required=True, help='Extracted profile codes')
    parser.add_argument("--loop_code", type=str, required=True, help='Extracted loop codes')
    parser.add_argument("--mode", type=str, required=True, help='uncond | cond')
    args = parser.parse_args()
    # Create training folder
    result_folder = args.output
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    # Start training
    train(args)