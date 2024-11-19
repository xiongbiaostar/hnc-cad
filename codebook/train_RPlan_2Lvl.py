import os
import torch
import argparse
from tqdm import tqdm
from config import *
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataset import SolidData, ProfileData, LoopData
from model.encoder import SolidEncoder, ProfileEncoder, LoopEncoder
from model.decoder import SolidDecoder, ProfileDecoder, LoopDecoder
from model.network import get_constant_schedule_with_warmup, squared_emd_loss


def parse_aug(format):
    """
    Find the corresponding function to run
    """
    data_func = {
        "solid": SolidData,
        "profile": ProfileData,
        "loop": LoopData
    }[format]

    train_data_path = {
        "solid": SOLID_TRAIN_PATH,
        "profile": PROFILE_TRAIN_PATH,
        "loop": LOOP_TRAIN_PATH
    }[format]

    val_data_path = {
        "solid": SOLID_VAL_PATH,
        "profile": PROFILE_VAL_PATH,
        "loop": LOOP_VAL_PATH
    }[format]

    test_data_path = {
        "solid": SOLID_TEST_PATH,
        "profile": PROFILE_TEST_PATH,
        "loop": LOOP_TEST_PATH
    }[format]

    enc_func = {
        "solid": SolidEncoder,
        "profile": ProfileEncoder,
        "loop": LoopEncoder
    }[format]

    dec_func = {
        "solid": SolidDecoder,
        "profile": ProfileDecoder,
        "loop": LoopDecoder
    }[format]

    return data_func, train_data_path, enc_func, dec_func, val_data_path, test_data_path

from tqdm import tqdm

def evaluate(dataloader, encoder, decoder):
    encoder.eval()
    decoder.eval()

    total_loss = 0
    total_batches = 0

    # Progress bar for evaluation
    progress_bar = tqdm(total=len(dataloader), desc="Evaluating")

    with torch.no_grad():
        for param, seq_mask, ignore_mask, _ in dataloader:
            param = param.cuda()
            seq_mask = seq_mask.cuda()
            ignore_mask = ignore_mask.cuda()

            # Forward pass
            latent_code, vq_loss, _, _ = encoder(param, seq_mask)
            param_logits = decoder(param, seq_mask, ignore_mask, latent_code)

            # Compute loss
            param_loss = squared_emd_loss(
                logits=param_logits,
                labels=param,
                num_classes=param_logits.shape[-1],
                mask=ignore_mask
            )
            total_loss += (param_loss + vq_loss).item()
            total_batches += 1

            # Update progress bar
            progress_bar.update(1)

    progress_bar.close()

    # Compute average loss
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    return avg_loss


def reinit_codebook(dataloader, encoder):
    code_encoded = []
    for param, seq_mask, _, _ in dataloader:
        param = param.cuda()
        seq_mask = seq_mask.cuda()
        with torch.no_grad():
            _, _code_encoded_ = encoder.count_code(param, seq_mask)
            code_encoded.append(_code_encoded_.reshape(-1, 256).detach().cpu())

    code_encoded = torch.vstack(code_encoded)
    code_encoded = code_encoded[torch.randperm(code_encoded.size()[0])]  # Random shuffle
    reinit_count = encoder.codebook.reinit(code_encoded)
    return reinit_count


def train(args):
    # gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    data_func, train_data_path, enc_func, dec_func, val_data_path, test_data_path = parse_aug(args.format)  # 选择对应的数据和ED

    # Initialize traindataset loader
    traindataset = data_func(train_data_path)
    traindataloader = torch.utils.data.DataLoader(traindataset,
                                             shuffle=True,
                                             batch_size=args.batchsize,
                                             num_workers=6)  # (6,1,5,6)
    
    valdataset = data_func(val_data_path)
    valdataloader = torch.utils.data.DataLoader(valdataset,
                                             shuffle=False,
                                             batch_size=args.batchsize,
                                             num_workers=6)  # (6,1,5,6)
    
    testdataset = data_func(test_data_path)
    testdataloader = torch.utils.data.DataLoader(testdataset,
                                             shuffle=False,
                                             batch_size=args.batchsize,
                                             num_workers=6)  # (6,1,5,6)

    # Initialize models
    encoder = enc_func()
    encoder = encoder.cuda().train()

    decoder = dec_func()
    decoder = decoder.cuda().train()

    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    scheduler = get_constant_schedule_with_warmup(optimizer, 2000)  # 学习率预热，先设置小的学习率再设为预设。
    writer = SummaryWriter(log_dir=args.output)

    # Main training loop
    iters = 0
    print('Start training...')
    for epoch in range(TOTAL_TRAIN_EPOCH):
        progress_bar = tqdm(total=len(traindataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        for param, seq_mask, ignore_mask, _ in traindataloader:  # seq_mask是对填充部分的掩码, ignore_mask是随机掩码部分。 param=(6,5,6)(batch_size,max_solid,solid_param_seq)    seq_mask,ignore_mask=(6,5) (batch_size,max_solid)
            param = param.cuda()  # param=(6,20,4)
            seq_mask = seq_mask.cuda()
            ignore_mask = ignore_mask.cuda()

            # Pass through encoder
            latent_code, vq_loss, selection, _ = encoder(param,
                                                         seq_mask)  # latent_code(6,1,256)最近邻选中的向量，selection对应codebook中的索引。

            # Pass through decoder
            param_logits = decoder(param, seq_mask, ignore_mask, latent_code)  # 输入掩码，原参数和对应codebook表示进行解码

            # Compute loss
            param_loss = squared_emd_loss(logits=param_logits,
                                          labels=param,
                                          num_classes=param_logits.shape[-1],
                                          mask=ignore_mask)

            total_loss = param_loss + vq_loss

            # logging
            if iters % 10 == 0:
                writer.add_scalar("Loss/Train_Total", total_loss.item(), iters)
                writer.add_scalar("Loss/Train_Coord", param_loss.item(), iters)
                writer.add_scalar("Loss/Train_VQ", vq_loss.item(), iters)

            if iters % 20 == 0 and selection is not None:
                writer.add_histogram('selection', selection, iters)

            # Update model
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)  # clip gradient
            optimizer.step()
            scheduler.step()  # linear warm up to 1e-3
            iters += 1
            progress_bar.update(1)

        progress_bar.close()

        # Evaluate on validation and test sets
        val_loss = evaluate(valdataloader, encoder, decoder)
        test_loss = evaluate(testdataloader, encoder, decoder)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Loss/Test", test_loss, epoch)
        print(f"Epoch {epoch}: Val_Loss = {val_loss:.4f}, Test_Loss = {test_loss:.4f}")
        writer.flush()

        # Re-init codebook
        if epoch < REINIT_TRAIN_EPOCH:
            reinit_count = reinit_codebook(traindataloader, encoder)            
            # print(f'{reinit_count} Codes Reinitialied')

        # Save model after n epoch
        if (epoch + 1) % 50 == 0 or epoch + 1 == TOTAL_TRAIN_EPOCH:
            torch.save(encoder.state_dict(), os.path.join(args.output, 'enc_epoch_' + str(epoch + 1) + '.pt'))
            torch.save(decoder.state_dict(), os.path.join(args.output, 'dec_epoch_' + str(epoch + 1) + '.pt'))

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Output folder to save the data", required=True)
    parser.add_argument("--batchsize", type=int, help="Batch size", required=True)
    parser.add_argument("--device", type=str, help="CUDA device", required=True)
    parser.add_argument("--format", type=str, help="Data type", required=True)
    args = parser.parse_args()

    # Create training folder
    result_folder = args.output
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Start training 
    train(args)