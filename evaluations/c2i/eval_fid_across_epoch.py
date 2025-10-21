import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
import argparse
from pathlib import Path
import tensorflow as tf

import sys
import itertools

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

from dataset.augmentation import center_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.vq_model import VQ_models

from evaluator_modify import Evaluator
import tensorflow.compat.v1 as tf

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

import wandb
import warnings
warnings.filterwarnings('ignore')

def dist_all_gather(tensor):
    """
    Gather tensors from all GPUs
    """
    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def extract_iteration_number(ckpt_path):
    """
    extract iteration number from checkpoint file name
    """
    filename = os.path.basename(ckpt_path)
    # remove extension, extract number
    name_without_ext = os.path.splitext(filename)[0]
    try:
        return int(name_without_ext)
    except ValueError:
        return None


def get_checkpoint_files(weights_dir, interval=2000):
    """
    file format: 0034000.pt (pure number.pt, guaranteed to be a multiple of 500)
    """
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        raise ValueError(f"Weights directory does not exist: {weights_dir}")
    
    # get all .pt files
    ckpt_files = list(weights_path.glob('*.pt'))
    
    # extract iteration and filter
    ckpt_dict = {}
    for ckpt_file in ckpt_files:
        iteration = extract_iteration_number(str(ckpt_file))
        if iteration is not None and iteration % interval == 0:
            ckpt_dict[iteration] = str(ckpt_file)
    
    # sort by iteration
    sorted_ckpts = sorted(ckpt_dict.items())
    return sorted_ckpts


def evaluate_checkpoint(args, vq_model, ckpt_path, ckpt_iter, device, rank, loader):
    """
    evaluate single checkpoint: load weights, generate reconstructed image, compute FID, PSNR and SSIM
    """
    # load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "ema" in checkpoint:
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    
    msg = vq_model.load_state_dict(model_weight, strict=False)
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Evaluating checkpoint at iteration {ckpt_iter}")
        print(f"Checkpoint path: {ckpt_path}")
        print(f"{'='*80}")
        print(msg)
    del checkpoint
    
    vq_model.eval()
    
    # collect samples, GT and PSNR/SSIM metrics
    total = 0
    samples = []
    gt = []
    psnr_val_rgb = []
    ssim_val_rgb = []
    
    progress_loader = tqdm(loader, desc=f'Evaluation for iteration {ckpt_iter:07d}', disable=not rank == 0)
    
    for x, _ in progress_loader:
        # save GT for PSNR/SSIM calculation (convert to [0,1] range)
        rgb_gts = (x.permute(0, 2, 3, 1).to("cpu").numpy() + 1.0) / 2.0
        
        with torch.no_grad():
            x = x.to(device, non_blocking=True)
            
            # generate reconstructed image
            if args.vq_model == 'AE':
                sample, _ = vq_model(x)
            else:
                sample = vq_model.img_to_reconstructed_img(x)
            
            # convert to uint8 format for FID calculation
            sample_uint8 = torch.clamp(127.5 * sample + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
            x_uint8 = torch.clamp(127.5 * x + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
        
        # collect all GPU results for FID
        sample_gathered = torch.cat(dist_all_gather(sample_uint8), dim=0)
        x_gathered = torch.cat(dist_all_gather(x_uint8), dim=0)
        
        samples.append(sample_gathered.to("cpu", dtype=torch.uint8).numpy())
        gt.append(x_gathered.to("cpu", dtype=torch.uint8).numpy())
        
        # compute PSNR and SSIM (on local samples, avoid duplicate)
        sample_np = sample_uint8.to("cpu", dtype=torch.uint8).numpy()
        for i, (sample_img, rgb_gt) in enumerate(zip(sample_np, rgb_gts)):
            # convert sample to [0,1] range
            rgb_restored = sample_img.astype(np.float32) / 255.
            psnr = psnr_loss(rgb_restored, rgb_gt)
            ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True, data_range=1.0, channel_axis=-1)
            psnr_val_rgb.append(psnr)
            ssim_val_rgb.append(ssim)
        
        total += sample_gathered.shape[0]
    
    if rank == 0:
        print(f"Evaluated total {total} files.")
    
    # collect PSNR and SSIM results
    dist.barrier()
    world_size = dist.get_world_size()
    gather_psnr_val = [None for _ in range(world_size)]
    gather_ssim_val = [None for _ in range(world_size)]
    dist.all_gather_object(gather_psnr_val, psnr_val_rgb)
    dist.all_gather_object(gather_ssim_val, ssim_val_rgb)
    
    # compute average PSNR and SSIM
    psnr_avg = None
    ssim_avg = None
    if rank == 0:
        gather_psnr_val = list(itertools.chain(*gather_psnr_val))
        gather_ssim_val = list(itertools.chain(*gather_ssim_val))
        psnr_avg = sum(gather_psnr_val) / len(gather_psnr_val)
        ssim_avg = sum(gather_ssim_val) / len(gather_ssim_val)
        print(f"PSNR: {psnr_avg:.4f}, SSIM: {ssim_avg:.4f}")
    
    dist.barrier()
    
    # compute FID (only on rank 0)
    FID = None
    if rank == 0:
        samples = np.concatenate(samples, axis=0)
        gt = np.concatenate(gt, axis=0)
        
        # configure TensorFlow
        config = tf.ConfigProto(
            allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        config.gpu_options.allow_growth = True
        
        # initialize evaluator
        evaluator = Evaluator(tf.Session(config=config), batch_size=32)
        evaluator.warmup()
        
        print("Computing reference batch activations...")
        ref_acts = evaluator.read_activations(gt)
        print("Computing reference batch statistics...")
        ref_stats, _ = evaluator.read_statistics(gt, ref_acts)
        
        print("Computing sample batch activations...")
        sample_acts = evaluator.read_activations(samples)
        print("Computing sample batch statistics...")
        sample_stats, _ = evaluator.read_statistics(samples, sample_acts)
        
        # compute FID
        FID = sample_stats.frechet_distance(ref_stats)
        
        print(f"\n{'='*80}")
        print(f"Iteration: {ckpt_iter:07d}")
        print(f"FID: {FID:.6f}, PSNR: {psnr_avg:.4f}, SSIM: {ssim_avg:.4f}")
        print(f"{'='*80}\n")
        
        # save results to file
        result_file = f"{args.result_dir}/evaluation_results.txt"
        os.makedirs(args.result_dir, exist_ok=True)
        with open(result_file, 'a') as f:
            f.write(f"Iteration: {ckpt_iter:07d}, FID: {FID:.6f}, PSNR: {psnr_avg:.4f}, SSIM: {ssim_avg:.4f}\n")
    
    dist.barrier()
    return FID, psnr_avg, ssim_avg

def main(args):
    # Setup PyTorch
    assert torch.cuda.is_available(), "Evaluation with DDP requires at least one GPU."
    torch.set_grad_enabled(False)
    
    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    # create model
    wandb_tracker = wandb.init(project='LlamaGen-VQ')

    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        codebook_l2_norm=args.codebook_l2_norm # Original llamagen is True
        )
    vq_model.to(device)
    vq_model.eval()
    
    # get all checkpoints to process
    if rank == 0:
        print(f"Scanning weights directory: {args.weights_dir}")
    checkpoint_list = get_checkpoint_files(args.weights_dir, args.ckpt_interval)
    
    if rank == 0:
        print(f"Found {len(checkpoint_list)} checkpoints to process:")
        for iter_num, ckpt_path in checkpoint_list:
            print(f"  - Iteration {iter_num}: {ckpt_path}")
    
    if len(checkpoint_list) == 0:
        if rank == 0:
            print("No checkpoints found matching the interval criteria!")
        dist.destroy_process_group()
        return
    
    # Setup data
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    if args.dataset == 'imagenet':
        dataset = build_dataset(args, transform=transform)
    elif args.dataset == 'coco':
        dataset = build_dataset(args, transform=transform)
    elif args.dataset == 'imagenet100':
        dataset = build_dataset(args, transform=transform)
    else:
        raise Exception("please check dataset")
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # evaluate each checkpoint
    fid_results = {}
    for ckpt_iter, ckpt_path in checkpoint_list:
        if ckpt_iter < 100000:
            continue
        
        fid, psnr, ssim = evaluate_checkpoint(
            args, vq_model, ckpt_path, ckpt_iter, 
            device, rank, loader
        )

        if rank == 0:
            fid_results[ckpt_iter] = fid
            wandb_tracker.log({
                "FID": fid,
                "PSNR": psnr,
                "SSIM": ssim
            }, step=ckpt_iter)
    
    # final summary
    if rank == 0:
        print("\n" + "="*80)
        print("FID Evaluation Summary")
        print("="*80)
        for iter_num, fid in sorted(fid_results.items()):
            print(f"Iteration {iter_num:07d}: FID = {fid:.6f}")
        print("="*80)
        
        # find best FID
        if fid_results:
            best_iter = min(fid_results.items(), key=lambda x: x[1])
            print(f"\nBest FID: {best_iter[1]:.6f} at iteration {best_iter[0]:07d}")
            print("="*80)
    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=['imagenet', 'coco', 'imagenet100'], default='imagenet')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--weights-dir", type=str, required=True, help="directory containing checkpoint files")
    parser.add_argument("--ckpt-interval", type=int, default=1000, help="process checkpoints every N iterations")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--codebook-l2-norm", action="store_true", default=False)
    args = parser.parse_args()

    results_dir = os.path.dirname(args.weights_dir)
    args.result_dir = os.path.join(results_dir, "eval_results")


    main(args)