import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler as scheduler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# *transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers.models.mbart.modeling_mbart import shift_tokens_right

# *user-defined
from dataloader.datasets import S2T_Dataset
import utils as utils
from models.models import gloss_free_model as gloss_free_model
from models.SignCL import SignCL
cl_criterion = SignCL(max_distance=64.0)

# *basic
import os
import time
import argparse, json, datetime
import numpy as np
from collections import OrderedDict
import yaml
import random
import wandb
from pathlib import Path
from typing import Iterable, Optional
import math, sys
from loguru import logger
from sacrebleu.metrics import BLEU
from hpman.m import _
import hpargparse

# *timm
from timm.optim import create_optimizer, create_optimizer_v2
from timm.utils import NativeScaler

# global definition
from definition import *


def get_args_parser():
    parser = argparse.ArgumentParser('Gloss-free Sign Language Translation script', add_help=False)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=80, type=int)

    # * Pretrained checkpoint from VLP (stage 1)
    parser.add_argument('--finetune', default='', help='Path to checkpoint from pretraining stage, e.g., Checkpoints/Phoenix/GFSLT/best_checkpoint.pth')

    # * Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer')
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--filter_bias_and_bn', action='store_false',
                        help='Disable filtering out bias, bn and other 1d params from weight decay. The default is True. If you set, it becomes false.')

    # * Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1.0e-08, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience_epochs', type=int, default=5, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 5')
    parser.add_argument('--decay_rate', type=float, default=0.5, metavar='RATE',
                        help='LR decay rate ')
    parser.add_argument('--gamma', type=float, default=0.922, metavar='RATE',
                        help='LR decay rate for exponentialLR scheduler')

    # * onecyclelr schedule parameters
    parser.add_argument('--max_lr', type=float, help='Max lr for onecyclelr scheduler')
    parser.add_argument('--pct_start', type=float, default=0.05,
                        help='The percentage of the cycle spent increasing the learning rate.')
    parser.add_argument('--onecyclelr_epochs', type=int, default=200,
                        help='The number of epochs to train for. If the model train starts from epoch0, then this will be equal to epochs.')

    
     # * Base params
    parser.add_argument('--output_dir', default='Checkpoints/Phoenix/GFSLT/slt',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='Checkpoints/Phoenix/GFSLT/slt/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--config', type=str, default='configs/phoenix/config1.yaml')

    # *Drop out params
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    
    # * data process params
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--resize', default=256, type=int)
    # parser.add_argument('--frame_sampling', default="random", type=str,
    #                     help='Strategy to downsample frames when video exceeds max_length (e.g. 300). Options: '
    #                          'random: randomly pick max_length frames. sequential: uniformly skip frames with adaptive_frame_skip.'
    #                          'However, sequential is not used since the majority of videos in Phoenix and CSL-Daily are already shorter than 300 frames.')

    # other parameters
    parser.add_argument('--dont_resume_lr', action='store_true',
                        help='Default action is to resume lr. But if you dont want to resume lr, then use this.')
    parser.add_argument('--accumulation_step',  type=int, default=1, help="accumulation step for loss backward.")
    parser.add_argument('--dont_resume_scheduler', action='store_true',
                        help='Default action is to re-load scheduler. But if you dont want to do, then use this.')
    parser.add_argument('--dont_resume_optimizer', action='store_true',
                        help='Default action is to re-load optimizer state. But if you dont want to do, then use this.')
    parser.add_argument('--wandb_name', default="GF-SLT", help='Project name for wandb')
    parser.add_argument('--gradient_clipping', default=0, help='max_norm value for gradients')

    # Code Benchmark: Selection of Model Type
    parser.add_argument("--model_type", type=str, default='gfslt', help="options: gfslt, cico, signcl, flallm, c2rl")

    # signcl specific parameters
    parser.add_argument('--zipf_factor', type=float, default=2.3, help='Zipf factor for signCL loss')
    parser.add_argument('--signcl_warmup_epochs', type=int, default=0, metavar='N',
                        help='For the first signcl_warmup_epochs, only the main loss is optimized. After warmup, a weighted signcl loss is added.')
    parser.add_argument('--signcl_decay_rate', type=float, default=0.5, metavar='RATE',
                        help='LR decay rate for signcl')
    
    # flallm, c2rl
    parser.add_argument('--lr_llm_adapter', type=float, default=5.0e-5, metavar='LR',
                        help='learning rate for llm adapter')
    parser.add_argument('--frozenFeatureExtractor', action='store_true', help='freeze feature extractor')

    return parser

def custom_param_group_fn(model):
    """
    Different learning is assigned for VL-adapter in Fla-llm and C2RL approach.
    So, this function arranges VL-adapter's (sign_emb) learning rate with args.lr_llm_adapter
    """
    return [
        {'params': [p for name, p in model.named_parameters()
                    if 'sign_emb' not in name and not name.endswith("bias")],
         'lr': args.lr,
         'weight_decay': args.weight_decay
         },
        {'params': [p for name, p in model.named_parameters()
                       if 'sign_emb' not in name and name.endswith("bias") and p.ndim <= 1 ],
            'lr': args.lr,
            'weight_decay': 0  # No weight decay for bias
        },
        {'params': [p for name, p in model.named_parameters() if 'sign_emb' in name],
         'lr': args.lr_llm_adapter,
         'weight_decay': args.weight_decay}
    ]

def optimizer_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
        opt=cfg.opt,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum)
    if getattr(cfg, 'opt_eps', None) is not None:
        kwargs['eps'] = cfg.opt_eps
    if getattr(cfg, 'opt_betas', None) is not None:
        kwargs['betas'] = cfg.opt_betas
    if getattr(cfg, 'layer_decay', None) is not None:
        kwargs['layer_decay'] = cfg.layer_decay
    if getattr(cfg, 'opt_args', None) is not None:
        kwargs.update(cfg.opt_args)
    return kwargs


LANGUAGE = ""
BLEU_TOKENIZE = "none"
def main(args, config):
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    global LANGUAGE
    global BLEU_TOKENIZE
    if not "dataset_name" in config['data'].keys() or config['data']['dataset_name'] == "phoenix":
        LANGUAGE = "de_DE"
    else:
        if config['data']['dataset_name'] == "csl-daily":
            LANGUAGE = "zh_CN"
            BLEU_TOKENIZE = "zh"

    print(f"Creating dataset:")
    tokenizer = MBartTokenizer.from_pretrained(config['model']['tokenizer'])
    tokenizer.model_max_length = 300

    train_data = S2T_Dataset(path=config['data']['train_label_path'], tokenizer = tokenizer, config=config, args=args, phase='train')
    print(train_data)
    train_sampler = torch.utils.data.RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 collate_fn=train_data.collate_fn,
                                 sampler=train_sampler, 
                                 pin_memory=args.pin_mem)
    
    
    dev_data = S2T_Dataset(path=config['data']['dev_label_path'], tokenizer = tokenizer, config=config, args=args, phase='dev')
    print(dev_data)
    dev_sampler = torch.utils.data.RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=dev_data.collate_fn,
                                 sampler=dev_sampler, 
                                 pin_memory=args.pin_mem)
    
    test_data = S2T_Dataset(path=config['data']['test_label_path'], tokenizer = tokenizer, config=config, args=args, phase='test')
    print(test_data)
    test_sampler = torch.utils.data.SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=test_data.collate_fn,
                                 sampler=test_sampler,
                                 pin_memory=args.pin_mem)


    tokenizer = MBartTokenizer.from_pretrained(config['model']['tokenizer'], src_lang = LANGUAGE, tgt_lang = LANGUAGE)
    print(f"Creating model:")
    model = gloss_free_model(config, args)
    model.to(device)
    print(model)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.model_type in ["flallm", "c2rl"] and args.finetune and not args.eval:
        print('***********************************')
        print('Load parameters for Visual Encoder (Only backbone)...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            if 'backbone' in k:
                new_state_dict[k] = v

        ret = model.load_state_dict(new_state_dict, strict=False)

    elif args.finetune:
        print('***********************************')
        print('Load parameters for Visual Encoder...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu')
        new_state_dict = OrderedDict()

        # print("Top-level keys:", state_dict.keys())
        # print("\nModel keys:")
        # for k in sorted(state_dict['model'].keys()):
        #     print(k)

        for k, v in state_dict['model'].items():
            # RGB image clip
            if 'model_images' in k:
                if 'conv_2d' in k or 'conv_1d' in k or 'mlp' in k:
                    k = 'backbone.'+'.'.join(k.split('.')[2:])
                    new_state_dict[k] = v
            elif 'trans_encoder' in k:
                k = 'mbart.model.encoder.'+'.'.join(k.split('.')[2:])
                new_state_dict[k] = v

        # Text decoder
        if 'text_decoder' in state_dict:
            for k, v in state_dict['text_decoder'].items():
                if 'decoder' in k:
                    k = 'mbart.model.decoder.' + '.'.join(k.split('.')[1:])
                    new_state_dict[k] = v
                elif any(item in k for item in ['final_logits_bias', 'lm_head.weight']):
                    k = 'mbart.' + k
                    new_state_dict[k] = v
            
        # *replace the word embedding
        model_dict = torch.load(config['model']['transformer']+'/pytorch_model.bin', map_location='cpu')
        for k, v in model_dict.items():
            if any(item in k for item in ['decoder.embed_tokens.weight', 'decoder.embed_positions.weight']):
                k = 'mbart.' + k
                new_state_dict[k] = v


        ret = model.load_state_dict(new_state_dict, strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))


    n_parameters = utils.count_parameters_in_MB(model)
    print(f'number of params: {n_parameters}M')

    if args.model_type in ["c2rl", "flallm"]: # different lr for VL-adapter, so call custom_param_group_fn
        optimizer = create_optimizer_v2(model,
                                        **optimizer_kwargs(cfg=args),
                                        param_group_fn=custom_param_group_fn,
                                        filter_bias_and_bn=False)
    else:
        optimizer = create_optimizer(args, model, filter_bias_and_bn=args.filter_bias_and_bn)

    min_lr = config["training"].get("min_lr", 1e-8)

    if args.sched == "cosine":
        lr_scheduler = scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    eta_min=min_lr,
                    T_max=args.epochs,
                )
    elif args.sched == "plateau":
        lr_scheduler = scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            factor=args.decay_rate, patience=args.patience_epochs, threshold_mode='rel',
            threshold=0.1,
            verbose=True,
            min_lr = args.min_lr
        )
    elif args.sched == "exp":
        lr_scheduler = scheduler.ExponentialLR(
                optimizer=optimizer,
                gamma=args.gamma,
            )
    elif args.sched == "onecyclelr":
        # initial_lr = max_lr/div_factor  - Note default div_factor = 25
        # min_lr = initial_lr/final_div_factor
        lr_scheduler = scheduler.OneCycleLR(
                optimizer,
                max_lr = [args.max_lr,args.max_lr],
                steps_per_epoch=len(train_dataloader),
                epochs= args.onecyclelr_epochs,
                pct_start=args.pct_start,  # 0.05= 5% of total steps for warmup
                anneal_strategy='cos'
            )
    loss_scaler = NativeScaler()

    labelsmooth = config["training"].get("label_smoothing", 0.2)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=labelsmooth)

    output_dir = Path(args.output_dir)
    if args.resume:
        print('Resuming Model Parameters... ')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
            if not args.dont_resume_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if not args.dont_resume_scheduler: # if dont_resume_scheduler True, skip this block
                print("Loading previous lr_scheduler state..")
                lr_scheduler_state = checkpoint['lr_scheduler']
                lr_scheduler.load_state_dict(lr_scheduler_state)
            if args.dont_resume_lr: # if dont_resume_lr True, then assign new lr
                print(f"Do not resume the last lr.. Assign lr={args.lr}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr


    if args.eval:
        if not args.resume:
            logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
        ckpt_name = args.resume.split("/")[-1].split(".")[0]
        # dev_stats = evaluate(args, dev_dataloader, model, tokenizer, criterion, config, UNK_IDX, SPECIAL_SYMBOLS,
        #                       PAD_IDX, device, split=f"dev-{ckpt_name}")
        # print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {dev_stats['bleu4']:.2f}")
        test_stats = evaluate(args, test_dataloader, model, tokenizer, criterion, config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, split=f"test-{ckpt_name}")
        print(f"BLEU-4 of the network on the {len(test_dataloader)} test videos: {test_stats['bleu4']:.2f}")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    # for signcl loss
    last_acc = 0.0
    last_acc_epoch = 1
    cl_decay = 1.0
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(args, model, criterion, train_dataloader, optimizer, device, epoch, config, loss_scaler, tokenizer, lr_scheduler=lr_scheduler)
        if args.sched == "cosine":
            lr_scheduler.step(epoch)

        if args.output_dir and utils.is_main_process():
            checkpoint_paths = [output_dir / f'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)
        
        test_stats = evaluate(args, dev_dataloader, model, tokenizer, criterion, config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['bleu4']:.2f}")
        if args.sched == "plateau":
            lr_scheduler.step(test_stats['bleu4'])
        elif args.sched == "exp":
            lr_scheduler.step()

        # check min lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], args.min_lr)

        if max_accuracy < test_stats["bleu4"]:
            max_accuracy = test_stats["bleu4"]

            # for signcl
            if test_stats["bleu4"] - last_acc > 2.0:
                last_acc_epoch = epoch
                last_acc = test_stats["bleu4"]

            if args.output_dir and utils.is_main_process():
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
            
        print(f'Max BLEU-4: {max_accuracy:.2f}%')
        # signcl loss
        if args.model_type == "signcl" and (epoch - last_acc_epoch) > 10:
            cl_decay = cl_decay * args.signcl_decay_rate
            last_acc_epoch = epoch

        if utils.is_main_process():
            wandb.log({'epoch':epoch+1,'training/train_loss':train_stats['loss'], 'dev/dev_loss':test_stats['loss'],
                           'dev/Bleu_4':test_stats['bleu4'], 'dev/Best_Bleu_4': max_accuracy,
                           'lr':train_stats['lr']})

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    # Last epoch
    test_on_last_epoch = True
    if test_on_last_epoch and args.output_dir:
        checkpoint = torch.load(args.output_dir+'/best_checkpoint.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        
        test_stats = evaluate(args, test_dataloader, model, tokenizer, criterion, config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BLEU-4 of the network on the {len(test_dataloader)} test videos: {test_stats['bleu4']:.2f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(args, model: torch.nn.Module, criterion: nn.CrossEntropyLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, config, loss_scaler, tokenizer, max_norm: float = 0,
                    set_training_mode=True, lr_scheduler=None, cl_decay=1.0):
    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 100

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        total_loss = 0.0
        if args.model_type == "c2rl":
            out_logits, cico_logits, cico_logits_gt = model(src_input, tgt_input)
            cico_loss = cico_logits["loss_itc"]
        else:
            out_logits, frames_feature = model(src_input, tgt_input)

        label = tgt_input['input_ids'].reshape(-1)
        logits = out_logits.reshape(-1,out_logits.shape[-1])
        loss = criterion(logits, label.to(device, non_blocking=True))

        if args.model_type == "c2rl":
            total_loss = loss + cico_loss
        elif args.model_type == "signcl":
            margin = max(10, int((frames_feature.shape[1] // tgt_input['input_ids'].shape[1] + 1) * args.zipf_factor)) * 2
            num_negative = 30
            margin = min(margin, int((frames_feature.shape[1] - num_negative) / 2))  # ensure num_frames margin for negative sampling
            cl_loss = cl_criterion(frames_feature, margin=margin)
            if epoch < args.signcl_warmup_epochs:
                total_loss += loss
            else:
                total_loss += loss + 0.01 * cl_loss * cl_decay

            metric_logger.update(cl_loss=cl_loss.item())
            metric_logger.update(cl_decay=cl_decay)
        else:
            total_loss += loss


        # normalize the gradient
        total_loss = total_loss / args.accumulation_step
        # torch.autograd.set_detect_anomaly(True)
        total_loss.backward()

        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)  # L2 norm
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        # print(f"Total gradient norm: {total_norm}")
        metric_logger.update(total_norm=total_norm)
        if args.gradient_clipping != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clipping)

        if (step + 1) % args.accumulation_step == 0:
            optimizer.step()
            if args.sched == "onecyclelr":
                lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        
        loss_value = loss.detach().item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # torch.cuda.empty_cache()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, dev_dataloader, model, tokenizer, criterion,  config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, split="dev"):
    model.eval()
    global LANGUAGE
    global BLEU_TOKENIZE

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []

        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(dev_dataloader, 10, header)):
            total_loss = 0.0
            total_cl_loss = 0.0
            if args.model_type == "c2rl":
                out_logits, cico_logits, cico_logits_gt = model(src_input, tgt_input)
                total_loss = cico_logits["loss_itc"]
            else:
                out_logits, frames_feature = model(src_input, tgt_input)

            label = tgt_input['input_ids'].reshape(-1)
            logits = out_logits.reshape(-1,out_logits.shape[-1])
            tgt_loss = criterion(logits, label.to(device))

            total_loss += tgt_loss

            if args.model_type == "signcl":
                margin = max(10, int((frames_feature.shape[1] // tgt_input['input_ids'].shape[1] + 1) * args.zipf_factor)) * 2
                num_negative = 30
                margin = min(margin, int((frames_feature.shape[1] - num_negative) / 2))  # ensure num_frames margin for negative sampling
                cl_loss = cl_criterion(frames_feature, margin=margin)
                total_cl_loss += cl_loss
                metric_logger.update(cl_loss=total_cl_loss.item())

            metric_logger.update(loss=total_loss.item())

            output = model.generate(src_input, max_new_tokens=150, num_beams = 4,
                        decoder_start_token_id=tokenizer.lang_code_to_id[LANGUAGE],
                                                num_return_sequences=1
                        )

            tgt_input['input_ids'] = tgt_input['input_ids'].to(device)
            for i in range(len(output)):
                tgt_pres.append(output[i,:])
                tgt_refs.append(tgt_input['input_ids'][i,:])

    pad_tensor = torch.ones(200-len(tgt_pres[0])).to(device)
    tgt_pres[0] = torch.cat((tgt_pres[0],pad_tensor.long()),dim = 0)
    tgt_pres = pad_sequence(tgt_pres,batch_first=True,padding_value=PAD_IDX)

    pad_tensor = torch.ones(200-len(tgt_refs[0])).to(device)
    tgt_refs[0] = torch.cat((tgt_refs[0],pad_tensor.long()),dim = 0)
    tgt_refs = pad_sequence(tgt_refs,batch_first=True,padding_value=PAD_IDX)

    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)
    tgt_refs = tokenizer.batch_decode(tgt_refs, skip_special_tokens=True)

    dataset_name = config["data"].get("dataset_name", "phoenix")
    # If it is done by dataloader, no need in here!!
    # if dataset_name == "csl-daily":
    #     tgt_pres = [' '.join(list(r)) for r in tgt_pres]
    #     tgt_refs = [' '.join(list(r)) for r in tgt_refs]
    if dataset_name == "phoenix" and args.eval:
        tgt_pres = [sent + " ." for sent in tgt_pres]
        tgt_refs = [sent + " ." for sent in tgt_refs]

    print("Printing some targets and predictions:")
    for print_i in range(3):
        print(f"Target refs[{print_i}]:", tgt_refs[print_i])
        print(f"Target pres[{print_i}]:", tgt_pres[print_i])

    # sacrablue
    bleu = BLEU()
    bleu_s = bleu.corpus_score(tgt_pres, [tgt_refs]).score
    metric_logger.meters['bleu4'].update(bleu_s)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* BLEU-4 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.bleu4, losses=metric_logger.loss))
    
    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        # also calculate ignite sacrebleu
        from metrics.bleu_score import BLEUScore
        ignite_sacrebleu = BLEUScore(pre_name="sacre")
        ignite_sacrebleu.reset()
        ignite_sacrebleu.update((tgt_pres, tgt_refs))
        ignite_sacrebleu_scores = ignite_sacrebleu.compute()

        # original sacrableu
        original_sacrableu = {}
        for i in range(1, 5):
            bleu = BLEU(max_ngram_order=i)
            original_sacrableu[f"sacre_bleu{i}"] = bleu.corpus_score(tgt_pres, [tgt_refs]).score
            signature = bleu.get_signature()
            print(signature)

        # write sentences into files
        Path(args.output_dir+"/"+split).mkdir(parents=True, exist_ok=True)
        with open(args.output_dir+ f'/{split}/tmp_pres.txt','w') as f:
            f.writelines(line + "\n" for line in tgt_pres)
        with open(args.output_dir+f'/{split}/tmp_refs.txt','w') as f:
            f.writelines(line + "\n" for line in tgt_refs)
        print('\n'+'*'*80)
        # ---- nlgeval
        # try:
        #     from nlgeval import compute_metrics
        # except:
        #     print('Please install nlgeval: pip install git+https://github.com/Maluuba/nlg-eval.git@master')
        
        # metrics_dict = compute_metrics(hypothesis=args.output_dir + f'/{split}/tmp_pres.txt',
        #                    references=[args.output_dir + f'/{split}/tmp_refs.txt'],no_skipthoughts=True,no_glove=True)
        # with open(args.output_dir+f'/{split}/scores.txt','w') as f:
        #     for key, value in metrics_dict.items():
        #         f.write(f"{key}: {value}\n")

        # save scores to txt files
        with open(args.output_dir + f'/{split}/ignite_sacrebleu_scores.txt', 'w') as f:
            for key, value in ignite_sacrebleu_scores.items():
                f.write(f"{key}: {value}\n")
        with open(args.output_dir + f'/{split}/sacrebleu_library_scores.txt', 'w') as f:
            for key, value in original_sacrableu.items():
                f.write(f"{key}: {value}\n")
            f.write(f"sacrableu signature: {signature}\n")
        print('*'*80)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Gloss-free Sign Language Translation script', parents=[get_args_parser()])
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()

    with open(args.config, 'r+',encoding='utf-8') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    
    os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
    if utils.is_main_process():
        # TODO: Please set the wand API key as an environment.
        wandb_user_key = os.environ.get("WANDB_API_KEY", None)
        if wandb_user_key is None:
            print("Warning: WANDB_API_KEY not set, logging disabled")
            os.environ["WANDB_MODE"] = "disabled"

        wandb.init(project=args.wandb_name,config=config)
        wandb.run.name = args.output_dir.split('/')[-1]
        wandb.define_metric("epoch")
        wandb.define_metric("training/*", step_metric="epoch")
        wandb.define_metric("dev/*", step_metric="epoch")
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)

