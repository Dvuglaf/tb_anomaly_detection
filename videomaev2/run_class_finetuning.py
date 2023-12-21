import argparse
import datetime
import json
import os
import random
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path

import pandas as pd
from torch import nn
from dataloader import PreflightWindowDataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.utils import ModelEma

import models
import utils
from engine_for_finetuning import (
    final_test,
    merge,
    train_one_epoch,
    validation_one_epoch,
)
from optim_factory import (
    LayerDecayValueAssigner,
    create_optimizer,
    get_parameter_groups,
)
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_samples_collate

import wandb

wandb.login(key="94cdfe481e8b7cd3255d0a038b900cda4ec77b3f", relogin=True)

wandb_init = wandb.init(name="test",
                        mode="online")


def get_args():
    parser = argparse.ArgumentParser(
        'VideoMAE fine-tuning and evaluation script for action classification',
        add_help=False)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model parameters
    parser.add_argument(
        '--model',
        default='vit_base_patch16_224',
        type=str,
        metavar='MODEL',
        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument(
        '--input_size', default=224, type=int, help='images input size')

    parser.add_argument(
        '--with_checkpoint', action='store_true', default=True)

    parser.add_argument('--with_class_weights', action='store_true', default=False)
    parser.add_argument('--with_freezing', action='store_true', default=False)

    parser.add_argument(
        '--drop',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Dropout rate (default: 0.)')
    parser.add_argument(
        '--attn_drop_rate',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Attention dropout rate (default: 0.)')
    parser.add_argument(
        '--drop_path',
        type=float,
        default=0.1,
        metavar='PCT',
        help='Drop path rate (default: 0.1)')
    parser.add_argument(
        '--head_drop_rate',
        type=float,
        default=0.0,
        metavar='PCT',
        help='cls head dropout rate (default: 0.)')

    parser.add_argument(
        '--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument(
        '--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument(
        '--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument(
        '--opt',
        default='adamw',
        type=str,
        metavar='OPTIMIZER',
        help='Optimizer (default: "adamw"')
    parser.add_argument(
        '--opt_eps',
        default=1e-8,
        type=float,
        metavar='EPSILON',
        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument(
        '--opt_betas',
        default=None,
        type=float,
        nargs='+',
        metavar='BETA',
        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument(
        '--clip_grad',
        type=float,
        default=None,
        metavar='NORM',
        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        metavar='M',
        help='SGD momentum (default: 0.9)')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.05,
        help='weight decay (default: 0.05)')
    parser.add_argument(
        '--weight_decay_end',
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        metavar='LR',
        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument(
        '--warmup_lr',
        type=float,
        default=1e-6,
        metavar='LR',
        help='warmup learning rate (default: 1e-6)')
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-5,
        metavar='LR',
        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=5,
        metavar='N',
        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=-1,
        metavar='N',
        help='num of steps to warmup LR, will overload warmup_epochs if set > 0'
    )

    # * Finetuning params
    parser.add_argument(
        '--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument(
        '--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument(
        '--nb_classes',
        default=400,
        type=int,
        help='number of the classification types')
    parser.add_argument(
        '--output_dir',
        default='tmp/',
        help='path where to save, empty for no saving')
    parser.add_argument(
        '--log_dir', default='log/', help='path where to tensorboard log')
    parser.add_argument(
        '--device',
        default='cpu',
        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument(
        '--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument(
        '--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument(
        '--eval', action='store_true', default=False, help='Perform evaluation only')

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help=
        'Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        '--world_size',
        default=4,
        type=int,
        help='number of distributed processes')

    known_args, _ = parser.parse_known_args()
    ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    print(args)
    print(args, file=open('output.txt', 'a'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    dataset_train = PreflightWindowDataset(f"train_clips_original.csv", mode='train')
    dataset_val = PreflightWindowDataset(f"test_clips_original.csv", mode='val')
    dataset_test = PreflightWindowDataset(f"inf_clips_original.csv", mode='test')

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    print(f"Num tasks: {num_tasks}")
    print(f"Global rank: {global_rank}")

    sampler_train = torch.utils.data.SequentialSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    collate_func = None
    mixup_fn = None

    model = create_model(
        args.model,
        img_size=args.input_size,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=16,
        tubelet_size=args.tubelet_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        head_drop_rate=args.head_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        with_cp=args.with_checkpoint,
    )

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))

    args.patch_size = patch_size

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        print("Load ckpt from %s" % args.finetune, file=open('output.txt', 'a'))

        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                print("Load state_dict by model_key = %s" % model_key, file=open('output.txt', 'a'))

                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        for old_key in list(checkpoint_model.keys()):
            if old_key.startswith('_orig_mod.'):
                new_key = old_key[10:]
                checkpoint_model[new_key] = checkpoint_model.pop(old_key)
        args.data_set = "MIT"  ##############################################
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[
                k].shape != state_dict[k].shape:
                if checkpoint_model[k].shape[
                    0] == 710 and args.data_set.startswith('Kinetics'):
                    print(f'Convert K710 head to {args.data_set} head')
                    if args.data_set == 'Kinetics-400':
                        label_map_path = 'misc/label_710to400.json'
                    elif args.data_set == 'Kinetics-600':
                        label_map_path = 'misc/label_710to600.json'
                    elif args.data_set == 'Kinetics-700':
                        label_map_path = 'misc/label_710to700.json'

                    label_map = json.load(open(label_map_path))
                    checkpoint_model[k] = checkpoint_model[k][label_map]
                else:
                    print(f"Removing key {k} from pretrained checkpoint")
                    print(f"Removing key {k} from pretrained checkpoint", file=open('output.txt', 'a'))
                    del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
            num_patches = model.patch_embed.num_patches  #
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1

            # height (== width) for the checkpoint position embedding
            orig_size = int(
                ((pos_embed_checkpoint.shape[-2] - num_extra_tokens) //
                 (args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(
                (num_patches //
                 (args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" %
                      (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(
                    -1, args.num_frames // model.patch_embed.tubelet_size,
                    orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                                embedding_size).permute(
                    0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=(new_size, new_size),
                    mode='bicubic',
                    align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                    -1, args.num_frames // model.patch_embed.tubelet_size,
                    new_size, new_size, embedding_size)
                pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        utils.load_state_dict(
            model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model

    print("params:")
    print("params:", file=open('output.txt', 'a'))
    if args.with_freezing:
        for i, p in enumerate(model.parameters()):
            p.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters_total = sum(p.numel() for p in model.parameters())
    print('\tnumber of train params:', n_parameters)
    print('\tnumber of total params:', n_parameters_total)

    print('\tnumber of train params:', n_parameters, file=open('output.txt', 'a'))
    print('\tnumber of total params:', n_parameters_total, file=open('output.txt', 'a'))

    print("Model = %s" % str(model_without_ddp))
    print("Model = %s" % str(model_without_ddp), file=open('output.txt', 'a'))

    total_batch_size = args.batch_size * args.update_freq * num_tasks
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    #########scale the lr#############
    # args.lr = args.lr * total_batch_size / 256
    # args.min_lr = args.min_lr * total_batch_size / 256
    # args.warmup_lr = args.warmup_lr * total_batch_size / 256
    #########scale the lr#############

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" %
          num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i)
                 for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    optimizer = create_optimizer(
        args,
        model_without_ddp,
        skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(args.weight_decay,
                                                args.weight_decay_end,
                                                args.epochs,
                                                num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" %
          (max(wd_schedule_values), min(wd_schedule_values)))

    if args.with_class_weights:
        class_weights = torch.tensor([0.755331088664422, 0.7371303395399781, 0.7695826186392224,
                                      0.9412587412587412, 0.377665544332211, 1.9422799422799424,
                                      0.8103552077062011, 2.49721706864564, 3.1375291375291376,
                                      3.059090909090909, 1.9422799422799424, ]).type(torch.float32)
        print(f"class weights: {class_weights}")
        print(f"class weights: {class_weights}", file=open('output.txt', 'a'))

        class_weights = class_weights.to(device, non_blocking=True)
        criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        model_ema=model_ema)

    if args.eval:
        preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
        test_stats = final_test(data_loader_test, model, criterion, device, preds_file)
        if global_rank == 0:
            print("Start merging results...")
            if args.output_dir and utils.is_main_process():
                with open(
                        os.path.join(args.output_dir, "log.txt"),
                        mode="a",
                        encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
        exit(0)

    wandb_init.watch(model)

    print(f"Start training for {args.epochs} epochs")
    print(f"Start training for {args.epochs} epochs", file=open('output.txt', 'a'))
    start_time = time.time()
    max_f1_val = 0.0
    max_acc_val = 0.0
    max_f1_test = 0.0
    max_acc_test = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch *
                                args.update_freq)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
        )
        if args.output_dir and args.save_ckpt:
            _epoch = epoch + 1
            if _epoch % args.save_ckpt_freq == 0 or _epoch == args.epochs:
                utils.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    model_ema=model_ema)
        if data_loader_val is not None:
            test_stats = validation_one_epoch(data_loader_val, model, criterion, device, wandb_init, "val")
            print(
                f"Accuracy of the network on the {len(dataset_val)} val images: {test_stats['acc1']:.2f}%"
            )
            print(
                f"F1-Score of the network on the {len(dataset_val)} val images: {test_stats['f1']:.2f}%"
            )

            print(
                f"Accuracy of the network on the {len(dataset_val)} val images: {test_stats['acc1']:.2f}%",
                file=open('output.txt', 'a')
            )
            print(
                f"F1-Score of the network on the {len(dataset_val)} val images: {test_stats['f1']:.2f}%",
                file=open('output.txt', 'a')
            )
            if max_f1_val < test_stats["f1"]:
                max_f1_val = test_stats["f1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best_f1_val",
                        model_ema=model_ema)
                    print("Saved checkpoint-best_f1_val.pth")
                    print("Saved checkpoint-best_f1_val.pth", file=open('output.txt', 'a'))

            if max_acc_val < test_stats["acc1"]:
                max_acc_val = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best_acc_val",
                        model_ema=model_ema)
                    print("Saved checkpoint-best_acc_val.pth")
                    print("Saved checkpoint-best_acc_val.pth", file=open('output.txt', 'a'))

            print(f'Max F1: {max_f1_val:.2f}%')
            print(f'Max Acc: {max_acc_val:.2f}%')

            print(f'Max F1: {max_f1_val:.2f}%', file=open('output.txt', 'a'))
            print(f'Max Acc: {max_acc_val:.2f}%', file=open('output.txt', 'a'))
            if log_writer is not None:
                log_writer.update(
                    val_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(
                    val_loss=test_stats['loss'], head="perf", step=epoch)

            if data_loader_test is not None:
                tb_video_stats = validation_one_epoch(data_loader_test, model, criterion, device, wandb_init, "inf")
                print(
                    f"Accuracy of the network on the {len(dataset_test)} test images: {tb_video_stats['acc1']:.2f}%"
                )
                print(
                    f"F1-Score of the network on the {len(dataset_test)} test images: {tb_video_stats['f1']:.2f}%"
                )

                print(
                    f"Accuracy of the network on the {len(dataset_test)} test images: {tb_video_stats['acc1']:.2f}%",
                    file=open('output.txt', 'a')
                )
                print(
                    f"F1-Score of the network on the {len(dataset_test)} test images: {tb_video_stats['f1']:.2f}%",
                    file=open('output.txt', 'a')
                )
                if max_f1_test < tb_video_stats["f1"]:
                    max_f1_test = tb_video_stats["f1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args,
                            model=model,
                            model_without_ddp=model_without_ddp,
                            optimizer=optimizer,
                            loss_scaler=loss_scaler,
                            epoch="best_f1_test",
                            model_ema=model_ema)
                        print("Saved checkpoint-best_f1_test.pth")
                        print("Saved checkpoint-best_f1_test.pth", file=open('output.txt', 'a'))

                if max_acc_test < tb_video_stats["acc1"]:
                    max_acc_test = tb_video_stats["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args,
                            model=model,
                            model_without_ddp=model_without_ddp,
                            optimizer=optimizer,
                            loss_scaler=loss_scaler,
                            epoch="best_acc_test",
                            model_ema=model_ema)
                        print("Saved checkpoint-best_acc_test.pth")
                        print("Saved checkpoint-best_acc_test.pth", file=open('output.txt', 'a'))

                print(f'Max F1 test: {max_f1_test:.2f}%')
                print(f'Max Acc test: {max_acc_test:.2f}%')

                print(f'Max F1 test: {max_f1_test:.2f}%', file=open('output.txt', 'a'))
                print(f'Max Acc test: {max_acc_test:.2f}%', file=open('output.txt', 'a'))
                if log_writer is not None:
                    log_writer.update(
                        test_acc1=test_stats['acc1'], head="perf", step=epoch)
                    log_writer.update(
                        test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {
                **{f'train_{k}': v
                   for k, v in train_stats.items()},
                **{f'val_{k}': v
                   for k, v in test_stats.items()},
                **{f'test_{k}': v
                   for k, v in tb_video_stats.items()}, 'epoch': epoch,
                'n_parameters': n_parameters
            }

            wandb_init.log(
                {
                    "Loss/train": train_stats['loss'],
                    "Loss/val": test_stats['loss'],
                    "Accuracy/val": test_stats['acc1'],
                    "F1/val": test_stats['f1'],
                    "Accuracy/inf": tb_video_stats['acc1'],
                    "F1/inf": tb_video_stats['f1'],
                    "Learning rate": train_stats['lr']
                }
            )
        else:
            log_stats = {
                **{f'train_{k}': v
                   for k, v in train_stats.items()}, 'epoch': epoch,
                'n_parameters': n_parameters
            }
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                    os.path.join(args.output_dir, "log.txt"),
                    mode="a",
                    encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Training time {}'.format(total_time_str), file=open('output.txt', 'a'))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
