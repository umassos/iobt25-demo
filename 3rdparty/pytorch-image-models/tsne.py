#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial
from torchvision.datasets import Food101
import time

import numpy as np
import mlflow
import torch
import torch.nn as nn
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from efficientnet_pytorch import EfficientNet, get_model_params
from ensemble_efficient_net_b0 import EnsembleEfficientNet
from ensemble_efficient_net_b1 import EnsembleEfficientNetB1
from ensemble_resnet_50 import EnsembleResnet50
from ensemble_vit import EnsembleViT
from ensemble_efficientnet import EfficientNetEarly
from torchinfo import summary

from timm.layers import SelectAdaptivePool2d, Linear
from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
# from torch.utils.data import DataLoader
from tqdm import tqdm
# matplotlib.rcParams['text.usetex'] = False

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion

    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')

_logger = logging.getLogger('train')

# print('Hello')

# def contrastive_loss(x1, x2, eps=4, lamb=1e-2):
#     m = x1.size(0)
#     diff = lamb / m * torch.sum((x1 - x2)**2)

#     return torch.clip(eps-diff, min=0)


# class EnsembleLoss(object):
#     def __init__(self, loss_fn, weights):
#         self.loss_fn = loss_fn
#         self.weights = weights

#     def to(self, device):
#         self.loss_fn.to(device)
#         return self

#     def __call__(self, output, target):
#         y1_loss = self.loss_fn(output[0], target)
#         y2_loss = self.loss_fn(output[1], target)
#         y_comb_loss = self.loss_fn(output[2], target)
#         # return y_comb_loss

#         return self.weights[0] * y1_loss + self.weights[1] * y2_loss + self.weights[2] * y_comb_loss


# class EnsembleContrastiveLoss(object):
#     def __init__(self, ensemble_loss_fn, eps=4, lamb=1e-2, use_contrastive=False):
#         self.ensemble_loss_fn = ensemble_loss_fn
#         self.eps = eps
#         self.lamb = lamb
#         self.use_contrastive = use_contrastive

#     def to(self, device):
#         self.ensemble_loss_fn.to(device)
#         return self

#     def __call__(self, output, target):
#         ensemble_loss = self.ensemble_loss_fn(output, target)

#         if self.use_contrastive:
#             ensemble_loss += contrastive_loss(
#                 output[0], output[1], self.eps, self.lamb)

#         return ensemble_loss


# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside the dataset group because it is positional.
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
group.add_argument('--train-split', metavar='NAME', default='train',
                   help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation',
                   help='dataset validation split (default: validation)')
group.add_argument('--dataset-download', action='store_true', default=False,
                   help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                   help='path to class to idx mapping file (default: "")')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                   help='Name of model to train (default: "resnet50")')
group.add_argument('--pretrained', action='store_true', default=False,
                   help='Start with pretrained version of specified network (if avail)')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                   help='Initialize model from this checkpoint (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                   help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', action='store_true', default=False,
                   help='prevent resume of optimizer state when resuming model')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                   help='number of label classes (Model default if None)')
group.add_argument('--gp', default=None, type=str, metavar='POOL',
                   help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--img-size', type=int, default=None, metavar='N',
                   help='Image size (default: None => model default)')
group.add_argument('--in-chans', type=int, default=None, metavar='N',
                   help='Image input channels (default: None => 3)')
group.add_argument('--input-size', default=None, nargs=3, type=int,
                   metavar='N N N',
                   help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=None, type=float,
                   metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                   help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                   help='Override std deviation of dataset')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                   help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                   help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                   help='Validation batch size override (default: None)')
group.add_argument('--channels-last', action='store_true', default=False,
                   help='Use channels_last memory layout')
group.add_argument('--fuser', default='', type=str,
                   help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
group.add_argument('--grad-accum-steps', type=int, default=1, metavar='N',
                   help='The number of steps to accumulate gradients (default: 1)')
group.add_argument('--grad-checkpointing', action='store_true', default=False,
                   help='Enable gradient checkpointing through model blocks/stages')
group.add_argument('--fast-norm', default=False, action='store_true',
                   help='enable experimental fast-norm')
group.add_argument('--model-kwargs', nargs='*',
                   default={}, action=utils.ParseKwargs)
group.add_argument('--head-init-scale', default=None, type=float,
                   help='Head initialization scale')
group.add_argument('--head-init-bias', default=None, type=float,
                   help='Head initialization bias value')

# scripting / codegen
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                   help='Optimizer (default: "sgd")')
group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                   help='Optimizer Epsilon (default: None, use opt default)')
group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                   help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=2e-5,
                   help='weight decay (default: 2e-5)')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                   help='Clip gradient norm (default: None, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                   help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument('--layer-decay', type=float, default=None,
                   help='layer-wise learning rate decay (default: None)')
group.add_argument('--opt-kwargs', nargs='*',
                   default={}, action=utils.ParseKwargs)

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "step"')
group.add_argument('--sched-on-updates', action='store_true', default=False,
                   help='Apply LR scheduler step on update instead of epoch end.')
group.add_argument('--lr', type=float, default=None, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')
group.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
                   help='base learning rate: lr = lr_base * global_batch_size / base_size')
group.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
                   help='base learning rate batch size (divisor, default: 256).')
group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                   help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                   help='learning rate noise on/off epoch percentages')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                   help='learning rate noise limit percent (default: 0.67)')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                   help='learning rate noise std-dev (default: 1.0)')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                   help='learning rate cycle len multiplier (default: 1.0)')
group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                   help='amount to decay each learning rate cycle (default: 0.5)')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                   help='learning rate cycle limit, cycles enabled if > 1')
group.add_argument('--lr-k-decay', type=float, default=1.0,
                   help='learning rate k-decay for cosine/poly (default: 1.0)')
group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                   help='warmup learning rate (default: 1e-5)')
group.add_argument('--min-lr', type=float, default=0, metavar='LR',
                   help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                   help='number of epochs to train (default: 300)')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                   help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                   help='manual epoch number (useful on restarts)')
group.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
                   help='list of decay epoch indices for multistep lr. must be increasing')
group.add_argument('--decay-epochs', type=float, default=90, metavar='N',
                   help='epoch interval to decay LR')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                   help='epochs to warmup LR, if scheduler supports')
group.add_argument('--warmup-prefix', action='store_true', default=False,
                   help='Exclude warmup period from decay schedule.'),
group.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                   help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                   help='patience epochs for Plateau LR scheduler (default: 10)')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                   help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                   help='Disable all training augmentation, override other train aug args')
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                   help='Random resize scale (default: 0.08 1.0)')
group.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                   help='Random resize aspect ratio (default: 0.75 1.33)')
group.add_argument('--hflip', type=float, default=0.5,
                   help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.,
                   help='Vertical flip training aug probability')
group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                   help='Color jitter factor (default: 0.4)')
group.add_argument('--aa', type=str, default=None, metavar='NAME',
                   help='Use AutoAugment policy. "v0" or "original". (default: None)'),
group.add_argument('--aug-repeats', type=float, default=0,
                   help='Number of augmentation repetitions (distributed training only) (default: 0)')
group.add_argument('--aug-splits', type=int, default=0,
                   help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
group.add_argument('--jsd-loss', action='store_true', default=False,
                   help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
group.add_argument('--bce-loss', action='store_true', default=False,
                   help='Enable BCE loss w/ Mixup/CutMix use.')
group.add_argument('--bce-target-thresh', type=float, default=None,
                   help='Threshold for binarizing softened BCE targets (default: None, disabled)')
group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                   help='Random erase prob (default: 0.)')
group.add_argument('--remode', type=str, default='pixel',
                   help='Random erase mode (default: "pixel")')
group.add_argument('--recount', type=int, default=1,
                   help='Random erase count (default: 1)')
group.add_argument('--resplit', action='store_true', default=False,
                   help='Do not random erase first (clean) augmentation split')
group.add_argument('--mixup', type=float, default=0.0,
                   help='mixup alpha, mixup enabled if > 0. (default: 0.)')
group.add_argument('--cutmix', type=float, default=0.0,
                   help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                   help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
group.add_argument('--mixup-prob', type=float, default=1.0,
                   help='Probability of performing mixup or cutmix when either/both is enabled')
group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                   help='Probability of switching to cutmix when both mixup and cutmix enabled')
group.add_argument('--mixup-mode', type=str, default='batch',
                   help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                   help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
group.add_argument('--smoothing', type=float, default=0.1,
                   help='Label smoothing (default: 0.1)')
group.add_argument('--train-interpolation', type=str, default='random',
                   help='Training interpolation (random, bilinear, bicubic default: "random")')
group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                   help='Dropout rate (default: 0.)')
group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                   help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                   help='Drop path rate (default: None)')
group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                   help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group(
    'Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
group.add_argument('--bn-momentum', type=float, default=None,
                   help='BatchNorm momentum override (if not None)')
group.add_argument('--bn-eps', type=float, default=None,
                   help='BatchNorm epsilon override (if not None)')
group.add_argument('--sync-bn', action='store_true',
                   help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
group.add_argument('--dist-bn', type=str, default='reduce',
                   help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
group.add_argument('--split-bn', action='store_true',
                   help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
group = parser.add_argument_group(
    'Model exponential moving average parameters')
group.add_argument('--model-ema', action='store_true', default=False,
                   help='Enable tracking moving average of model weights')
group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                   help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
group.add_argument('--model-ema-decay', type=float, default=0.9998,
                   help='decay factor for model weights moving average (default: 0.9998)')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
group.add_argument('--worker-seeding', type=str, default='all',
                   help='worker seed mode (default: all)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging training status')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                   help='how many batches to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=1, metavar='N',
                   help='number of checkpoints to keep (default: 10)')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                   help='how many training processes to use (default: 4)')
group.add_argument('--save-images', action='store_true', default=False,
                   help='save images of input bathes every log interval for debugging')
group.add_argument('--amp', action='store_true', default=False,
                   help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
group.add_argument('--amp-dtype', default='float16', type=str,
                   help='lower precision AMP dtype (default: float16)')
group.add_argument('--amp-impl', default='native', type=str,
                   help='AMP impl to use, "native" or "apex" (default: native)')
group.add_argument('--no-ddp-bb', action='store_true', default=False,
                   help='Force broadcast buffers for native DDP to off.')
group.add_argument('--synchronize-step', action='store_true', default=False,
                   help='torch.cuda.synchronize() end of each step')
group.add_argument('--pin-mem', action='store_true', default=False,
                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--no-prefetcher', action='store_true', default=False,
                   help='disable fast prefetcher')
group.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output')
group.add_argument('--run-name', default='default', type=str, metavar='NAME',
                   help='Name of this run')
group.add_argument('--eval-metric', default='acc_comb', type=str, metavar='EVAL_METRIC',
                   help='Best metric (default: "acc_comb"')
group.add_argument('--tta', type=int, default=0, metavar='N',
                   help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                   help='use the multi-epochs-loader to save time at the beginning of every epoch')
group.add_argument('--log-wandb', action='store_true', default=False,
                   help='log training and validation metrics to wandb')

# Ensemble EfficientNet arguments
group.add_argument("--cut-point", type=int, default=4, dest="cut_point",
                   help="Cut point of the origin efficient net")
group.add_argument("--loss-weights", type=float, nargs="+", dest="loss_weights", default=[1, 1, 1],
                   help="Weights of training losses")
group.add_argument("--width-ratio", type=float, dest="width_ratio", default=1.0,
                   help="Width ratio of the model to adjust")
group.add_argument("--contrastive-eps", type=float, dest="contrastive_eps", default=4.0,
                   help="Contrastive loss eps")
group.add_argument("--head-channels", type=int, default=1280, dest="head_channels",
                   help="Number of channels in the last layer")
group.add_argument("--use-contrastive", type=bool, default=False, dest="use_contrastive",
                   help="Use contrastive loss in the loss formulation")
# group.add_argument("--use-contrastive",
#     action="store_true",
#     help="Enable contrastive loss"
# )
group.add_argument("--head-type", type=str, default='', dest="head_type",
                   help="Merge head type (default: '')")
group.add_argument("--hidden-dim", type=int, default=256, dest="hidden_dim",
                   help="Merge head FC hidden dim (default: 256)")
group.add_argument("--freeze-nn1", type=bool, default=False, dest="freeze_nn1",
                   help="Freeze NN-1 weights while training")
group.add_argument("--freeze-nn2", type=bool, default=False, dest="freeze_nn2",
                   help="Freeze NN-2 weights while training")
group.add_argument('--checkpoint-nn1', default='', type=str, metavar='PATH', dest="checkpt_nn1",
                   help='Initialize NN-1 Backbone model from this checkpoint (default: none)')
group.add_argument('--checkpoint-nn2', default='', type=str, metavar='PATH', dest="checkpt_nn2",
                   help='Initialize NN-2 Backbone model from this checkpoint (default: none)')
group.add_argument('--tsne-nn1', default='', type=str, metavar='PATH', dest="tsne_nn1",
                   help='t-SNE output path for NN-1 (default: none)')
group.add_argument('--tsne-nn2', default='', type=str, metavar='PATH', dest="tsne_nn2",
                   help='t-SNE output path for NN-2 (default: none)')
group.add_argument('--tsne-nn3', default='', type=str, metavar='PATH', dest="tsne_nn3",
                   help='t-SNE output path for NN-3 (default: none)')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher
    args.grad_accum_steps = max(1, args.grad_accum_steps)
    device = utils.init_distributed_device(args)
    # device = torch.device('cpu')
    if args.distributed:
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(
            f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    # TODO:Fix it in the commands line arguments
    args.use_contrastive = False
    print("Use contastive: ", args.use_contrastive)

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            use_amp = 'apex'
            assert args.amp_dtype == 'float16'
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            use_amp = 'native'
            assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    # model = EnsembleEfficientNet(args.model,
    #                              pretrained=args.pretrained,
    #                              in_chans=in_chans,
    #                              num_classes=args.num_classes,
    #                              drop_rate=args.drop,
    #                              drop_path_rate=args.drop_path,
    #                              drop_block_rate=args.drop_block,
    #                              global_pool=args.gp,
    #                              bn_momentum=args.bn_momentum,
    #                              bn_eps=args.bn_eps,
    #                              scriptable=args.torchscript,
    #                              checkpoint_path=args.initial_checkpoint,
    #                              **args.model_kwargs)
    if args.model == 'efficientnet_b0':
        model = EnsembleEfficientNet(num_classes=args.num_classes, cut_point=args.cut_point, width_ratio=args.width_ratio,
                                     last_channels=args.head_channels, head_type=args.head_type, hidden_dim=args.hidden_dim)
    elif args.model == 'efficientnet_b1':
        model = EnsembleEfficientNetB1(num_classes=args.num_classes, cut_point=args.cut_point, width_ratio=args.width_ratio,
                                       last_channels=args.head_channels, head_type=args.head_type, hidden_dim=args.hidden_dim)
    elif args.model == 'resnet50':
        model = EnsembleResnet50(num_classes=args.num_classes,
                                 cut_point=args.cut_point, head_channels=args.head_channels, head_type=args.head_type, hidden_dim=args.hidden_dim)
    elif args.model == 'vit':
        model = EnsembleViT(
            image_size=224, num_classes=args.num_classes, cut_point=args.cut_point)
    elif args.model == 'efficientnet_b0_ss1':
        model = EfficientNetEarly(num_classes=args.num_classes, cut_point=args.cut_point, use_head=False,
                                  last_channels=args.head_channels)

    if args.initial_checkpoint:
        checkpoint = torch.load(args.initial_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

    if args.checkpt_nn1 or args.checkpt_nn2:
        model.initialize_encoders(args.checkpt_nn1, args.checkpt_nn2, device)

    if args.freeze_nn1 or args.freeze_nn2:
        model.freeze_and_unfreeze_encoders(args.freeze_nn1, args.freeze_nn2)

    if args.head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(args.head_init_scale)
            model.get_classifier().bias.mul_(args.head_init_scale)
    if args.head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, args.head_init_bias)

    if args.num_classes is None:
        assert hasattr(
            model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        # FIXME handle model default vs config num_classes more elegantly
        args.num_classes = model.num_classes

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if utils.is_primary(args):
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(
        vars(args), model=model, verbose=utils.is_primary(args))

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.to(device=device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ''  # disable dist_bn when sync BN active
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not args.torchcompile
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any(
                [o in on for o in ('ada', 'lamb')]) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) '
                f'and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.')

    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        assert device.type == 'cuda'
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        try:
            amp_autocast = partial(
                torch.autocast, device_type=device.type, dtype=amp_dtype)
        except (AttributeError, TypeError):
            # fallback to CUDA only AMP for PyTorch < 1.10
            assert device.type == 'cuda'
            amp_autocast = torch.cuda.amp.autocast
        if device.type == 'cuda' and amp_dtype == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler()
        if utils.is_primary(args):
            _logger.info(
                'Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(args):
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[
                              device], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    if args.torchcompile:
        # torch compile should be done after DDP
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        model = torch.compile(model, backend=args.torchcompile)

    # create the train and eval datasets
    if args.data and not args.data_dir:
        args.data_dir = args.data

    # dataset_train = Food101(root=args.data_dir,
    #                         split="train",
    #                         download=args.dataset_download)

    # dataset_train = create_dataset(
    #     args.dataset,
    #     root=args.data_dir,
    #     split=args.train_split,
    #     is_training=True,
    #     class_map=args.class_map,
    #     download=args.dataset_download,
    #     batch_size=args.batch_size,
    #     seed=args.seed,
    #     repeats=args.epoch_repeats,
    # )

    # dataset_eval = Food101(root=args.data_dir,
    #                        split="test",
    #                        download=args.dataset_download)
    dataset_eval = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
    )

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )
        if args.prefetcher:
            # collate conflict (need to support deinterleaving in collate mixup)
            assert not num_aug_splits
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    # loader_train = create_loader(
    #     dataset_train,
    #     input_size=data_config['input_size'],
    #     batch_size=args.batch_size,
    #     is_training=True,
    #     use_prefetcher=args.prefetcher,
    #     no_aug=args.no_aug,
    #     re_prob=args.reprob,
    #     re_mode=args.remode,
    #     re_count=args.recount,
    #     re_split=args.resplit,
    #     scale=args.scale,
    #     ratio=args.ratio,
    #     hflip=args.hflip,
    #     vflip=args.vflip,
    #     color_jitter=args.color_jitter,
    #     auto_augment=args.aa,
    #     num_aug_repeats=args.aug_repeats,
    #     num_aug_splits=num_aug_splits,
    #     interpolation=train_interpolation,
    #     mean=data_config['mean'],
    #     std=data_config['std'],
    #     num_workers=args.workers,
    #     distributed=args.distributed,
    #     collate_fn=collate_fn,
    #     pin_memory=args.pin_mem,
    #     device=device,
    #     use_multi_epochs_loader=args.use_multi_epochs_loader,
    #     worker_seeding=args.worker_seeding,
    # )

    eval_workers = args.workers
    if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
        # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
        eval_workers = min(2, args.workers)
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=eval_workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        device=device,
    )

    # setup loss function
    # if args.jsd_loss:
    #     assert num_aug_splits > 1  # JSD only valid with aug splits set
    #     train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    # elif mixup_active:
    #     # smoothing is handled with mixup target transform which outputs sparse, soft targets
    #     if args.bce_loss:
    #         train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
    #     else:
    #         train_loss_fn = SoftTargetCrossEntropy()
    # elif args.smoothing:
    #     if args.bce_loss:
    #         train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
    #     else:
    #         train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    # else:

    # train_ensemble_loss = EnsembleLoss(
    #     nn.CrossEntropyLoss(), args.loss_weights)
    # valid_ensemble_loss = EnsembleLoss(
    #     nn.CrossEntropyLoss(), args.loss_weights)

    # train_loss_fn = train_ensemble_loss.to(device=device)
    # validate_loss_fn = valid_ensemble_loss.to(device=device)
    contrastive_weight = 1e-2

    # train_loss_fn = EnsembleContrastiveLoss(train_ensemble_loss, eps=args.contrastive_eps,
    #                                         lamb=contrastive_weight, use_contrastive=args.use_contrastive).to(device=device)
    # validate_loss_fn = EnsembleContrastiveLoss(
    #     valid_ensemble_loss, eps=args.contrastive_eps, lamb=contrastive_weight, use_contrastive=args.use_contrastive).to(device=device)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = ''
        decreasing = True if eval_metric == 'loss' else False
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    if utils.is_primary(args) and args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`")

    raw_model = model.module if hasattr(model, 'module') else model

    tsne1_path = args.tsne_nn1 if args.tsne_nn1 else './tsne/tsne1' + \
        time.now().strftime("%Y%m%d-%H%M%S") + '.npy'
    tsne2_path = args.tsne_nn2 if args.tsne_nn2 else './tsne/tsne2' + \
        time.now().strftime("%Y%m%d-%H%M%S") + '.npy'

    # if os.path.exists(tsne1_path) and os.path.exists(tsne2_path):
    #     tsne1 = np.load(tsne1_path)
    #     tsne2 = np.load(tsne2_path)
    #     _logger.info('Loaded t-SNE results from disk.')
    # else:
    #     f1, f2, labels = extract_features(model, loader_eval, device)
    #     tsne1 = run_tsne(f1, labels, title='t-SNE-f1',
    #                      num_classes=args.num_classes)
    #     np.save(tsne1_path, tsne1)
    #     _logger.info('Running t-SNE for NN-1 features done and saved.')
    #     tsne2 = run_tsne(f2, labels, title='t-SNE-f2',
    #                      num_classes=args.num_classes)
    #     np.save(tsne2_path, tsne2)
    #     _logger.info('Running t-SNE for NN-2 features done and saved.')

    if os.path.exists(tsne1_path):
        tsne1 = np.load(tsne1_path)
        # tsne2 = np.load(tsne2_path)
        _logger.info('Loaded t-SNE results from disk.')
    else:
        f1, labels = extract_features_ss1(model, loader_eval, device)
        tsne1 = run_tsne(f1, labels, title='t-SNE-f1',
                         num_classes=args.num_classes)
        np.save(tsne1_path, tsne1)
        _logger.info('Running t-SNE for NN-1 features done and saved.')
        # tsne2 = run_tsne(f2, labels, title='t-SNE-f2',
        #                  num_classes=args.num_classes)
        # np.save(tsne2_path, tsne2)
        # _logger.info('Running t-SNE for NN-2 features done and saved.')

    # run_tsne(f2, labels, num_classes=args.num_classes)

    # plt.style.use('science')
    # plt.figure(figsize=(10, 8))

    # # Plot t-SNE results for Model 1
    # scatter1 = plt.scatter(
    #     tsne1[:, 0], tsne1[:, 1], c='blue', s=10, alpha=0.2, label=r'$h_{\{1\}}$'
    # )

    # # Plot t-SNE results for Model 2
    # scatter2 = plt.scatter(
    #     tsne2[:, 0], tsne2[:, 1], c='red', s=10, alpha=0.2, label=r'$h_{\{2\}}$'
    # )

    # # Add legend for models
    # plt.legend(
    #     title="Upstream models", loc="best", bbox_to_anchor=(1, 1))

    # # Add title and save the plot
    # # plt.title('t-sne for EfficientNet-B0 ($\mathcal{B}$5)')
    # # plt.tight_layout()
    # save_path = args.output if args.output else 't-sne-' + \
    #     time.now().strftime("%Y%m%d-%H%M%S") + '.png'
    # plt.savefig(save_path, dpi=300)
    # plt.show()


def extract_features_ss1(model, dataloader, device):
    model.eval()
    # features = []
    f1 = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            x = model.encoder(images)  # This sets encoder1._x and encoder2._x
            x = x.detach().cpu()
            # x1 = model.encoder1._x.detach().cpu()
            # x2 = model.encoder2._x.detach().cpu()
            # combined = torch.cat([x1, x2], dim=1)
            x = torch.mean(x, dim=(2, 3))
            # x2 = torch.mean(x2, dim=(2, 3))
            f1.append(x)
            # f2.append(x2)
            # features.append(combined)
            labels.append(targets)
    return torch.cat(f1).numpy(), torch.cat(labels).cpu().numpy()


def extract_features(model, dataloader, device):
    model.eval()
    # features = []
    f1, f2 = [], []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            _ = model(images)  # This sets encoder1._x and encoder2._x
            x1 = model.encoder1._x.detach().cpu()
            x2 = model.encoder2._x.detach().cpu()
            # combined = torch.cat([x1, x2], dim=1)
            x1 = torch.mean(x1, dim=(2, 3))
            x2 = torch.mean(x2, dim=(2, 3))
            f1.append(x1)
            f2.append(x2)
            # features.append(combined)
            labels.append(targets)
    return torch.cat(f1).numpy(), torch.cat(f2).numpy(), torch.cat(labels).cpu().numpy()


def extract_features_3(model, dataloader, device):
    model.eval()
    # features = []
    f1, f2, f3 = [], [], []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            _ = model(images)  # This sets encoder1._x and encoder2._x
            x1 = model.encoder1._x.detach().cpu()
            x2 = model.encoder2._x.detach().cpu()
            x3 = model.encoder3._x.detach().cpu()

            # combined = torch.cat([x1, x2], dim=1)
            x1 = torch.mean(x1, dim=(2, 3))
            x2 = torch.mean(x2, dim=(2, 3))
            x3 = torch.mean(x3, dim=(2, 3))
            f1.append(x1)
            f2.append(x2)
            f3.append(x3)
            # features.append(combined)
            labels.append(targets)
    return torch.cat(f1).numpy(), torch.cat(f2).numpy(), torch.cat(f3).numpy(), torch.cat(labels).cpu().numpy()


def run_tsne(features, labels, title='t-SNE', num_classes=100):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
                init='random', random_state=42)
    tsne_result = tsne.fit_transform(features)
    return tsne_result


if __name__ == '__main__':
    main()
