#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial
from sys import maxsize

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from ensemble_efficient_net_b0 import EnsembleEfficientNet

from timm.data import create_dataset, create_loader, resolve_data_config, ImageNetInfo, infer_imagenet_subset
from timm.layers import apply_test_time_pool
from timm.models import create_model
from timm.utils import AverageMeter, setup_default_logging, set_jit_fuser, ParseKwargs
from timm import utils

try:
    from apex import amp
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
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')


_FMT_EXT = {
    'json': '.json',
    'json-record': '.json',
    'json-split': '.json',
    'parquet': '.parquet',
    'csv': '.csv',
}

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')


def contrastive_loss(x1, x2, eps=4, lamb=1e-2):
    m = x1.size(0)
    diff = lamb / m * torch.sum((x1 - x2)**2)

    return torch.clip(eps-diff, min=0)


class EnsembleLoss(object):
    def __init__(self, loss_fn, weights):
        self.loss_fn = loss_fn
        self.weights = weights

    def to(self, device):
        self.loss_fn.to(device)
        return self

    def __call__(self, output, target, super_target):
        y1_loss = self.loss_fn(output[0], super_target)
        y2_loss = self.loss_fn(output[1], super_target)
        y_comb_loss = self.loss_fn(output[2], target)

        return self.weights[0] * y1_loss + self.weights[1] * y2_loss + self.weights[2] * y_comb_loss


class EnsembleContrastiveLoss(object):
    def __init__(self, ensemble_loss_fn, eps=4, lamb=1e-2):
        self.ensemble_loss_fn = ensemble_loss_fn
        self.eps = eps
        self.lamb = lamb

    def to(self, device):
        self.ensemble_loss_fn.to(device)
        return self

    def __call__(self, output, target, super_target):
        ensemble_loss = self.ensemble_loss_fn(output, target, super_target)
        # c_loss = contrastive_loss(output[0], output[1], self.eps, self.lamb)

        # return ensemble_loss + c_loss
        return ensemble_loss


parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=None, nargs=3, type=int, metavar='N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--amp', action='store_true', default=False,
                    help='use Native AMP for mixed precision training')
parser.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype (default: float16)')
parser.add_argument('--model-dtype', default=None, type=str,
                    help='Model dtype override (non-AMP) (default: float32)')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--model-kwargs', nargs='*',
                    default={}, action=ParseKwargs)
parser.add_argument('--torchcompile-mode', type=str, default=None,
                    help="torch.compile mode (default: None).")

scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', default=False, action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                             help="Enable AOT Autograd support.")

parser.add_argument('--results-dir', type=str, default=None,
                    help='folder for output results')
parser.add_argument('--results-file', type=str, default=None,
                    help='results filename (relative to results-dir)')
parser.add_argument('--results-format', type=str, nargs='+', default=['csv'],
                    help='results format (one of "csv", "json", "json-split", "parquet")')
parser.add_argument('--results-separate-col', action='store_true', default=False,
                    help='separate output columns per result index.')
parser.add_argument('--topk', default=1, type=int,
                    metavar='N', help='Top-k to output to CSV')
parser.add_argument('--fullname', action='store_true', default=False,
                    help='use full sample name in output (not just basename).')
parser.add_argument('--filename-col', type=str, default='filename',
                    help='name for filename / sample name column')
parser.add_argument('--index-col', type=str, default='index',
                    help='name for output indices column(s)')
parser.add_argument('--label-col', type=str, default='label',
                    help='name for output indices column(s)')
parser.add_argument('--output-col', type=str, default=None,
                    help='name for logit/probs output column(s)')
parser.add_argument('--output-type', type=str, default='prob',
                    help='output type colum ("prob" for probabilities, "logit" for raw logits)')
parser.add_argument('--label-type', type=str, default='description',
                    help='type of label to output, one of  "none", "name", "description", "detailed"')
parser.add_argument('--include-index', action='store_true', default=False,
                    help='include the class index in results')
parser.add_argument('--exclude-output', action='store_true', default=False,
                    help='exclude logits/probs from results, just indices. topk must be set !=0.')
parser.add_argument('--no-console-results', action='store_true', default=False,
                    help='disable printing the inference results to the console')


# Adittional Parameters added
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--run-name', default='default', type=str,
                   metavar='NAME', help='Name of this run')
group.add_argument("--cut-point", type=int, default=4, dest="cut_point",
                   help="Cut point of the origin efficient net")
group.add_argument("--loss-weights", type=float, nargs="+", dest="loss_weights", default=[1, 1, 1],
                   help="Weights of training losses")
group.add_argument("--width-ratio", type=float, dest="width_ratio", default=1.0,
                   help="Width ratio of the model to adjust")
group.add_argument("--contrastive-eps", type=float, dest="contrastive_eps", default=4.0,
                   help="Contrastive loss eps")
group.add_argument("--last-channels", type=int, default=1280, dest="last_channels",
                   help="Number of channels in the last layer")
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging training status')


def main():
    setup_default_logging()
    args = parser.parse_args()
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    # resolve AMP arguments based on PyTorch / Apex availability
    amp_autocast = suppress
    if args.amp:
        assert args.amp_dtype in ('float16', 'bfloat16')
        amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
        amp_autocast = partial(
            torch.autocast, device_type=device.type, dtype=amp_dtype)
        _logger.info(
            'Running inference in mixed precision with native PyTorch AMP.')
    else:
        _logger.info('Running inference in float32. AMP not enabled.')

    if args.fuser:
        set_jit_fuser(args.fuser)

    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model = EnsembleEfficientNet(num_classes=args.num_classes, cut_point=args.cut_point, width_ratio=args.width_ratio,
                                 last_channels=args.last_channels)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

    if args.num_classes is None:
        assert hasattr(
            model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    _logger.info(
        f'Model {args.model} created, param count: {sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    model = model.to(device=device)
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        model = torch.jit.script(model)
    elif args.torchcompile:
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        torch._dynamo.reset()
        model = torch.compile(
            model, backend=args.torchcompile, mode=args.torchcompile_mode)
    elif args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(
            model, device_ids=list(range(args.num_gpu)))

    root_dir = args.data or args.data_dir
    dataset = create_dataset(
        root=root_dir,
        name=args.dataset,
        split=args.split,
        class_map=args.class_map,
    )

    if test_time_pool:
        data_config['crop_pct'] = 1.0

    workers = 1 if 'tfds' in args.dataset or 'wds' in args.dataset else args.workers
    loader = create_loader(
        dataset,
        batch_size=args.batch_size,
        use_prefetcher=True,
        num_workers=workers,
        device=device,
        **data_config,
    )

    to_label = None
    if args.label_type in ('name', 'description', 'detail'):
        imagenet_subset = infer_imagenet_subset(model)
        if imagenet_subset is not None:
            dataset_info = ImageNetInfo(imagenet_subset)
            if args.label_type == 'name':
                def to_label(x): return dataset_info.index_to_label_name(x)
            elif args.label_type == 'detail':
                def to_label(x): return dataset_info.index_to_description(
                    x, detailed=True)
            else:
                def to_label(x): return dataset_info.index_to_description(x)
            to_label = np.vectorize(to_label)
        else:
            _logger.error(
                "Cannot deduce ImageNet subset from model, no labelling will be performed.")

    # test_loss_fn = valid_ensemble_loss.to(device=device)
    contrastive_weight = 1e-2
    test_ensemble_loss = EnsembleLoss(
        torch.nn.CrossEntropyLoss(), args.loss_weights)

    fine_to_coarse = torch.tensor([4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                   16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                   18,  1,  2, 15,  6,  0, 17,  8, 14, 13], device=device)

    test_loss_fn = EnsembleContrastiveLoss(
        test_ensemble_loss, eps=args.contrastive_eps, lamb=contrastive_weight).to(device=device)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("NN-1 encoder number of parameters",
                         model.encoder1_stat.total_params)
        mlflow.log_param("NN-1 classifier number of parameters",
                         model.classifier1_stat.total_params)
        mlflow.log_param("NN-2 encoder number of parameters",
                         model.encoder2_stat.total_params)
        mlflow.log_param("NN-2 classifier number of parameters",
                         model.classifier2_stat.total_params)
        mlflow.log_param("NN-12 number of parameters",
                         model.classifier_comb_stat.total_params)
        eval_metrics = test(model, loader, test_loss_fn, args,
                            fine_to_coarse, amp_autocast=amp_autocast, device=device)


def save_results(df, results_filename, results_format='csv', filename_col='filename'):
    np.set_printoptions(threshold=maxsize)
    results_filename += _FMT_EXT[results_format]
    if results_format == 'parquet':
        df.set_index(filename_col).to_parquet(results_filename)
    elif results_format == 'json':
        df.set_index(filename_col).to_json(
            results_filename, indent=4, orient='index')
    elif results_format == 'json-records':
        df.to_json(results_filename, lines=True, orient='records')
    elif results_format == 'json-split':
        df.to_json(results_filename, indent=4, orient='split', index=False)
    else:
        df.to_csv(results_filename, index=False)


def class_wise_accuracy(output, target, num_classes=100, topk=(1,)):
    """Computes class-wise accuracy for each fine class."""
    batch_size = target.size(0)
    maxk = min(max(topk), output.size()[1])
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.view(-1)  # Reshape to (batch_size,)

    # Initialize correct and total counts per class
    class_correct = torch.zeros(
        num_classes, dtype=torch.float, device=output.device)
    class_total = torch.zeros(
        num_classes, dtype=torch.float, device=output.device)

    # Count correct predictions per class
    for i in range(batch_size):
        true_label = target[i].item()
        pred_label = pred[i].item()

        class_total[true_label] += 1  # Count total occurrences of the class
        if pred_label == true_label:
            class_correct[true_label] += 1  # Count correct predictions

    # Compute accuracy per class (avoid division by zero)
    class_accuracy = torch.where(class_total > 0, class_correct /
                                 class_total * 100, torch.tensor(0.0, device=output.device))

    # Returns a tensor of shape (num_classes,) with accuracy per class
    return class_accuracy, class_total

# def super_accuracy(output, target, fine_to_coarse, device, topk=(1,)):
#     """Computes the super class accuracy based on summation of probabilities"""
#     super_target = fine_to_coarse[target]
#     batch_size = super_target.size(0)

#     comb_probs = F.softmax(output, dim=-1)
#     super_comb_probs = torch.zeros(comb_probs.size(0), 20, device=device)
#     for i in range(20):
#         sub_idx = (fine_to_coarse == i).nonzero(as_tuple=True)[0]
#         super_comb_probs[:, i] = comb_probs[:, sub_idx].sum(dim=-1)

#     return utils.accuracy(super_comb_probs, super_target, topk=(1,))


def super_accuracy(output, target, fine_to_coarse, device, topk=(1,)):
    """Computes the super class accuracy based on the fine class predicted """
    super_target = fine_to_coarse[target]
    batch_size = super_target.size(0)

    maxk = min(max(topk), output.size()[1])
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    super_pred = fine_to_coarse[pred]
    correct = super_pred.eq(super_target.reshape(1, -1).expand_as(super_pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

# def classwise_accuracy(output, target, topk=(1,)):
#     maxk = min(max(topk), output.size()[1])
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()


def pred(output, target, topk=(1,)):
    maxk = min(max(topk), output.size()[1])
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = target.reshape(1, -1).expand_as(pred)
    return pred, correct


def test(
        model,
        loader,
        loss_fn,
        args,
        fine_to_coarse,
        amp_autocast=suppress,
        device=torch.device('cuda'),
        log_suffix=''
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    diff_m = utils.AverageMeter()
    acc1_m = utils.AverageMeter()
    acc2_m = utils.AverageMeter()
    acc_comb_m = utils.AverageMeter()

    # class_acc1_m = [utils.AverageMeter() for _ in range(args.num_classes)]
    # class_acc2_m = [utils.AverageMeter() for _ in range(args.num_classes)]

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    output1_preds = []
    output2_preds = []
    output_comb_preds = []

    labels = []

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = input.to(device)
            target = target.to(device)
            super_target = fine_to_coarse[target]

            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
                # if isinstance(output, (tuple, list)):
                #     output = output[0]

                # augmentation reduction
                # reduce_factor = args.tta
                # if reduce_factor > 1:
                #     output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                #     target = target[0:target.size(0):reduce_factor]

                diff = torch.sum((output[0] - output[1])
                                 ** 2) / output[0].size(0)
                loss = loss_fn(output, target, super_target)

            # acc1 = super_accuracy(
            #     output[0], target, fine_to_coarse, device, topk=(1,))[0]
            # acc2 = super_accuracy(
            #     output[1], target, fine_to_coarse, device, topk=(1,))[0]
            # acc_comb = super_accuracy(
            #     output[2], target, fine_to_coarse, device, topk=(1,))[0]
            # class_acc1 = class_wise_accuracy(output[0], target, args.num_classes, topk=(1,))
            # class_acc2 = class_wise_accuracy(output[1], target, args.num_classes, topk=(1,))

            acc1 = utils.accuracy(output[0], target, topk=(1,))[0]
            acc2 = utils.accuracy(output[1], target, topk=(1,))[0]
            acc_comb = utils.accuracy(output[2], target, topk=(1,))[0]

            reduced_loss = loss.data

            output1_preds.extend(pred(output[0], target)[0].cpu().tolist())
            output2_preds.extend(pred(output[1], target)[0].cpu().tolist())
            output_comb_preds.extend(pred(output[2], target)[0].cpu().tolist())

            labels.extend(pred(output[0], target)[1].cpu().tolist())

            if device.type == 'cuda':
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            diff_m.update(diff.item(), input.size(0))
            acc1_m.update(acc1.item(), output[0].size(0))
            acc2_m.update(acc2.item(), output[1].size(0))
            # for i in range(args.num_classes):
            #     if class_acc1[1][i].item() > 0:
            #         class_acc1_m[i].update(class_acc1[0][i].item(), class_acc1[1][i].item())
            #     if class_acc2[1][i].item() > 0:
            #         class_acc2_m[i].update(class_acc2[0][i].item(), class_acc2[1][i].item())
            acc_comb_m.update(acc_comb.item(), output[2].size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  '
                    f'Diff: {diff_m.val:>7.3f} ({diff_m.avg:>6.3f})'
                    f'Acc@1: {acc1_m.val:>7.3f} ({acc1_m.avg:>7.3f})  '
                    f'Acc@2: {acc2_m.val:>7.3f} ({acc2_m.avg:>7.3f})'
                    f'Acc@comb: {acc_comb_m.val:>7.3f} ({acc_comb_m.avg:>7.3f})'
                )

    # class_acc1_avg = [round(_.avg, 2) for _ in class_acc1_m]
    # class_acc2_avg = [round(_.avg, 2) for _ in class_acc2_m]

    output1_preds = [item for sublist in output1_preds for item in sublist]
    output2_preds = [item for sublist in output2_preds for item in sublist]
    output_comb_preds = [
        item for sublist in output_comb_preds for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    nn1_f1_score = f1_score(labels, output1_preds, average='macro')
    nn2_f1_score = f1_score(labels, output2_preds, average='macro')
    nn_comb_f1_score = f1_score(labels, output_comb_preds, average='macro')
    metrics = OrderedDict(
        [('loss', losses_m.avg), ('acc1', acc1_m.avg),
         ('acc2', acc2_m.avg), ('acc_comb', acc_comb_m.avg), ("diff", diff_m.avg)])

    mlflow.log_metric("valid loss", losses_m.avg)
    mlflow.log_metric("Valid Intermediate Diff", diff_m.avg)
    mlflow.log_metric("Valid NN-1 accuracy", acc1_m.avg)
    mlflow.log_metric("Valid NN-2 accuracy", acc2_m.avg)
    mlflow.log_metric("Valid NN-12 accuracy", acc_comb_m.avg)
    mlflow.log_metric("Valid NN-1 F-1 score", nn1_f1_score)
    mlflow.log_metric("Valid NN-2 F-1 score", nn2_f1_score)
    mlflow.log_metric("Valid NN-12 F-1 score", nn_comb_f1_score)

    # with open('class_accuracy1.txt', 'w') as file:
    #     json.dump(class_acc1_avg, file)

    # with open('class_accuracy2.txt', 'w') as file:
    #     json.dump(class_acc2_avg, file)

    # mlflow.log_param("Validate NN-1 classwise accuracy", json.dumps(class_acc1_avg))
    # mlflow.log_params("Validate NN-2 classwise accuracy", json.dumps(class_acc2_avg))

    return metrics


if __name__ == '__main__':
    main()
