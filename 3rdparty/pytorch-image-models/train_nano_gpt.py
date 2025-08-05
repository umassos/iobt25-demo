"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --args.batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import yaml
import pickle
import mlflow
import logging
import argparse
from timm.utils import AverageMeter, setup_default_logging
from contextlib import nullcontext


import numpy as np
import torch
from torchinfo import summary
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from ensemble_nano_gpt import GPTConfig, GPT, EnsembleGPT, GPTEarly

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O

config_parser = parser = argparse.ArgumentParser(
    description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# add this after you’ve created `parser = argparse.ArgumentParser(...)`
group = parser.add_argument_group('Training options')

group.add_argument('--out-dir', type=str, default='./output/train',
                   help='output directory for checkpoints & logs (default: out)')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                   help='Initialize model from this checkpoint (default: none)')
group.add_argument('--eval-interval', type=int, default=2000, metavar='N',
                   help='run evaluation every N iterations (default: 2000)')
group.add_argument('--log-interval', type=int, default=1, metavar='N',
                   help='print training loss every N iterations (default: 1)')
group.add_argument('--eval-iters', type=int, default=200, metavar='N',
                   help='number of iterations to use when evaluating (default: 200)')
group.add_argument('--eval-only', action='store_true', default=False,
                   help='if set, only run evaluation and then exit')
group.add_argument('--always-save-checkpoint', action='store_true', default=False,
                   help='save a checkpoint after every evaluation even if not best')
group.add_argument('--init-from', type=str, default='scratch', metavar='MODE',
                   choices=['scratch', 'resume', 'gpt2*'],
                   help="initialize model from 'scratch', 'resume', or pretrained 'gpt2*' (default: scratch)")

group.add_argument('--wandb-log', action='store_true', default=False,
                   help='enable logging to Weights & Biases')
group.add_argument('--wandb-project', type=str, default='owt',
                   help='Weights & Biases project name (default: owt)')
group.add_argument('--wandb-run-name', type=str, default='gpt2',
                   help='Weights & Biases run name (default: gpt2)')

group.add_argument('--dataset', type=str, default='openwebtext',
                   help='name of the args.dataset to train on (default: openwebtext)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
group.add_argument('--gradient-accumulation-steps', type=int, default=5*8, metavar='N',
                   help='number of gradient accumulation steps (default: 40)')
group.add_argument('-b', '--batch-size', type=int, default=12, metavar='N',
                   help='micro-batch size per GPU (default: 12)')
group.add_argument('--block-size', type=int, default=1024, metavar='N',
                   help='sequence length / context window size (default: 1024)')
group.add_argument('--vocab-size', type=int, default=50257, metavar='N',
                   help='vocabulary size (default: 50257)')

group.add_argument('--n-layer', type=int, default=12, metavar='N',
                   help='number of transformer layers (default: 12)')
group.add_argument('--n-head', type=int, default=12, metavar='N',
                   help='number of attention heads per layer (default: 12)')
group.add_argument('--n-embd', type=int, default=768, metavar='N',
                   help='dimensionality of embeddings and hidden states (default: 768)')
group.add_argument('--dropout', type=float, default=0.0, metavar='R',
                   help='dropout probability for all layers (default: 0.0)')
group.add_argument('--bias', action='store_true', default=False,
                   help='use bias terms in LayerNorm and Linear layers (default: False)')

group.add_argument('-lr', '--learning-rate', type=float, default=6e-4, metavar='LR',
                   help='maximum learning rate (default: 6e-4)')
group.add_argument('--max-iters', type=int, default=200000, metavar='N',
                   help='total number of training iterations (default: 600000)')
group.add_argument('--weight-decay', type=float, default=1e-2, metavar='W',
                   help='weight decay for AdamW (default: 0.1)')
group.add_argument('--beta1', type=float, default=0.9, metavar='B1',
                   help='beta1 for AdamW optimizer (default: 0.9)')
group.add_argument('--beta2', type=float, default=0.95, metavar='B2',
                   help='beta2 for AdamW optimizer (default: 0.95)')
group.add_argument('--grad-clip', type=float, default=1.0, metavar='NORM',
                   help='gradient clipping norm (default: 1.0)')

group.add_argument('--decay-lr', action='store_true', default=True,
                   help='enable learning rate decay schedule (default: True)')
group.add_argument('--warmup-iters', type=int, default=8000, metavar='N',
                   help='number of iterations to warm up LR (default: 2000)')
group.add_argument('--lr-decay-iters', type=int, default=200000, metavar='N',
                   help='iterations over which to decay LR (default: 600000)')
group.add_argument('--min-lr', type=float, default=6e-5, metavar='LR',
                   help='minimum learning rate after decay (default: 6e-5)')

group.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output')
group.add_argument('--run-name', default='default', type=str, metavar='NAME',
                   help='Name of this run')

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
group.add_argument("--use-contrastive", type=bool, default=True, dest="use_contrastive",
                   help="Use contrastive loss in the loss formulation")
group.add_argument("--freeze-nn1", type=bool, default=False, dest="freeze_nn1",
                   help="Freeze NN-1 weights while training")
group.add_argument("--freeze-nn2", type=bool, default=False, dest="freeze_nn2",
                   help="Freeze NN-2 weights while training")
group.add_argument('--checkpoint-nn1', default='', type=str, metavar='PATH', dest="checkpt_nn1",
                   help='Initialize NN-1 Backbone model from this checkpoint (default: none)')
group.add_argument('--checkpoint-nn2', default='', type=str, metavar='PATH', dest="checkpt_nn2",
                   help='Initialize NN-2 Backbone model from this checkpoint (default: none)')

# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.
# system
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = 'bfloat16' if torch.cuda.is_available(
) and torch.cuda.is_bf16_supported() else 'float16'
# use PyTorch 2.0 to compile the model to be faster
compile = False
# -----------------------------------------------------------------------------
# config_keys = [k for k, v in globals().items() if not k.startswith(
#     '_') and isinstance(v, (int, float, bool, str))]
# # overrides from command line or config file
# exec(open('configurator.py').read())
# config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

_logger = logging.getLogger('train')
setup_default_logging()


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


args, args_text = _parse_args()

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    print(f'Available device count{torch.cuda.device_count()}')
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    # this process will do logging, checkpointing etc.
    master_process = ddp_rank == 0
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert args.gradient_accumulation_steps % ddp_world_size == 0
    args.gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = args.gradient_accumulation_steps * \
    ddp_world_size * args.batch_size * args.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# if master_process:
#     os.makedirs(args.out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
# for later use in torch.autocast
device_type = 'cuda' if 'cuda' in device else 'cpu'
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32,
           'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
    device_type=device_type, dtype=ptdtype)

# poor man's data loader
# data_dir = os.path.join(
#     '/work/pi_shenoy_umass_edu/kgudipaty/datasets/shakespeare/nanoGPT/data/shakespeare_char', args.dataset)


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(args.data_dir, 'train.bin'),
                         dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(args.data_dir, 'val.bin'),
                         dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i:i+args.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy((data[i+1:i+1+args.block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if args.init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the args.dataset
meta_path = os.path.join(args.data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
if 0 < args.cut_point < 8:
    model_args = dict(n_layer=args.cut_point, n_head=args.n_head,
                      n_embd=args.n_embd, block_size=args.block_size,
                      bias=args.bias, vocab_size=None, dropout=args.dropout)
else:
    model_args = dict(n_layer=8, n_head=8, n_embd=512,
                      block_size=512, bias=False, vocab_size=None, dropout=0.0)


if args.init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        model_args['vocab_size'] = args.vocab_size
    gptconf = GPTConfig(**model_args)
    if 0 < args.cut_point < 12:
        model = GPTEarly(gptconf)
    else:
        model = GPT(gptconf)
elif args.init_from == 'resume':
    print(f"Resuming training from {args.initial_checkpoint}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(args.initial_checkpoint, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    if 0 < model_args['n_layer'] < 6:
        model = GPTEarly(gptconf)
    else:
        model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif args.init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {args.init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=args.dropout)
    model = GPT.from_pretrained(args.init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if args.block_size < model.config.block_size:
    model.crop_args.block_size(args.block_size)
    # so that the checkpoint will have the right value
    model_args['block_size'] = args.block_size
model_stat = summary(model)
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(
    args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)
if args.init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        total_loss = AverageMeter()
        log_name = 'Loss ' + split
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            total_loss.update(loss)
            if k % args.log_interval == 0:
                _logger.info(
                    f'{log_name}: [{k:>4d}/{args.eval_iters}]  '
                    f'Total Loss: {total_loss.val:>7.3f} ({total_loss.avg:>7.3f})  '
                )
        out[split] = {'total_loss': total_loss.avg}

    mlflow.log_metric("Val total loss",
                      total_loss.avg, step=iter_num)
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / (args.warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > args.lr_decay_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_iters) / \
        (args.lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


# logging
# TODO: Fix config
if args.wandb_log and master_process:
    import wandb
    wandb.init(project=args.wandb_project,
               name=args.wandb_run_name, config=config)

# training loop
X, Y = get_batch('train')  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
with mlflow.start_run(run_name=args.run_name):
    try:
        mlflow.log_param("lr", args.learning_rate)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("Model parameters", model_stat.total_params)
        original_model = model.module if ddp else model

        best_iter = None
        while True:

            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num) if args.decay_lr else args.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % args.eval_interval == 0 and master_process:
                losses = estimate_loss()
                print(
                    f"step {iter_num}: train loss {losses['train']['total_loss']:.4f}, val loss {losses['val']['total_loss']:.4f}")
                if args.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                        "mfu": running_mfu*100,  # convert to percentage
                    })
                if losses['val']['total_loss'] < best_val_loss or args.always_save_checkpoint:
                    best_val_loss = losses['val']['total_loss']
                    best_iter = iter_num
                    if iter_num > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': args_text,
                        }
                        output_dir = f'./output/train/{args.experiment}'
                        print(f"saving checkpoint to {output_dir}")
                        os.makedirs(output_dir, exist_ok=True)
                        torch.save(checkpoint, os.path.join(
                            output_dir, 'ckpt.pt'))
            if iter_num == 0 and args.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            total_loss = AverageMeter()
            for micro_step in range(args.gradient_accumulation_steps):
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (
                        micro_step == args.gradient_accumulation_steps - 1)
                with ctx:
                    logits, loss = model(X, Y)
                    # scale the loss to account for gradient accumulation
                    total_loss.update(loss.item())
                    loss = loss / args.gradient_accumulation_steps
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = get_batch('train')
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()

            mlflow.log_metric("Train total loss",
                              total_loss.avg, step=iter_num)
            mlflow.log_metric("Learning rate", lr, step=iter_num)
            # clip the gradient
            if args.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % args.log_interval == 0 and master_process:

                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * args.gradient_accumulation_steps
                # if local_iter_num >= 5:  # let the training loop settle a bit
                # mfu = raw_model.estimate_mfu(
                #     args.batch_size * args.gradient_accumulation_steps, dt)
                # running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu

                _logger.info(
                    f'Train Iter: [{iter_num:>4d}/{args.max_iters}]  '
                    f'Total Loss: {total_loss.val:>7.3f} ({total_loss.avg:>7.3f})  '
                )
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > args.max_iters:
                break

    except KeyboardInterrupt:
        pass

    if best_iter is not None:
        _logger.info(
            '*** Best Loss: {0} (iter {1})'.format(best_val_loss, best_iter))

if ddp:
    destroy_process_group()
