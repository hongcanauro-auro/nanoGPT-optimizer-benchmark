"""
Training script with Subset-Norm optimizer support
Based on nanoGPT train.py
Extended to support Subset-Norm and Subspace-Momentum optimizers
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import gc

# Import modified model
from model_subset_norm import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb logging
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# optimizer settings
optimizer_type = 'AdamW'  # 'AdamW', 'AdamSN', or 'AdamSNSM'
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Subset-Norm specific parameters
subset_size = 'heuristics'  # 'heuristics' or specific int value
correct_bias = True

# Subspace-Momentum specific parameters (for AdamSNSM)
rank = 128
proj_type = 'svd'  # 'svd' or 'random'
update_proj_gap = 200

# learning rate decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# DDP settings
backend = 'nccl'

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

loss_log_file = 'loss_log.csv'
log_all_iterations = True

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config_keys.extend(['loss_log_file','log_all_iterations'])
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}


ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, loss_log_file), 'w') as f:
        f.write('iteration,train_loss,val_loss,timestamp,lr,mfu\n')
    print(f"损失记录将保存到: {os.path.join(out_dir, loss_log_file)}")


def monitor_memory(iteration):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        cached = torch.cuda.memory_reserved() / 1024 ** 3
        max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3

        if iteration % 100 == 0:
            print(f"Iter {iteration}: Alloc: {allocated:.2f}GB, "
                  f"Cached: {cached:.2f}GB, Max: {max_allocated:.2f}GB")

        if iteration > 1000 and max_allocated > 7.5:
            print("警告：内存使用接近极限")


def clear_memory_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def log_losses(iter_num, train_loss, val_loss, lr, mfu, timestamp):
    if master_process:
        with open(os.path.join(out_dir, loss_log_file), 'a') as f:
            f.write(f'{iter_num},{train_loss:.6f},{val_loss:.6f},{timestamp},{lr:.8f},{mfu:.4f}\n')


tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loader
data_dir = os.path.join('data', dataset)


def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init
iter_num = 0
best_val_loss = 1e9

# model init
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# initialize GradScaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
print(f"Configuring {optimizer_type} optimizer")

optimizer_kwargs = {}
if optimizer_type in ['AdamSN', 'AdamSNSM']:
    optimizer_kwargs['subset_size'] = subset_size
    optimizer_kwargs['correct_bias'] = correct_bias
    
if optimizer_type == 'AdamSNSM':
    optimizer_kwargs['rank'] = rank
    optimizer_kwargs['proj_type'] = proj_type
    optimizer_kwargs['update_proj_gap'] = update_proj_gap

optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type,
    optimizer_type=optimizer_type, **optimizer_kwargs
)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# compile model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# wrap in DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    # linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # after lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
print(f"\n{'='*60}")
print(f"Starting training with {optimizer_type}")
print(f"{'='*60}\n")

X, Y = get_batch('train')
val_X, val_Y = get_batch('val')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    # set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        current_time = time.time()
        log_losses(iter_num, losses['train'], losses['val'], lr, running_mfu, current_time)

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })
            
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))




    if iter_num == 0 and eval_only:
        break

    # forward backward update
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        if micro_step == gradient_accumulation_steps - 1:
            val_X, val_Y = get_batch('val')
        scaler.scale(loss).backward()
        
    # clip gradients
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
    # optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # timing
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu

        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if log_all_iterations:
            model.eval()
            with torch.no_grad():
                with ctx:
                    val_logits, val_loss_single = model(val_X, val_Y)
            model.train()

            current_time = time.time()
            log_losses(iter_num, lossf, val_loss_single.item(), lr, running_mfu, current_time)


    iter_num += 1
    local_iter_num += 1
    monitor_memory(iter_num)

    if iter_num % 500 == 0:
        clear_memory_cache()
        
    # termination
    if iter_num > max_iters:
        break

if master_process:
    final_losses = estimate_loss()
    current_time = time.time()
    log_losses(iter_num, final_losses['train'], final_losses['val'], lr, running_mfu, current_time)

if ddp:
    destroy_process_group()
