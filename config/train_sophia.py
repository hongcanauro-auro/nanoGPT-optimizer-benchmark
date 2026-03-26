import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

out_dir = 'out-sophia'
eval_interval = 500
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = 'nanogpt'
wandb_run_name = 'gpt2-124M-sophia'

dataset = 'openwebtext'
gradient_accumulation_steps = 6
batch_size = 6
block_size = 512

n_layer = 10
n_head = 10
n_embd = 760
dropout = 0.0

optimizer_name = 'sophiag'
learning_rate = 6e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 1.5e-5
warmup_iters = 2000
weight_decay = 0.2
beta1 = 0.965
beta2 = 0.99
grad_clip = 1.0
rho = 0.03
hess_interval = 10

compile = False


