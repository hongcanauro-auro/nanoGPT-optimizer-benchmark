import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

out_dir = 'out-subset-norm'
eval_interval = 500
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = 'nanogpt'
wandb_run_name = 'gpt2-124M-subset-norm'

dataset = 'openwebtext'
gradient_accumulation_steps = 20
batch_size = 6
block_size = 512

n_layer = 10
n_head = 10
n_embd = 760
dropout = 0.0

optimizer_type = 'AdamSNSM'

learning_rate = 6e-3
max_iters = 20000
lr_decay_iters = 20000
min_lr = 6e-4
warmup_iters = 2000

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

subset_size = 'heuristics'
correct_bias = True

rank = 128
proj_type = 'svd'
update_proj_gap = 200

compile = False
