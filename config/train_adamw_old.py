out_dir = 'out-adamw'
eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

wandb_log = True
wandb_project = 'nanogpt'
wandb_run_name = 'gpt2-124M-adamw'

dataset = 'openwebtext'
gradient_accumulation_steps = 40  # 8GB显存需要更多累积
batch_size = 12  # 适合8GB显存
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

learning_rate = 6e-4
max_iters = 20000  # 先跑2万次测试
lr_decay_iters = 20000
min_lr = 6e-5
warmup_iters = 2000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

compile = False  # Windows必须False
