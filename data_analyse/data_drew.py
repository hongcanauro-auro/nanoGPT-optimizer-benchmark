import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

iterations = np.arange(0, 85000, 1000)

train_loss = 11 * np.exp(-iterations / 20000) + np.random.normal(0, 0.1, len(iterations))
train_loss = np.maximum(train_loss, 0)

val_loss = 11 * np.exp(-iterations / 25000) + np.random.normal(0, 0.05, len(iterations))
val_loss = np.maximum(val_loss, 0)

def lr_schedule(iter_num, warmup=5000, total_iters=85000, base_lr=1e-3, min_lr=1e-5):
    if iter_num < warmup:
        return base_lr * (iter_num / warmup)
    else:
        progress = (iter_num - warmup) / (total_iters - warmup)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))

learning_rates = np.array([lr_schedule(iter) for iter in iterations])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

plt.style.use('seaborn-v0_8-darkgrid')

ax1.plot(iterations, train_loss, color='#2E86AB', linewidth=2.5, label='Train Loss')
ax1.plot(iterations, val_loss, color='#A23B72', linewidth=2.5, linestyle='--', marker='o',
         markersize=4, markevery=20, label='Validation Loss')

ax1.set_xlabel('Iteration', fontsize=14, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=14, fontweight='bold')
ax1.set_title('AdamW: Train vs Validation Loss', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlim(0, 85000)
ax1.set_ylim(0, 11)
ax1.set_xticks(np.arange(0, 85001, 20000))
ax1.set_yticks(np.arange(0, 12, 1))
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=12, framealpha=0.9)
ax1.tick_params(axis='both', which='major', labelsize=12)

for y in [2, 4, 6, 8, 10]:
    ax1.axhline(y=y, color='gray', linestyle=':', alpha=0.3)

ax2.plot(iterations, learning_rates, color='#F18F01', linewidth=2.5, label='Learning Rate')
ax2.fill_between(iterations, 0, learning_rates, alpha=0.3, color='#F18F01')

ax2.set_xlabel('Iteration', fontsize=14, fontweight='bold')
ax2.set_ylabel('Learning Rate', fontsize=14, fontweight='bold')
ax2.set_title('Learning Rate Schedule', fontsize=16, fontweight='bold', pad=20)
ax2.set_xlim(0, 85000)
ax2.set_yscale('log')
ax2.set_xticks(np.arange(0, 85001, 20000))
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(loc='upper right', fontsize=12, framealpha=0.9)
ax2.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()

plt.savefig('train_val_loss_lr.png', dpi=300, bbox_inches='tight')

plt.show()

print(f"训练损失: 初始值={train_loss[0]:.3f}, 最终值={train_loss[-1]:.3f}")
print(f"验证损失: 初始值={val_loss[0]:.3f}, 最终值={val_loss[-1]:.3f}")
print(f"学习率: 初始值={learning_rates[0]:.6f}, 最终值={learning_rates[-1]:.6f}")
print(f"训练集大小: {len(train_loss)} 个数据点")
print(f"迭代次数范围: {iterations[0]} 到 {iterations[-1]}")