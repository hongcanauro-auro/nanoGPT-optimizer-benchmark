import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_loss_log(filepath):
    df = pd.read_csv(filepath)
    return df


def plot_comparison(adamw_path='loss_log_adamw.csv',
                    adamsn_path='loss_log_sophia.csv',
                    adamsnsm_path='loss_log_adamsnsm.csv',
                    save_path='comparison_plot.png'):
    adamw_df = load_loss_log(adamw_path)
    adamsn_df = load_loss_log(adamsn_path)
    adamsnsm_df = load_loss_log(adamsnsm_path)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle('Optimizer Comparison: AdamW vs Sophia vs AdamSNSM', fontsize=16, fontweight='bold')

    ax1 = axes[0]
    if adamw_df is not None:
        ax1.plot(adamw_df['iteration'], adamw_df['train_loss'],
                 label='AdamW', alpha=0.7, linewidth=2)
    if adamsn_df is not None:
        ax1.plot(adamsn_df['iteration'], adamsn_df['train_loss'],
                 label='Sophia', alpha=0.7, linewidth=2)
    if adamsnsm_df is not None:
        ax1.plot(adamsnsm_df['iteration'], adamsnsm_df['train_loss'],
                 label='AdamSNSM', alpha=0.7, linewidth=2)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    if adamw_df is not None:
        ax2.plot(adamw_df['iteration'], adamw_df['val_loss'],
                 label='AdamW', alpha=0.7, linewidth=2)
    if adamsn_df is not None:
        ax2.plot(adamsn_df['iteration'], adamsn_df['val_loss'],
                 label='Sophia', alpha=0.7, linewidth=2)
    if adamsnsm_df is not None:
        ax2.plot(adamsnsm_df['iteration'], adamsnsm_df['val_loss'],
                 label='AdamSNSM', alpha=0.7, linewidth=2)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    if adamw_df is not None:
        ax3.plot(adamw_df['iteration'], adamw_df['lr'],
                 label='AdamW', alpha=0.7, linewidth=2)
    if adamsn_df is not None:
        ax3.plot(adamsn_df['iteration'], adamsn_df['lr'],
                 label='Sophia', alpha=0.7, linewidth=2)
    if adamsnsm_df is not None:
        ax3.plot(adamsnsm_df['iteration'], adamsnsm_df['lr'],
                 label='AdamSNSM', alpha=0.7, linewidth=2)

    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    for name, df in [('AdamW', adamw_df), ('Sophia', adamsn_df), ('AdamSNSM', adamsnsm_df)]:
        if df is not None and len(df) > 0:
            print(f"\n{name}:")
            print(f"  最终训练损失: {df['train_loss'].iloc[-1]:.4f}")
            print(f"  最终验证损失: {df['val_loss'].iloc[-1]:.4f}")
            print(f"  最低验证损失: {df['val_loss'].min():.4f}")

    if adamw_df is not None and adamsn_df is not None:
        print("\nSophia vs AdamW:")
        min_len = min(len(adamw_df), len(adamsn_df))
        if min_len > 0:
            adamw_final = adamw_df['val_loss'].iloc[min_len - 1]
            adamsn_final = adamsn_df['val_loss'].iloc[min_len - 1]
            improvement = (adamw_final - adamsn_final) / adamw_final * 100
            print(f"  验证损失相对改进: {improvement:+.2f}%")

    if adamw_df is not None and adamsnsm_df is not None:
        print("\nAdamSNSM vs AdamW:")
        min_len = min(len(adamw_df), len(adamsnsm_df))
        if min_len > 0:
            adamw_final = adamw_df['val_loss'].iloc[min_len - 1]
            adamsnsm_final = adamsnsm_df['val_loss'].iloc[min_len - 1]
            improvement = (adamw_final - adamsnsm_final) / adamw_final * 100
            print(f"  验证损失相对改进: {improvement:+.2f}%")

    print("\n" + "=" * 60)

    return fig


def plot_memory_comparison():
    n_embd = 760
    n_layer = 10

    params_per_layer = (
            3 * n_embd * n_embd +
            n_embd * n_embd +
            n_embd * 4 * n_embd +
            4 * n_embd * n_embd
    )
    total_linear_params = params_per_layer * n_layer

    adamw_memory = 2.0
    adamsn_memory = 1.0 + np.sqrt(n_embd) / n_embd
    adamsnsm_memory = 128 / n_embd + np.sqrt(n_embd) / n_embd

    optimizers = ['AdamW', 'Sophia', 'AdamSNSM']
    memory_mult = [adamw_memory, adamsn_memory, adamsnsm_memory]
    memory_relative = [m / adamw_memory * 100 for m in memory_mult]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars1 = ax1.bar(optimizers, memory_mult, color=colors, alpha=0.7)
    ax1.set_ylabel('Optimizer State Memory (×Parameters)')
    ax1.set_title('Absolute Memory Usage')
    ax1.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}×',
                 ha='center', va='bottom', fontweight='bold')

    bars2 = ax2.bar(optimizers, memory_relative, color=colors, alpha=0.7)
    ax2.set_ylabel('Memory Usage (%)')
    ax2.set_title('Relative Memory Usage (vs AdamW)')
    ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='AdamW Baseline')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        savings = 100 - height
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}%\n(saves {savings:.1f}%)',
                 ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Theoretical Optimizer Memory Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('memory_comparison.png', dpi=300, bbox_inches='tight')
    print("\n内存对比图已保存到: memory_comparison.png")

    return fig


if __name__ == '__main__':
    plot_comparison()
    plot_memory_comparison()
    print("  - comparison_plot.png: 训练过程对比")
    print("  - memory_comparison.png: 内存占用对比")