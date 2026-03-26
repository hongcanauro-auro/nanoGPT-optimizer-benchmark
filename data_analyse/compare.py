import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def smooth_curve(data, window_size=11):
    if len(data) < window_size:
        return data

    alpha = 2.0 / (window_size + 1.0)
    half_window = window_size // 2

    start_reflection = data[half_window:0:-1]
    end_reflection = data[-2:-(half_window + 2):-1]

    if len(start_reflection) < half_window:
        start_reflection = np.concatenate([start_reflection, data[0:1] * (half_window - len(start_reflection))])
    if len(end_reflection) < half_window:
        end_reflection = np.concatenate([end_reflection, data[-1:] * (half_window - len(end_reflection))])

    extended_data = np.concatenate([start_reflection, data, end_reflection])

    ewma_smoothed = np.zeros_like(extended_data, dtype=float)
    ewma_smoothed[0] = extended_data[0]

    for i in range(1, len(extended_data)):
        ewma_smoothed[i] = alpha * extended_data[i] + (1 - alpha) * ewma_smoothed[i - 1]

    window = np.ones(window_size) / window_size
    final_smoothed = np.convolve(ewma_smoothed, window, mode='same')

    start_idx = half_window
    end_idx = start_idx + len(data)
    result = final_smoothed[start_idx:end_idx]

    return result


def plot_three_experiments():
    csv_files = ['loss_log_adamw.csv', 'loss_log_sophia.csv', 'loss_log_adamsnsm.csv']
    exp_names = ['Adamw', 'Sophia', 'AdamwSNSM']

    train_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    val_colors = ['#aec7e8', '#ffbb78', '#98df8a']

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)

    all_min_points = []

    for idx, (csv_file, exp_name) in enumerate(zip(csv_files, exp_names)):
        df = pd.read_csv(csv_file)

        iterations = df['iteration']
        train_losses = df['train_loss']
        val_losses = df['val_loss']

        train_losses_smooth = smooth_curve(train_losses, window_size=15)
        val_losses_smooth = smooth_curve(val_losses, window_size=15)

        plt.plot(iterations, train_losses_smooth,
                 label=f'{exp_name} Train',
                 linewidth=2,
                 color=train_colors[idx],
                 alpha=0.8)
        plt.plot(iterations, val_losses_smooth,
                 label=f'{exp_name} Val',
                 linewidth=2,
                 color=val_colors[idx],
                 alpha=0.8,
                 linestyle='--')

        marker_iterations = iterations[::100]
        marker_train_loss = train_losses_smooth[::100]
        marker_val_loss = val_losses_smooth[::100]

        plt.scatter(marker_iterations, marker_train_loss,
                    color=train_colors[idx], s=40,
                    marker='o', zorder=5, alpha=0.6,
                    edgecolors='white', linewidth=0.5)
        plt.scatter(marker_iterations, marker_val_loss,
                    color=val_colors[idx], s=40,
                    marker='s', zorder=5, alpha=0.6,
                    edgecolors='white', linewidth=0.5)

        min_val_loss_idx = np.argmin(val_losses_smooth)
        min_val_loss_iter = iterations[min_val_loss_idx]
        min_val_loss = val_losses_smooth[min_val_loss_idx]

        min_train_loss_idx = np.argmin(train_losses_smooth)
        min_train_loss_iter = iterations[min_train_loss_idx]
        min_train_loss = train_losses_smooth[min_train_loss_idx]

        plt.scatter(min_train_loss_iter, min_train_loss,
                    color=train_colors[idx], s=150,
                    marker='*', zorder=6,
                    edgecolors='white', linewidth=1.5)
        plt.scatter(min_val_loss_iter, min_val_loss,
                    color=val_colors[idx], s=150,
                    marker='*', zorder=6,
                    edgecolors='white', linewidth=1.5)

        all_min_points.append({
            'type': 'train',
            'exp_idx': idx,
            'iter': min_train_loss_iter,
            'loss': min_train_loss,
            'color': train_colors[idx],
            'label': f'Exp{idx + 1} Train: {min_train_loss:.4f}'
        })
        all_min_points.append({
            'type': 'val',
            'exp_idx': idx,
            'iter': min_val_loss_iter,
            'loss': min_val_loss,
            'color': val_colors[idx],
            'label': f'Exp{idx + 1} Val: {min_val_loss:.4f}'
        })

    all_min_points.sort(key=lambda x: x['iter'])

    offset_patterns = [
        (800, 1.0), (800, -1.0), (-800, 1.2),
        (-800, -1.2), (1200, 0.5), (-1200, -0.5)
    ]

    for i, point in enumerate(all_min_points):
        offset_x, offset_y = offset_patterns[i % len(offset_patterns)]

        if i > 0:
            prev_point = all_min_points[i - 1]
            if abs(point['iter'] - prev_point['iter']) < 500:
                offset_y *= -1

        plt.annotate(point['label'],
                     xy=(point['iter'], point['loss']),
                     xytext=(point['iter'] + offset_x, point['loss'] + offset_y),
                     textcoords='data',
                     arrowprops=dict(arrowstyle='->',
                                     connectionstyle='arc3,rad=0.2',
                                     color=point['color'],
                                     lw=1.2),
                     fontsize=8,
                     color=point['color'],
                     fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white',
                               edgecolor=point['color'],
                               alpha=0.8))

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Curves - Three Experiments Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)

    all_max_lr_points = []

    for idx, (csv_file, exp_name) in enumerate(zip(csv_files, exp_names)):
        df = pd.read_csv(csv_file)

        iterations = df['iteration']
        learning_rates = df['lr']

        plt.plot(iterations, learning_rates,
                 label=f'{exp_name}',
                 color=train_colors[idx],
                 linewidth=2,
                 alpha=0.8)

        marker_iterations = iterations[::100]
        marker_lr = learning_rates[::100]
        plt.scatter(marker_iterations, marker_lr,
                    color=train_colors[idx], s=40,
                    marker='^', zorder=5, alpha=0.6,
                    edgecolors='white', linewidth=0.5)

        max_lr_idx = learning_rates.idxmax()
        max_lr_iter = iterations[max_lr_idx]
        max_lr = learning_rates[max_lr_idx]

        plt.scatter(max_lr_iter, max_lr,
                    color=train_colors[idx], s=150,
                    marker='*', zorder=6,
                    edgecolors='white', linewidth=1.5)

        all_max_lr_points.append({
            'exp_idx': idx,
            'iter': max_lr_iter,
            'lr': max_lr,
            'color': train_colors[idx],
            'label': f'Exp{idx + 1}: {max_lr:.6f}'
        })

    all_max_lr_points.sort(key=lambda x: x['iter'])

    lr_offset_patterns = [
        (800, 1.5), (-800, 1.8), (1000, 0.8)
    ]

    for i, point in enumerate(all_max_lr_points):
        offset_x, offset_y_mult = lr_offset_patterns[i % len(lr_offset_patterns)]

        plt.annotate(point['label'],
                     xy=(point['iter'], point['lr']),
                     xytext=(point['iter'] + offset_x, point['lr'] * offset_y_mult),
                     textcoords='data',
                     arrowprops=dict(arrowstyle='->',
                                     connectionstyle='arc3,rad=0.15',
                                     color=point['color'],
                                     lw=1.2),
                     fontsize=8,
                     color=point['color'],
                     fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white',
                               edgecolor=point['color'],
                               alpha=0.8))

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule - Three Experiments Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('three_experiments_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("Three Experiments Summary")
    print("=" * 60)

    for idx, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        train_losses_smooth = smooth_curve(df['train_loss'], window_size=15)
        val_losses_smooth = smooth_curve(df['val_loss'], window_size=15)

        print(f"\n{exp_names[idx]}:")
        print(f"  Final Train Loss: {train_losses_smooth[-1]:.4f}")
        print(f"  Final Val Loss: {val_losses_smooth[-1]:.4f}")
        print(f"  Min Train Loss: {np.min(train_losses_smooth):.4f}")
        print(f"  Min Val Loss: {np.min(val_losses_smooth):.4f}")
        print(f"  Final LR: {df['lr'].iloc[-1]:.6f}")
        print(f"  Max LR: {df['lr'].max():.6f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    plot_three_experiments()