import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

def smooth_curve(data, window_size=11):
    if len(data) < window_size:
        return data

    alpha = 2.0 / (window_size + 1.0)

    half_window = window_size // 2

    start_reflection = data[half_window:0:-1]
    end_reflection = data[-2:-(half_window+2):-1]

    if len(start_reflection) < half_window:
        start_reflection = np.concatenate([start_reflection, data[0:1] * (half_window - len(start_reflection))])
    if len(end_reflection) < half_window:
        end_reflection = np.concatenate([end_reflection, data[-1:] * (half_window - len(end_reflection))])

    extended_data = np.concatenate([start_reflection, data, end_reflection])

    ewma_smoothed = np.zeros_like(extended_data, dtype=float)
    ewma_smoothed[0] = extended_data[0]

    for i in range(1, len(extended_data)):
        ewma_smoothed[i] = alpha * extended_data[i] + (1 - alpha) * ewma_smoothed[i-1]

    window = np.ones(window_size) / window_size

    final_smoothed = np.convolve(ewma_smoothed, window, mode='same')

    start_idx = half_window
    end_idx = start_idx + len(data)
    result = final_smoothed[start_idx:end_idx]

    return result

def plot_loss_and_lr_from_log():
    df = pd.read_csv('loss_log.csv')

    iterations = df['iteration']
    train_losses = df['train_loss']
    val_losses = df['val_loss']
    learning_rates = df['lr']

    train_losses_smooth = smooth_curve(train_losses, window_size=7)
    val_losses_smooth = smooth_curve(val_losses, window_size=7)

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)

    plt.plot(iterations, train_losses_smooth, label='Train Loss',
             linewidth=2, color='blue', alpha=0.8)
    plt.plot(iterations, val_losses_smooth, label='Validation Loss',
             linewidth=2, color='orange', alpha=0.8)

    marker_iterations = iterations[::50]
    marker_train_loss = train_losses_smooth[::50]
    marker_val_loss = val_losses_smooth[::50]

    plt.scatter(marker_iterations, marker_train_loss, color='blue', s=60,
                marker='o', zorder=5, alpha=0.7, edgecolors='white', linewidth=1)
    plt.scatter(marker_iterations, marker_val_loss, color='orange', s=60,
                marker='s', zorder=5, alpha=0.7, edgecolors='white', linewidth=1)

    min_val_loss_idx = np.argmin(val_losses_smooth)
    min_val_loss_iter = iterations[min_val_loss_idx]
    min_val_loss = val_losses_smooth[min_val_loss_idx]

    min_train_loss_idx = np.argmin(train_losses_smooth)
    min_train_loss_iter = iterations[min_train_loss_idx]
    min_train_loss = train_losses_smooth[min_train_loss_idx]

    plt.scatter(min_train_loss_iter, min_train_loss, color='darkblue', s=120,
                marker='*', zorder=6, label=f'Min Train Loss: {min_train_loss:.4f}',
                edgecolors='white', linewidth=1.5)
    plt.scatter(min_val_loss_iter, min_val_loss, color='darkorange', s=120,
                marker='*', zorder=6, label=f'Min Val Loss: {min_val_loss:.4f}',
                edgecolors='white', linewidth=1.5)

    plt.annotate(f'Min Train: {min_train_loss:.4f}',
                 xy=(min_train_loss_iter, min_train_loss),
                 xytext=(min_train_loss_iter + 500, min_train_loss + 0.8),
                 textcoords='data',
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle='arc3,rad=0.1',
                                 color='darkblue',
                                 lw=1.5),
                 fontsize=9,
                 color='darkblue',
                 fontweight='bold')

    plt.annotate(f'Min Val: {min_val_loss:.4f}',
                 xy=(min_val_loss_iter, min_val_loss),
                 xytext=(min_val_loss_iter + 500, min_val_loss - 0.8),
                 textcoords='data',
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle='arc3,rad=-0.1',
                                 color='darkorange',
                                 lw=1.5),
                 fontsize=9,
                 color='darkorange',
                 fontweight='bold')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)

    max_lr_idx = learning_rates.idxmax()
    max_lr_iter = iterations[max_lr_idx]
    max_lr = learning_rates[max_lr_idx]

    plt.plot(iterations, learning_rates, label='Learning Rate', color='green',
             linewidth=2, alpha=0.8)

    marker_lr = learning_rates[::50]
    plt.scatter(marker_iterations, marker_lr, color='green', s=60,
                marker='^', zorder=5, alpha=0.7, edgecolors='white', linewidth=1)

    plt.scatter(max_lr_iter, max_lr, color='darkgreen', s=120,
                marker='*', zorder=6, label=f'Max LR: {max_lr:.6f}',
                edgecolors='white', linewidth=1.5)

    plt.annotate(f'Max: {max_lr:.6f}',
                 xy=(max_lr_iter, max_lr),
                 xytext=(max_lr_iter + 500, max_lr * 0.6),
                 textcoords='data',
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle='arc3,rad=0.1',
                                 color='darkgreen',
                                 lw=1.5),
                 fontsize=9,
                 color='darkgreen',
                 fontweight='bold')

    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()

    plt.savefig('loss_lr_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Training Summary:")
    print(f"Final Train Loss: {train_losses_smooth[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses_smooth[-1]:.4f}")
    print(f"Minimum Train Loss: {min_train_loss:.4f} at iteration {min_train_loss_iter}")
    print(f"Minimum Validation Loss: {min_val_loss:.4f} at iteration {min_val_loss_iter}")
    print(f"Final Learning Rate: {learning_rates.iloc[-1]:.6f}")
    print(f"Maximum Learning Rate: {max_lr:.6f} at iteration {max_lr_iter}")


def plot_loss_only_smoothed():
    df = pd.read_csv('loss_log.csv')

    iterations = df['iteration']
    train_losses = df['train_loss']
    val_losses = df['val_loss']

    train_losses_smooth = smooth_curve(train_losses, window_size=15)
    val_losses_smooth = smooth_curve(val_losses, window_size=15)

    plt.figure(figsize=(10, 6))

    plt.plot(iterations, train_losses_smooth, label='Train Loss', linewidth=2)
    plt.plot(iterations, val_losses_smooth, label='Validation Loss', linewidth=2)

    marker_iterations = iterations[::500]
    marker_train = train_losses_smooth[::500]
    marker_val = val_losses_smooth[::500]

    plt.scatter(marker_iterations, marker_train, color='blue', s=50,
                marker='o', zorder=5, alpha=0.8)
    plt.scatter(marker_iterations, marker_val, color='orange', s=50,
                marker='s', zorder=5, alpha=0.8)

    min_val_idx = np.argmin(val_losses_smooth)
    min_val_iter = iterations[min_val_idx]
    min_val_loss = val_losses_smooth[min_val_idx]

    min_train_idx = np.argmin(train_losses_smooth)
    min_train_iter = iterations[min_train_idx]
    min_train_loss = train_losses_smooth[min_train_idx]

    plt.scatter(min_val_iter, min_val_loss, color='red', s=120, zorder=5, marker='*', label='Min Val Loss')
    plt.scatter(min_train_iter, min_train_loss, color='darkblue', s=120, zorder=5, marker='*', label='Min Train Loss')

    plt.annotate(f'Min Val: {min_val_loss:.4f}',
                 xy=(min_val_iter, min_val_loss),
                 xytext=(min_val_iter + 1000, min_val_loss + 0.5),
                 textcoords='data',
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle='arc3,rad=0.1',
                                 color='red',
                                 lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                 fontsize=10,
                 ha='left')

    plt.annotate(f'Min Train: {min_train_loss:.4f}',
                 xy=(min_train_iter, min_train_loss),
                 xytext=(min_train_iter + 1000, min_train_loss - 0.5),
                 textcoords='data',
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle='arc3,rad=-0.1',
                                 color='darkblue',
                                 lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                 fontsize=10,
                 ha='left')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_curve.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    plot_loss_and_lr_from_log()
    plot_loss_only_smoothed()