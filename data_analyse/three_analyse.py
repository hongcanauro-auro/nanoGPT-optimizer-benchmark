import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def load_loss_log(filepath):
    df = pd.read_csv(filepath)
    return df


def calculate_metrics(df, optimizer_name):
    if df is None or len(df) == 0:
        return None

    metrics = {
        'Optimizer': optimizer_name,
        'Final Train Loss': df['train_loss'].iloc[-1],
        'Final Val Loss': df['val_loss'].iloc[-1],
        'Min Val Loss': df['val_loss'].min(),
        'Train Loss Std': df['train_loss'].std(),
        'Val Loss Std': df['val_loss'].std(),
        'Avg Learning Rate': df['lr'].mean(),
        'Final Learning Rate': df['lr'].iloc[-1],
        'Iterations': len(df)
    }

    metrics['Overfitting'] = metrics['Final Train Loss'] - metrics['Min Val Loss']

    if len(df) > 1:
        x_train = np.arange(len(df))
        y_train = df['train_loss'].values
        slope_train, _, _, _, _ = stats.linregress(x_train, y_train)
        metrics['Train Loss Decline Rate'] = -slope_train

        x_val = np.arange(len(df))
        y_val = df['val_loss'].values
        slope_val, _, _, _, _ = stats.linregress(x_val, y_val)
        metrics['Val Loss Decline Rate'] = -slope_val

        metrics['Overall Decline Rate'] = (metrics['Train Loss Decline Rate'] + metrics['Val Loss Decline Rate']) / 2
    else:
        metrics['Train Loss Decline Rate'] = 0
        metrics['Val Loss Decline Rate'] = 0
        metrics['Overall Decline Rate'] = 0

    if metrics['Avg Learning Rate'] > 0:
        metrics['Train Stability'] = metrics['Train Loss Std'] / metrics['Final Train Loss']
        metrics['Val Stability'] = metrics['Val Loss Std'] / metrics['Final Val Loss']
    else:
        metrics['Train Stability'] = 0
        metrics['Val Stability'] = 0

    return metrics


def analyze_optimizer_comparison(adamw_path='loss_log_adamw.csv',
                                 sophia_path='loss_log_sophia.csv',
                                 adamsnsm_path='loss_log_adamsnsm.csv',
                                 save_csv='optimizer_comparison.csv',
                                 save_plot='metrics_comparison.png'):

    adamw_df = load_loss_log(adamw_path)
    sophia_df = load_loss_log(sophia_path)
    adamsnsm_df = load_loss_log(adamsnsm_path)

    metrics_list = []

    metrics_adamw = calculate_metrics(adamw_df, 'AdamW')
    if metrics_adamw:
        metrics_list.append(metrics_adamw)

    metrics_sophia = calculate_metrics(sophia_df, 'Sophia')
    if metrics_sophia:
        metrics_list.append(metrics_sophia)

    metrics_adamsnsm = calculate_metrics(adamsnsm_df, 'AdamSNSM')
    if metrics_adamsnsm:
        metrics_list.append(metrics_adamsnsm)

    if not metrics_list:
        print("No data available for analysis")
        return None

    comparison_df = pd.DataFrame(metrics_list)

    pd.set_option('display.float_format', '{:.4f}'.format)

    print("=" * 80)
    print("Optimizer Performance Comparison Analysis")
    print("=" * 80)

    column_order = [
        'Optimizer', 'Iterations', 'Final Train Loss', 'Final Val Loss', 'Min Val Loss',
        'Overfitting', 'Train Loss Decline Rate', 'Val Loss Decline Rate', 'Overall Decline Rate',
        'Train Loss Std', 'Val Loss Std', 'Train Stability', 'Val Stability',
        'Avg Learning Rate', 'Final Learning Rate'
    ]

    column_order = [col for col in column_order if col in comparison_df.columns]
    comparison_df = comparison_df[column_order]

    print("\nDetailed Metrics Comparison:")
    print(comparison_df.to_string(index=False))

    if 'AdamW' in comparison_df['Optimizer'].values:
        adamw_row = comparison_df[comparison_df['Optimizer'] == 'AdamW'].iloc[0]

        print("\n" + "=" * 80)
        print("Relative Improvement Analysis (vs AdamW)")
        print("=" * 80)

        for idx, row in comparison_df.iterrows():
            if row['Optimizer'] != 'AdamW':
                optimizer_name = row['Optimizer']

                val_loss_improvement = (adamw_row['Final Val Loss'] - row['Final Val Loss']) / adamw_row[
                    'Final Val Loss'] * 100
                train_loss_improvement = (adamw_row['Final Train Loss'] - row['Final Train Loss']) / adamw_row[
                    'Final Train Loss'] * 100
                min_val_loss_improvement = (adamw_row['Min Val Loss'] - row['Min Val Loss']) / adamw_row[
                    'Min Val Loss'] * 100
                speed_improvement = (row['Overall Decline Rate'] - adamw_row['Overall Decline Rate']) / adamw_row[
                    'Overall Decline Rate'] * 100

                print(f"\n{optimizer_name} vs AdamW:")
                print(f"  - Final Val Loss Improvement: {val_loss_improvement:+.2f}%")
                print(f"  - Final Train Loss Improvement: {train_loss_improvement:+.2f}%")
                print(f"  - Min Val Loss Improvement: {min_val_loss_improvement:+.2f}%")
                print(f"  - Loss Decline Rate Improvement: {speed_improvement:+.2f}%")
                print(f"  - Overfitting: {'Better' if row['Overfitting'] < adamw_row['Overfitting'] else 'Worse'}")
                print(f"  - Training Stability: {'Better' if row['Train Stability'] < adamw_row['Train Stability'] else 'Worse'}")

    print("\n" + "=" * 80)
    print("Key Performance Metrics Ranking")
    print("=" * 80)

    key_metrics = ['Final Val Loss', 'Min Val Loss', 'Overall Decline Rate', 'Overfitting', 'Train Stability']

    for metric in key_metrics:
        if metric in comparison_df.columns:
            if metric in ['Final Val Loss', 'Min Val Loss', 'Overfitting', 'Train Stability']:
                sorted_df = comparison_df.sort_values(by=metric)
                ranking = "(Lower is Better)"
            else:
                sorted_df = comparison_df.sort_values(by=metric, ascending=False)
                ranking = "(Higher is Better)"

            print(f"\n{metric} {ranking}:")
            for i, (_, row) in enumerate(sorted_df.iterrows()):
                print(f"  Rank {i+1}: {row['Optimizer']} ({row[metric]:.4f})")

    comparison_df.to_csv(save_csv, index=False, encoding='utf-8-sig')
    print(f"\nComparison table saved to: {save_csv}")

    create_metrics_visualization(comparison_df, save_plot)

    return comparison_df


def create_metrics_visualization(comparison_df, save_path='metrics_comparison.png'):

    if len(comparison_df) == 0:
        return

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(comparison_df)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Optimizer Performance Metrics Comparison', fontsize=16, fontweight='bold')

    optimizers = comparison_df['Optimizer'].values

    ax1 = axes[0, 0]
    x = np.arange(len(optimizers))
    width = 0.35

    train_bars = ax1.bar(x - width / 2, comparison_df['Final Train Loss'],
                         width, label='Train Loss', color='#3498db', alpha=0.7)
    val_bars = ax1.bar(x + width / 2, comparison_df['Final Val Loss'],
                       width, label='Val Loss', color='#e74c3c', alpha=0.7)

    ax1.set_xlabel('Optimizer')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('Final Train/Val Loss Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(optimizers)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    for bars in [train_bars, val_bars]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}', ha='center', va='bottom')

    ax2 = axes[0, 1]
    bars = ax2.bar(optimizers, comparison_df['Overall Decline Rate'],
                   color=colors, alpha=0.7)

    ax2.set_xlabel('Optimizer')
    ax2.set_ylabel('Decline Rate')
    ax2.set_title('Average Loss Decline Rate Comparison')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.6f}', ha='center', va='bottom')

    ax3 = axes[1, 0]
    bars = ax3.bar(optimizers, comparison_df['Overfitting'],
                   color=colors, alpha=0.7)

    ax3.set_xlabel('Optimizer')
    ax3.set_ylabel('Overfitting')
    ax3.set_title('Overfitting Comparison (Train Loss - Min Val Loss)')
    ax3.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}', ha='center', va='bottom')

    ax4 = axes[1, 1]
    x = np.arange(len(optimizers))
    width = 0.35

    train_stability = ax4.bar(x - width / 2, comparison_df['Train Stability'],
                              width, label='Train Stability', color='#2ecc71', alpha=0.7)
    val_stability = ax4.bar(x + width / 2, comparison_df['Val Stability'],
                            width, label='Val Stability', color='#f39c12', alpha=0.7)

    ax4.set_xlabel('Optimizer')
    ax4.set_ylabel('Stability Coefficient')
    ax4.set_title('Train/Val Stability Comparison (Coefficient of Variation)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(optimizers)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    for bars in [train_stability, val_stability]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison chart saved to: {save_path}")

    return fig


def generate_analysis_report(comparison_df, save_path='optimizer_analysis_report.txt'):

    if comparison_df is None or len(comparison_df) == 0:
        return

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Optimizer Performance Analysis Report\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. Overview\n")
        f.write("-" * 40 + "\n")
        f.write(f"This comparison analyzed the performance of {len(comparison_df)} optimizers.\n\n")

        f.write("2. Detailed Metrics Comparison\n")
        f.write("-" * 40 + "\n")
        f.write(comparison_df.to_string(index=False) + "\n\n")

        f.write("3. Performance Ranking\n")
        f.write("-" * 40 + "\n")

        comparison_df_copy = comparison_df.copy()

        for col in comparison_df_copy.columns:
            if col not in ['Optimizer', 'Iterations']:
                if col in ['Final Train Loss', 'Final Val Loss', 'Min Val Loss',
                           'Overfitting', 'Train Loss Std', 'Val Loss Std',
                           'Train Stability', 'Val Stability']:
                    comparison_df_copy[col + '_norm'] = 1 - (
                                comparison_df_copy[col] - comparison_df_copy[col].min()) / (
                                                                    comparison_df_copy[col].max() - comparison_df_copy[
                                                                col].min() + 1e-10)
                elif col in ['Train Loss Decline Rate', 'Val Loss Decline Rate', 'Overall Decline Rate']:
                    comparison_df_copy[col + '_norm'] = (comparison_df_copy[col] - comparison_df_copy[col].min()) / (
                                comparison_df_copy[col].max() - comparison_df_copy[col].min() + 1e-10)

        weight_mapping = {
            'Final Val Loss_norm': 0.25,
            'Min Val Loss_norm': 0.20,
            'Overall Decline Rate_norm': 0.20,
            'Overfitting_norm': 0.15,
            'Train Stability_norm': 0.10,
            'Val Stability_norm': 0.10
        }

        comparison_df_copy['Overall Score'] = 0
        for col, weight in weight_mapping.items():
            if col in comparison_df_copy.columns:
                comparison_df_copy['Overall Score'] += comparison_df_copy[col] * weight

        comparison_df_copy = comparison_df_copy.sort_values(by='Overall Score', ascending=False)

        f.write("\nOverall Performance Ranking (Weighted Score):\n")
        for i, (_, row) in enumerate(comparison_df_copy.iterrows()):
            f.write(f"  Rank {i+1}: {row['Optimizer']} (Overall Score: {row['Overall Score']:.4f})\n")

        f.write("\n4. Optimization Recommendation\n")
        f.write("-" * 40 + "\n")

        best_optimizer = comparison_df_copy.iloc[0]['Optimizer']
        f.write(f"Based on the analysis results, {best_optimizer} optimizer is recommended, for the following reasons:\n")

        best_row = comparison_df[comparison_df['Optimizer'] == best_optimizer].iloc[0]

        advantages = []
        if best_row['Final Val Loss'] == comparison_df['Final Val Loss'].min():
            advantages.append("Lowest final validation loss")
        if best_row['Min Val Loss'] == comparison_df['Min Val Loss'].min():
            advantages.append("Best minimum validation loss achieved")
        if best_row['Overall Decline Rate'] == comparison_df['Overall Decline Rate'].max():
            advantages.append("Fastest loss decline rate")
        if best_row['Overfitting'] == comparison_df['Overfitting'].min():
            advantages.append("Minimum overfitting")

        for i, advantage in enumerate(advantages, 1):
            f.write(f"  {i}. {advantage}\n")

        f.write("\n5. Notes\n")
        f.write("-" * 40 + "\n")
        f.write("1. Analysis results are based on provided training log data\n")
        f.write("2. Optimizer performance may vary across different tasks and datasets\n")
        f.write("3. It is recommended to fine-tune based on specific tasks in practical applications\n")

    print(f"Detailed analysis report saved to: {save_path}")


if __name__ == '__main__':
    comparison_df = analyze_optimizer_comparison(
        adamw_path='loss_log_adamw.csv',
        sophia_path='loss_log_sophia.csv',
        adamsnsm_path='loss_log_adamsnsm.csv',
        save_csv='optimizer_comparison.csv',
        save_plot='metrics_comparison.png'
    )

    if comparison_df is not None:
        generate_analysis_report(comparison_df, 'optimizer_analysis_report.txt')

        print("  - optimizer_comparison.csv: Detailed metrics comparison table")
        print("  - metrics_comparison.png: Key metrics visualization chart")
        print("  - optimizer_analysis_report.txt: Detailed analysis report")