import os
from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_segmentation_samples(
    class_samples: Dict[str, Dict[str, np.ndarray]],
    output_path: str
) -> None:
    """
    Generate multi-panel visual comparison: Original, Cytoplasm Mask, and Nucleus Mask
    across all 4 leukocyte classes.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    classes = list(class_samples.keys())
    fig, axes = plt.subplots(len(classes), 3, figsize=(10, 3.2 * len(classes)))

    for i, c_name in enumerate(classes):
        data = class_samples[c_name]
        orig = data['original']
        cyto = data['cytoplasm_mask']
        nuc = data['nucleus_mask']

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"{c_name.capitalize()} - Original", fontsize=11)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(cyto, cmap='gray')
        axes[i, 1].set_title(f"{c_name.capitalize()} - Cytoplasm Mask", fontsize=11)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(nuc, cmap='gray')
        axes[i, 2].set_title(f"{c_name.capitalize()} - Nucleus Mask", fontsize=11)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_correlation_matrix(
    features_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Generate and save the 51-feature correlation matrix heatmap.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    corr = features_df.corr().values

    fig, ax = plt.subplots(figsize=(14, 12))
    cax = ax.imshow(corr, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(features_df.columns)))
    ax.set_yticks(range(len(features_df.columns)))
    ax.set_xticklabels(features_df.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(features_df.columns, fontsize=7)
    ax.set_title("51 Handcrafted Features Correlation Matrix", fontsize=14, pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pca_cumulative_variance(
    explained_variance_ratio: np.ndarray,
    target_variance: float = 0.95,
    output_path: str = "results/figures/pca_cumulative_variance.png"
) -> int:
    """
    Plot cumulative explained variance curve and mark the threshold cutoff.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cumsum = np.cumsum(explained_variance_ratio)
    cutoff_idx = int(np.searchsorted(cumsum, target_variance)) + 1

    fig, ax = plt.subplots(figsize=(8, 5))
    components = np.arange(1, len(explained_variance_ratio) + 1)
    ax.plot(components, cumsum * 100, marker='o', markersize=4, linestyle='-', color='#1f77b4', label='Cumulative Variance')
    ax.axhline(y=target_variance * 100, color='r', linestyle='--', label=f'{int(target_variance*100)}% Threshold')
    ax.axvline(x=cutoff_idx, color='g', linestyle=':', label=f'{cutoff_idx} Components')

    ax.set_xlabel('Principal Components', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
    ax.set_title('PCA Cumulative Explained Variance Curve', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return cutoff_idx


def plot_confusion_matrices(
    confusion_matrices: Dict[str, Dict[str, Any]],
    class_names: List[str],
    output_path: str
) -> None:
    """
    Plot grid of confusion matrices across evaluated models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_models = len(confusion_matrices)
    cols = min(2, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5.5 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (m_name, m_info) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        cm = np.array(m_info['confusion_matrix'])
        cax = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(f"{m_name}\nAcc: {m_info['overall_accuracy']*100:.2f}% | F1: {m_info['macro_f1']:.3f}", fontsize=11)
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=35, ha='right', fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=10)

    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_vs_latency(
    benchmark_data: List[Dict[str, Any]],
    output_path: str = "results/figures/accuracy_vs_latency.png"
) -> None:
    """
    Generate comparative scatter plot of Test Accuracy vs Inference Latency.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    for item in benchmark_data:
        name = item['name']
        acc = item['accuracy'] * 100.0
        lat = item['latency_ms']
        category = item.get('category', 'Classical')

        color = '#1f77b4' if category == 'Classical' else '#d62728'
        marker = 's' if category == 'Classical' else 'o'

        ax.scatter(lat, acc, s=120, color=color, marker=marker, edgecolors='black', linewidth=1.2, zorder=5)
        ax.annotate(name, (lat, acc), textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_xlabel('Inference Latency (ms / image)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy vs. Single-Image Inference Latency', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
