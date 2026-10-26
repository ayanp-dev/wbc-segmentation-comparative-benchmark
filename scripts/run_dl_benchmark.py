import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.deep_learning.dataset import create_dl_splits, get_dataloaders
from src.deep_learning.transfer_model import build_transfer_model
from src.deep_learning.train_eval import (
    train_model,
    evaluate_model,
    measure_inference_latency
)
from src.utils.visualization import (
    plot_confusion_matrices,
    plot_accuracy_vs_latency
)


def generate_benchmark_table_markdown(all_results: dict, output_path: str = 'results/benchmark_table.md') -> str:
    """Generate Markdown comparison table across classical and deep learning models."""
    header = (
        "| Architecture / Model | Regime / Features | Test Accuracy (%) | Macro F1 | "
        "Sensitivity (Eosinophil) | Specificity (Eosinophil) | Latency (ms/img) |\n"
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n"
    )

    rows = []
    for model_key, res in all_results.items():
        name = res['name']
        regime = res['regime']
        acc = res['test_accuracy'] * 100.0
        f1 = res['macro_f1']
        eos_sens = res['per_class']['eosinophil']['sensitivity']
        eos_spec = res['per_class']['eosinophil']['specificity']
        latency = res['latency_ms']

        rows.append(
            f"| {name} | {regime} | {acc:.2f}% | {f1:.4f} | {eos_sens:.4f} | {eos_spec:.4f} | {latency:.2f} |"
        )

    table_md = header + "\n".join(rows) + "\n"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(table_md)

    return table_md


def main():
    print("=== Deep Learning Benchmark Execution ===")
    os.makedirs('results/figures', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing DL benchmark on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    splits = create_dl_splits('bloodcells_4classes_500', random_state=42, low_data_fraction=0.15)
    class_names = splits['classes']

    print(f"Data Partitions: Train Full={len(splits['train_full'])}, Train Low (15%)={len(splits['train_low'])}, "
          f"Val={len(splits['val'])}, Test={len(splits['test'])}")

    regimes = [
        ('low_data_baseline', 'ResNet-18 (Low-Data 15% Baseline)', 'Low-Data 15% (No Aug)'),
        ('low_data_augmented', 'ResNet-18 (Low-Data 15% + Synthetic Aug)', 'Low-Data 15% + Domain Aug'),
        ('full_data', 'ResNet-18 (Full-Data 100% Upper Bound)', 'Full-Data 100%')
    ]

    dl_results = {}
    confusion_dict = {}

    for regime_key, display_name, regime_label in regimes:
        print(f"\n==========================================")
        print(f"Running Regime: {display_name}")
        print(f"==========================================")

        train_loader, val_loader, test_loader = get_dataloaders(
            splits=splits,
            regime=regime_key,
            batch_size=32,
            num_workers=4,
            img_size=(224, 224)
        )

        model = build_transfer_model('resnet18', num_classes=len(class_names), pretrained=True)

        print(f"Training {display_name}...")
        train_start = time.perf_counter()
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=20,
            lr=1e-4,
            device=device,
            class_names=class_names,
            patience=6
        )
        train_duration = time.perf_counter() - train_start

        # Evaluation on test set
        criterion = torch.nn.CrossEntropyLoss()
        test_loss, test_metrics, _, _ = evaluate_model(
            trained_model, test_loader, criterion, device, class_names
        )

        # Single-image latency
        latency_ms = measure_inference_latency(trained_model, device, input_shape=(1, 3, 224, 224), num_runs=100)

        print(f"Test Accuracy: {test_metrics['overall_accuracy']*100:.2f}%, Macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"Inference Latency: {latency_ms:.2f} ms/image, Training Duration: {train_duration:.1f} s")

        for c in class_names:
            c_m = test_metrics['per_class'][c]
            print(f"  {c:>12}: Sensitivity={c_m['sensitivity']:.4f}, Specificity={c_m['specificity']:.4f}")

        dl_results[regime_key] = {
            'name': 'ResNet-18',
            'regime': regime_label,
            'category': 'Deep Learning',
            'train_time_sec': train_duration,
            'latency_ms': latency_ms,
            'test_accuracy': test_metrics['overall_accuracy'],
            'macro_f1': test_metrics['macro_f1'],
            'per_class': test_metrics['per_class'],
            'confusion_matrix': test_metrics['confusion_matrix']
        }
        confusion_dict[display_name] = test_metrics

    # Save combined confusion matrices across Classical and Deep Learning
    all_confusion_dict = {}
    if os.path.exists('results/classical_metrics.json'):
        with open('results/classical_metrics.json', 'r') as f:
            c_data = json.load(f)
        if 'SVM' in c_data:
            all_confusion_dict['Classical SVM (Raw 51)'] = c_data['SVM']['test_metrics']
        if 'Random Forest' in c_data:
            all_confusion_dict['Classical Random Forest'] = c_data['Random Forest']['test_metrics']
    for k, v in confusion_dict.items():
        all_confusion_dict[k] = v

    plot_confusion_matrices(
        all_confusion_dict,
        class_names,
        'results/figures/confusion_matrices_all.png'
    )
    print("Saved combined confusion matrices to results/figures/confusion_matrices_all.png.")

    # Merge Classical and Deep Learning benchmarks
    combined_benchmark = {}

    if os.path.exists('results/classical_metrics.json'):
        with open('results/classical_metrics.json', 'r') as f:
            classical_data = json.load(f)

        for m_name, m_res in classical_data.items():
            test_m = m_res['test_metrics']
            combined_benchmark[m_name] = {
                'name': m_res['model_name'],
                'regime': f"Classical ({'PCA 95%' if m_res['use_pca'] else 'Raw 51 Feats'})",
                'category': 'Classical',
                'train_time_sec': m_res['train_time_sec'],
                'latency_ms': m_res['total_inference_latency_ms'],
                'test_accuracy': test_m['overall_accuracy'],
                'macro_f1': test_m['macro_f1'],
                'per_class': test_m['per_class'],
                'confusion_matrix': test_m['confusion_matrix']
            }

    for regime_key, res in dl_results.items():
        combined_benchmark[f"DL_{regime_key}"] = res

    # Write combined metrics summary
    with open('results/metrics_summary.json', 'w') as f:
        json.dump(combined_benchmark, f, indent=2)
    print("\nComplete metrics written to results/metrics_summary.json.")

    # Generate benchmark Markdown table
    table_md = generate_benchmark_table_markdown(combined_benchmark, 'results/benchmark_table.md')
    print("\nBenchmark Comparison Table:")
    print(table_md)

    # Generate Accuracy vs Inference Latency
    scatter_data = [
        {
            'name': res['name'] if res['category'] == 'Deep Learning' else res['name'],
            'accuracy': res['test_accuracy'],
            'latency_ms': res['latency_ms'],
            'category': res['category']
        }
        for res in combined_benchmark.values()
    ]
    plot_accuracy_vs_latency(scatter_data, 'results/figures/accuracy_vs_latency.png')
    print("Saved accuracy vs latency plot to results/figures/accuracy_vs_latency.png.")


if __name__ == '__main__':
    main()
