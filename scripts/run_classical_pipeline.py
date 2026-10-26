import os
import sys
import glob
import time
import json
from typing import Tuple, Dict, Any, List
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.classical.preprocessing import preprocess_image
from src.classical.segmentation import segment_wbc
from src.classical.features import extract_features, FEATURE_NAMES
from src.classical.classifiers import train_and_evaluate_model
from src.utils.visualization import (
    plot_segmentation_samples,
    plot_feature_correlation_matrix,
    plot_pca_cumulative_variance,
    plot_confusion_matrices
)


def load_dataset_metadata(data_dir: str = 'bloodcells_4classes_500') -> pd.DataFrame:
    """Collect image paths and class labels from the dataset directory structure."""
    classes = ['basophil', 'eosinophil', 'erythroblast', 'ig']
    records = []

    for c_name in sorted(classes):
        class_folder = os.path.join(data_dir, c_name)
        pattern = os.path.join(class_folder, '*.jpg')
        image_paths = sorted(glob.glob(pattern))
        for p in image_paths:
            records.append({'image_path': p, 'label': c_name})

    df = pd.DataFrame(records)
    print(f"Dataset loaded: {len(df)} images across {df['label'].nunique()} classes.")
    return df


def extract_all_features(df: pd.DataFrame, csv_path: str = 'results/features.csv') -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
    """Extract 51 handcrafted features for each image in the dataset or load from cached CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    sample_visuals: Dict[str, Dict[str, np.ndarray]] = {}
    found_classes = set()

    if os.path.exists(csv_path):
        print(f"Loading cached features from {csv_path}...")
        features_df = pd.read_csv(csv_path)

        # Extract sample visuals for each class for figure generation
        for c_name in ['basophil', 'eosinophil', 'erythroblast', 'ig']:
            sample_row = df[df['label'] == c_name].iloc[0]
            img = cv2.imread(sample_row['image_path'])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            preprocessed = preprocess_image(img_rgb)
            seg = segment_wbc(preprocessed['sftmap'], preprocessed['cmyk'][:, :, 1])
            sample_visuals[c_name] = {
                'original': preprocessed['resized_rgb'],
                'cytoplasm_mask': seg['cytoplasm_mask'],
                'nucleus_mask': seg['nucleus_mask']
            }
        mean_latency_ms = 0.0
        return features_df, mean_latency_ms, sample_visuals

    print("Extracting handcrafted features across all images...")
    feature_rows = []
    latencies = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting WBC features"):
        img_path = row['image_path']
        label = row['label']

        t0 = time.perf_counter()
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        preprocessed = preprocess_image(img_rgb)
        seg = segment_wbc(preprocessed['sftmap'], preprocessed['cmyk'][:, :, 1])
        feats = extract_features(preprocessed, seg)
        t_elapsed = time.perf_counter() - t0
        latencies.append(t_elapsed * 1000.0)

        if label not in found_classes:
            sample_visuals[label] = {
                'original': preprocessed['resized_rgb'],
                'cytoplasm_mask': seg['cytoplasm_mask'],
                'nucleus_mask': seg['nucleus_mask']
            }
            found_classes.add(label)

        row_dict = {name: val for name, val in zip(FEATURE_NAMES, feats)}
        row_dict['label'] = label
        row_dict['image_path'] = img_path
        feature_rows.append(row_dict)

    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(csv_path, index=False)
    mean_latency_ms = float(np.mean(latencies))
    print(f"Features saved to {csv_path}. Mean extraction latency: {mean_latency_ms:.2f} ms/image.")

    return features_df, mean_latency_ms, sample_visuals


def split_data(
    features_df: pd.DataFrame,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Execute stratified 60/20/20 train/val/test split."""
    X = features_df[FEATURE_NAMES].values
    y = features_df['label'].values
    class_names = sorted(list(np.unique(y)))

    # First split: 60% train, 40% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=random_state
    )

    # Second split: 20% val, 20% test (50/50 of temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )

    print(f"Stratified Split: Train={len(X_train)} (60%), Val={len(X_val)} (20%), Test={len(X_test)} (20%)")
    return X_train, y_train, X_val, y_val, X_test, y_test, class_names


def main():
    print("=== Classical Pipeline Execution ===")
    os.makedirs('results/figures', exist_ok=True)

    df_meta = load_dataset_metadata('bloodcells_4classes_500')
    features_df, extraction_latency_ms, sample_visuals = extract_all_features(df_meta, 'results/features.csv')

    # Multi-panel segmentation comparison across 4 classes
    print("Plotting segmentation sample comparisons...")
    plot_segmentation_samples(sample_visuals, 'results/figures/segmentation_visual_comparison.png')

    # 51-feature correlation matrix heatmap
    print("Generating feature correlation heatmap...")
    feats_only = features_df[FEATURE_NAMES]
    plot_feature_correlation_matrix(feats_only, 'results/figures/feature_correlation_heatmap.png')

    # Data splitting
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = split_data(features_df, random_state=42)

    # PCA Cumulative Explained Variance (fit on X_train)
    print("Plotting PCA cumulative explained variance...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    pca_fit = PCA().fit(X_train_scaled)
    cutoff_components = plot_pca_cumulative_variance(
        pca_fit.explained_variance_ratio_,
        target_variance=0.95,
        output_path='results/figures/pca_cumulative_variance.png'
    )
    print(f"PCA components required for 95% variance: {cutoff_components}")

    # Model evaluations
    models_to_train = [
        ('svm', False),
        ('svm', True),
        ('random_forest', False),
        ('random_forest', True)
    ]

    classical_results = {}
    confusion_matrices_dict = {}

    for model_type, use_pca in models_to_train:
        desc = f"{'PCA ' if use_pca else ''}{'SVM' if model_type == 'svm' else 'Random Forest'}"
        print(f"\n--- Training and Evaluating {desc} ---")
        res = train_and_evaluate_model(
            model_type=model_type,
            use_pca=use_pca,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            random_state=42
        )

        test_m = res['test_metrics']
        total_latency_ms = extraction_latency_ms + res['inference_latency_ms']
        print(f"{desc} -> Test Accuracy: {test_m['overall_accuracy']*100:.2f}%, Macro F1: {test_m['macro_f1']:.4f}")
        print(f"Inference Latency: {total_latency_ms:.2f} ms/image (Extraction: {extraction_latency_ms:.2f} ms, Predict: {res['inference_latency_ms']:.2f} ms)")

        # Per-class metrics
        for c in class_names:
            c_metrics = test_m['per_class'][c]
            print(f"  {c:>12}: Sensitivity={c_metrics['sensitivity']:.4f}, Specificity={c_metrics['specificity']:.4f}")

        clean_res = {
            'model_name': desc,
            'model_type': model_type,
            'use_pca': use_pca,
            'num_components': res['num_components'],
            'train_time_sec': res['train_time_sec'],
            'extraction_latency_ms': extraction_latency_ms,
            'prediction_latency_ms': res['inference_latency_ms'],
            'total_inference_latency_ms': total_latency_ms,
            'validation_metrics': res['validation_metrics'],
            'test_metrics': test_m
        }
        classical_results[desc] = clean_res
        confusion_matrices_dict[desc] = test_m

    # Classical part - Confusion matrices
    print("\nGenerating classical confusion matrices...")
    plot_confusion_matrices(
        confusion_matrices_dict,
        class_names,
        'results/figures/confusion_matrices_classical.png'
    )

    # Save summary
    with open('results/classical_metrics.json', 'w') as f:
        json.dump(classical_results, f, indent=2)
    print("\nClassical metrics saved to results/classical_metrics.json.")


if __name__ == '__main__':
    main()
