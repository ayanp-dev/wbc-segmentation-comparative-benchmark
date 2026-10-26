import time
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from src.utils.metrics import compute_classification_metrics


def build_pipeline(model_type: str, use_pca: bool = False, random_state: int = 42) -> Pipeline:
    """
    Construct an sklearn Pipeline with StandardScaler and optional PCA(0.95).
    Transformations are fitted strictly on training data during the fit step of the pipeline.
    """
    steps = [('scaler', StandardScaler())]

    if use_pca:
        steps.append(('pca', PCA(n_components=0.95, random_state=random_state)))

    if model_type == 'svm':
        steps.append(('clf', SVC(kernel='rbf', decision_function_shape='ovo', random_state=random_state)))
    elif model_type == 'random_forest':
        steps.append(('clf', RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)))
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected 'svm' or 'random_forest'.")

    return Pipeline(steps)


def train_and_evaluate_model(
    model_type: str,
    use_pca: bool,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    random_state: int = 42
) -> Dict[str, Any]:
    """Train pipeline strictly on X_train and evaluate on validation and test partitions."""
    pipeline = build_pipeline(model_type, use_pca=use_pca, random_state=random_state)

    train_start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - train_start

    val_preds = pipeline.predict(X_val)
    val_metrics = compute_classification_metrics(y_val, val_preds, class_names)

    # Benchmark test inference latency
    test_start = time.perf_counter()
    test_preds = pipeline.predict(X_test)
    test_time = time.perf_counter() - test_start
    mean_latency_ms = (test_time / len(X_test)) * 1000.0

    test_metrics = compute_classification_metrics(y_test, test_preds, class_names)

    num_components = None
    if use_pca:
        pca_step = pipeline.named_steps.get('pca')
        if pca_step is not None:
            num_components = int(pca_step.n_components_)

    return {
        'model_name': f"{'PCA ' if use_pca else ''}{'SVM' if model_type == 'svm' else 'Random Forest'}",
        'model_type': model_type,
        'use_pca': use_pca,
        'num_components': num_components,
        'train_time_sec': train_time,
        'inference_latency_ms': mean_latency_ms,
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'pipeline': pipeline
    }
