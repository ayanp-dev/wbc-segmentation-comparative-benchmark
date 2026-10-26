import time
import copy
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.metrics import compute_classification_metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str]
) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Evaluate model and return loss, classification metrics, predictions, and ground truth.
    """
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    total = len(all_labels)
    total_loss = running_loss / total if total > 0 else 0.0

    y_true_named = np.array([class_names[idx] for idx in all_labels])
    y_pred_named = np.array([class_names[idx] for idx in all_preds])

    metrics = compute_classification_metrics(y_true_named, y_pred_named, class_names)
    return total_loss, metrics, np.array(all_preds), np.array(all_labels)


def measure_inference_latency(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    num_runs: int = 100,
    warmup: int = 15
) -> float:
    """
    Measure single-image inference latency in milliseconds.
    """
    model.eval()
    dummy = torch.randn(input_shape, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_runs):
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return (elapsed / num_runs) * 1000.0


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 25,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    class_names: List[str] = None,
    patience: int = 7
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train model with early stopping tracking validation macro F1-score.
    """
    if class_names is None:
        class_names = ['basophil', 'eosinophil', 'erythroblast', 'ig']

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_f1 = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    history: Dict[str, List[float]] = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device, class_names)
        scheduler.step()

        val_acc = val_metrics['overall_accuracy']
        val_f1 = val_metrics['macro_f1']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    model.load_state_dict(best_weights)
    return model, history
