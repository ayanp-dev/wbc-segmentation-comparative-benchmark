import os
import glob
from typing import Tuple, List, Dict, Optional, Callable
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from src.deep_learning.augmentations import (
    get_baseline_transforms,
    get_domain_augmentations,
    get_eval_transforms
)


class WBCDataset(Dataset):
    """
    PyTorch Dataset for White Blood Cell peripheral smear microscopy images.
    """
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def create_dl_splits(
    data_dir: str = 'bloodcells_4classes_500',
    random_state: int = 42,
    low_data_fraction: float = 0.15
) -> Dict[str, pd.DataFrame]:
    """
    Execute stratified 60/20/20 train/val/test split and low-data stratified sub-sampling.
    Matches the exact partition scheme and seed (random_state=42) used in the classical pipeline.
    """
    classes = ['basophil', 'eosinophil', 'erythroblast', 'ig']
    class_to_idx = {c: i for i, c in enumerate(classes)}

    records = []
    for c_name in sorted(classes):
        pattern = os.path.join(data_dir, c_name, '*.jpg')
        for p in sorted(glob.glob(pattern)):
            records.append({
                'image_path': p,
                'label_name': c_name,
                'label_idx': class_to_idx[c_name]
            })

    df = pd.DataFrame(records)

    # 60% Train, 40% Temp
    train_df, temp_df = train_test_split(
        df, test_size=0.40, stratify=df['label_idx'], random_state=random_state
    )

    # 20% Val, 20% Test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['label_idx'], random_state=random_state
    )

    # Stratified sub-sample for low-data regimes
    train_low_df, _ = train_test_split(
        train_df,
        train_size=low_data_fraction,
        stratify=train_df['label_idx'],
        random_state=random_state
    )

    return {
        'train_full': train_df.reset_index(drop=True),
        'train_low': train_low_df.reset_index(drop=True),
        'val': val_df.reset_index(drop=True),
        'test': test_df.reset_index(drop=True),
        'classes': classes,
        'class_to_idx': class_to_idx
    }


def get_dataloaders(
    splits: Dict[str, pd.DataFrame],
    regime: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (224, 224)
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders for a specific training regime.

    Supported regimes:
        - 'low_data_baseline': Stratified low-data subset without synthetic augmentation
        - 'low_data_augmented': Identical stratified low-data subset with domain augmentations
        - 'full_data': 100% training data split
    """
    eval_transform = get_eval_transforms(img_size)

    if regime == 'low_data_baseline':
        train_df = splits['train_low']
        train_transform = get_baseline_transforms(img_size)
    elif regime == 'low_data_augmented':
        train_df = splits['train_low']
        train_transform = get_domain_augmentations(img_size)
    elif regime == 'full_data':
        train_df = splits['train_full']
        train_transform = get_baseline_transforms(img_size)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    train_dataset = WBCDataset(
        train_df['image_path'].tolist(),
        train_df['label_idx'].tolist(),
        transform=train_transform
    )
    val_dataset = WBCDataset(
        splits['val']['image_path'].tolist(),
        splits['val']['label_idx'].tolist(),
        transform=eval_transform
    )
    test_dataset = WBCDataset(
        splits['test']['image_path'].tolist(),
        splits['test']['label_idx'].tolist(),
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader
