import torch
import torchvision.transforms as transforms
from typing import Tuple


class AddGaussianNoise:
    """
    Add zero-mean Gaussian noise to an image tensor.
    """
    def __init__(self, mean: float = 0.0, std: float = 0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


def get_baseline_transforms(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Standard resizing and ImageNet normalization without synthetic augmentations.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_domain_augmentations(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Domain-specific synthetic augmentations for peripheral blood smear microscopy.
    Includes photometric stain jitter, affine/perspective warps, additive Gaussian noise,
    and synthetic occlusion (cutout).
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.06),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.015),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.15), value='random')
    ])


def get_eval_transforms(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Deterministic evaluation transforms for validation and test sets.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
