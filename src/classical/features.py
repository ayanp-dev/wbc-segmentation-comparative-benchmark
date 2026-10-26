import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


FEATURE_NAMES: List[str] = [
    # 24 Nucleus-to-ConvexHull color ratios (12 Means, then 12 Stds)
    "R_NucMean", "G_NucMean", "B_NucMean",
    "H_NucMean", "S_NucMean", "V_NucMean",
    "L_NucMean", "A_NucMean", "B_Lab_NucMean",
    "Y_NucMean", "Cr_NucMean", "Cb_NucMean",
    "R_NucStd", "G_NucStd", "B_NucStd",
    "H_NucStd", "S_NucStd", "V_NucStd",
    "L_NucStd", "A_NucStd", "B_Lab_NucStd",
    "Y_NucStd", "Cr_NucStd", "Cb_NucStd",

    # 24 ROC-to-ConvexHull color ratios (12 Means, then 12 Stds)
    "R_RocMean", "G_RocMean", "B_RocMean",
    "H_RocMean", "S_RocMean", "V_RocMean",
    "L_RocMean", "A_RocMean", "B_Lab_RocMean",
    "Y_RocMean", "Cr_RocMean", "Cb_RocMean",
    "R_RocStd", "G_RocStd", "B_RocStd",
    "H_RocStd", "S_RocStd", "V_RocStd",
    "L_RocStd", "A_RocStd", "B_Lab_RocStd",
    "Y_RocStd", "Cr_RocStd", "Cb_RocStd",

    # 3 Geometric shape descriptors
    "Circularity", "Convexity", "Solidity"
]


def extract_shape_features(nu_mask: np.ndarray, cvx_mask: np.ndarray) -> np.ndarray:
    """
    Compute geometric shape descriptors: Circularity, Convexity, and Solidity.
    Perimeter is computed via morphological boundary extraction.

    Returns:
        np.ndarray: 1D float array of shape (3,).
    """
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    nu_binary = (nu_mask > 0).astype(np.uint8)
    cvx_binary = (cvx_mask > 0).astype(np.uint8)

    nu_perim_mask = nu_binary & ~cv2.erode(nu_binary, se)
    cvx_perim_mask = cvx_binary & ~cv2.erode(cvx_binary, se)

    perim_nu = float(np.sum(nu_perim_mask))
    perim_cyt = float(np.sum(cvx_perim_mask))

    area_nu = float(np.sum(nu_binary))
    area_chull = float(np.sum(cvx_binary))

    if area_nu > 0:
        circularity = (perim_nu ** 2) / (4.0 * np.pi * area_nu)
    else:
        circularity = 0.0

    if perim_nu > 0:
        convexity = perim_cyt / perim_nu
    else:
        convexity = 0.0

    if area_chull > 0:
        solidity = area_nu / area_chull
    else:
        solidity = 0.0

    return np.array([circularity, convexity, solidity], dtype=np.float64)


def extract_color_features(
    channels: List[np.ndarray],
    nu_mask: np.ndarray,
    cvx_mask: np.ndarray,
    roc_mask: np.ndarray
) -> np.ndarray:
    """
    Compute 48 color ratio features across 12 color channels using vectorized boolean mask indexing.

    Computes means and sample standard deviations for Nucleus, Convex Hull, and ROC,
    and returns ratios of Nucleus/ConvexHull and ROC/ConvexHull.

    Returns:
        np.ndarray: 1D float array of shape (48,).
    """
    nu_idx = nu_mask > 0
    cvx_idx = cvx_mask > 0
    roc_idx = roc_mask > 0

    num_channels = len(channels)
    ncl_stats = np.zeros((num_channels, 2), dtype=np.float64)
    cvx_stats = np.zeros((num_channels, 2), dtype=np.float64)
    roc_stats = np.zeros((num_channels, 2), dtype=np.float64)

    for i, ch in enumerate(channels):
        ncl_pxls = ch[nu_idx].astype(np.float64)
        cvx_pxls = ch[cvx_idx].astype(np.float64)
        roc_pxls = ch[roc_idx].astype(np.float64)

        if len(ncl_pxls) > 0:
            ncl_stats[i, 0] = np.mean(ncl_pxls)
            ncl_stats[i, 1] = np.std(ncl_pxls, ddof=1) if len(ncl_pxls) > 1 else 0.0

        if len(cvx_pxls) > 0:
            cvx_stats[i, 0] = np.mean(cvx_pxls)
            cvx_stats[i, 1] = np.std(cvx_pxls, ddof=1) if len(cvx_pxls) > 1 else 0.0

        if len(roc_pxls) > 0:
            roc_stats[i, 0] = np.mean(roc_pxls)
            roc_stats[i, 1] = np.std(roc_pxls, ddof=1) if len(roc_pxls) > 1 else 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        ncl_ratio_matrix = ncl_stats / cvx_stats
        roc_ratio_matrix = roc_stats / cvx_stats

    ratio_ncl = ncl_ratio_matrix.flatten(order='F')
    ratio_roc = roc_ratio_matrix.flatten(order='F')

    color_features = np.concatenate([ratio_ncl, ratio_roc])
    color_features = np.nan_to_num(color_features, nan=0.0, posinf=1.0, neginf=1.0)

    return color_features


def extract_features(
    preprocessed: Dict[str, np.ndarray],
    segmentation: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Extract complete 51-dimensional handcrafted feature vector for a single leukocyte image.
    """
    balanced_rgb = preprocessed['balanced_rgb']
    hsv = preprocessed['hsv']
    lab = preprocessed['lab']
    ycrcb = preprocessed['ycrcb']

    channels = [
        balanced_rgb[:, :, 0],  # R
        balanced_rgb[:, :, 1],  # G
        balanced_rgb[:, :, 2],  # B
        hsv[:, :, 0],           # H
        hsv[:, :, 1],           # S
        hsv[:, :, 2],           # V
        lab[:, :, 0],           # L
        lab[:, :, 1],           # A
        lab[:, :, 2],           # B
        ycrcb[:, :, 0],         # Y
        ycrcb[:, :, 1],         # Cr
        ycrcb[:, :, 2]          # Cb
    ]

    nu_mask = segmentation['nucleus_mask']
    cvx_mask = segmentation['convex_hull_mask']
    roc_mask = segmentation['roc_mask']

    color_feats = extract_color_features(channels, nu_mask, cvx_mask, roc_mask)
    shape_feats = extract_shape_features(nu_mask, cvx_mask)

    return np.concatenate([color_feats, shape_feats])
