import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from typing import Dict, Tuple, Optional


def segment_cytoplasm(sftmap: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Generate the cytoplasm binary mask from the dimensionless soft-map.

    Applies global Otsu binarization, morphological hole filling, disk dilation,
    and connected-component filtering to retain the single largest foreground region.

    Returns:
        np.ndarray: 2D uint8 mask with values in {0, 1}.
    """
    _, binary = cv2.threshold(sftmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    filled = binary_fill_holes(binary > 0).astype(np.uint8)

    kernel_dim = 2 * radius + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_dim, kernel_dim))
    dilated = cv2.dilate(filled, se)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    if num_labels <= 1:
        return dilated

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cytmask = (labels == largest_label).astype(np.uint8)

    return cytmask


def segment_nucleus(
    sftmap: np.ndarray,
    m_channel: np.ndarray,
    cytmask: np.ndarray,
    radius: int = 3
) -> np.ndarray:
    """
    Generate the nucleus binary mask constrained within the cytoplasm boundary.

    Smooths the soft-map (or fallback Magenta channel) via Gaussian filtering, applies
    Otsu thresholding, disk erosion, logical conjunction with the cytoplasm mask,
    and extracts the filled primary contour.

    Returns:
        np.ndarray: 2D uint8 mask with values in {0, 1}.
    """
    sftmap_smooth = cv2.GaussianBlur(sftmap, (5, 5), sigmaX=1.0, sigmaY=1.0, borderType=cv2.BORDER_REPLICATE)
    m_smooth = cv2.GaussianBlur(m_channel, (5, 5), sigmaX=1.0, sigmaY=1.0, borderType=cv2.BORDER_REPLICATE)

    thresh_val, binary = cv2.threshold(sftmap_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(binary > 0) == 0:
        _, binary = cv2.threshold(m_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_dim = 2 * radius + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_dim, kernel_dim))
    eroded = cv2.erode(binary, se)

    constrained = (eroded > 0) & (cytmask > 0)

    contours, _ = cv2.findContours(constrained.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    nu_mask = np.zeros_like(cytmask, dtype=np.uint8)
    if len(contours) > 0:
        largest_contour = max(contours, key=len)
        cv2.drawContours(nu_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
    else:
        nu_mask = constrained.astype(np.uint8)

    return nu_mask


def generate_auxiliary_regions(cytmask: np.ndarray, nu_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate convex hull, ROC (Region of Cytoplasm), and primary contours for cyt and nu masks.

    Returns:
        convex_hull_mask (np.ndarray): Binary mask of cytoplasm convex hull {0, 1}.
        roc_mask (np.ndarray): Binary mask of Region of Cytoplasm {0, 1}.
        cyt_contour (np.ndarray): Largest external contour coordinates of cytoplasm.
        nu_contour (np.ndarray): Largest external contour coordinates of nucleus.
    """
    h, w = cytmask.shape
    cvx_mask = np.zeros((h, w), dtype=np.uint8)
    cyt_contours, _ = cv2.findContours(cytmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cyt_contour = np.empty((0, 1, 2), dtype=np.int32)
    if len(cyt_contours) > 0:
        cyt_contour = max(cyt_contours, key=len)
        hull = cv2.convexHull(cyt_contour)
        cv2.drawContours(cvx_mask, [hull], -1, 1, thickness=cv2.FILLED)
    else:
        cvx_mask = cytmask.copy()

    roc_mask = ((cvx_mask > 0) & ~(nu_mask > 0)).astype(np.uint8)

    nu_contours, _ = cv2.findContours(nu_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    nu_contour = max(nu_contours, key=len) if len(nu_contours) > 0 else np.empty((0, 1, 2), dtype=np.int32)

    return cvx_mask, roc_mask, cyt_contour, nu_contour


def segment_wbc(
    sftmap: np.ndarray,
    m_channel: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Execute complete WBC morphological segmentation returning all component masks.
    """
    cytmask = segment_cytoplasm(sftmap)
    nu_mask = segment_nucleus(sftmap, m_channel, cytmask)
    cvx_mask, roc_mask, cyt_contour, nu_contour = generate_auxiliary_regions(cytmask, nu_mask)

    return {
        'cytoplasm_mask': cytmask,
        'nucleus_mask': nu_mask,
        'convex_hull_mask': cvx_mask,
        'roc_mask': roc_mask,
        'cytoplasm_contour': cyt_contour,
        'nucleus_contour': nu_contour
    }
