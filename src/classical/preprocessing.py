import cv2
import numpy as np
from typing import Tuple, Dict


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Resize input RGB image to the specified spatial dimensions using bilinear interpolation.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def grey_world_balance(rgb_image: np.ndarray) -> np.ndarray:
    """
    Perform per-channel grey-world color balancing on an 8-bit RGB image.

    Scales each color channel by the ratio of the global grayscale intensity mean
    to the channel's mean intensity, clamping the result to [0, 255].
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    mean_gray = float(np.mean(gray))

    balanced = np.empty_like(rgb_image, dtype=np.float32)
    for c in range(3):
        mean_c = float(np.mean(rgb_image[:, :, c]))
        scale = (mean_gray / mean_c) if mean_c > 1e-6 else 1.0
        balanced[:, :, c] = rgb_image[:, :, c] * scale

    return np.clip(balanced, 0, 255).astype(np.uint8)


def rgb_to_cmyk(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert an 8-bit RGB image to CMYK color space matching standard 8-bit quantization.

    Returns:
        np.ndarray: 4-channel uint8 array containing Cyan, Magenta, Yellow, Black (K) in [0, 255].
    """
    rgb_norm = rgb_image.astype(np.float32) / 255.0
    r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]

    c_max = np.maximum(np.maximum(r, g), b)
    k = 1.0 - c_max

    denominator = 1.0 - k
    valid_mask = denominator > 1e-7

    c = np.zeros_like(k)
    m = np.zeros_like(k)
    y = np.zeros_like(k)

    c[valid_mask] = (1.0 - r[valid_mask] - k[valid_mask]) / denominator[valid_mask]
    m[valid_mask] = (1.0 - g[valid_mask] - k[valid_mask]) / denominator[valid_mask]
    y[valid_mask] = (1.0 - b[valid_mask] - k[valid_mask]) / denominator[valid_mask]

    c = np.clip(c, 0.0, 1.0)
    m = np.clip(m, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    k = np.clip(k, 0.0, 1.0)

    cmyk = np.stack([c, m, y, k], axis=-1)
    return np.round(cmyk * 255.0).astype(np.uint8)


def rgb_to_hls(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert an 8-bit RGB image to HLS color space with 8-bit normalization across all channels.

    Hue is mapped linearly from [0, 360) to [0, 255], and Luminance and Saturation to [0, 255].

    Returns:
        np.ndarray: 3-channel uint8 array containing Hue, Lightness, Saturation in [0, 255].
    """
    rgb_norm = rgb_image.astype(np.float32) / 255.0
    r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]

    c_max = np.maximum(np.maximum(r, g), b)
    c_min = np.minimum(np.minimum(r, g), b)
    delta = c_max - c_min

    l = (c_max + c_min) / 2.0

    s = np.zeros_like(l)
    non_zero_delta = delta > 1e-7

    low_l = non_zero_delta & (l <= 0.5)
    high_l = non_zero_delta & (l > 0.5)

    s[low_l] = delta[low_l] / (c_max[low_l] + c_min[low_l] + 1e-7)
    s[high_l] = delta[high_l] / (2.0 - c_max[high_l] - c_min[high_l] + 1e-7)

    h = np.zeros_like(l)
    r_is_max = non_zero_delta & (c_max == r)
    g_is_max = non_zero_delta & (c_max == g) & (~r_is_max)
    b_is_max = non_zero_delta & (c_max == b) & (~r_is_max) & (~g_is_max)

    h[r_is_max] = ((g[r_is_max] - b[r_is_max]) / delta[r_is_max]) % 6.0
    h[g_is_max] = ((b[g_is_max] - r[g_is_max]) / delta[g_is_max]) + 2.0
    h[b_is_max] = ((r[b_is_max] - g[b_is_max]) / delta[b_is_max]) + 4.0

    h_deg = (h * 60.0) % 360.0

    h_8bit = np.clip(np.round(h_deg / 360.0 * 255.0), 0, 255).astype(np.uint8)
    l_8bit = np.clip(np.round(s * 0.0 + l * 255.0), 0, 255).astype(np.uint8)
    s_8bit = np.clip(np.round(s * 255.0), 0, 255).astype(np.uint8)

    return np.stack([h_8bit, l_8bit, s_8bit], axis=-1)


def rgb_to_hsv(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert an 8-bit RGB image to HSV color space with 8-bit normalization across all channels.

    Hue is mapped linearly from [0, 360) to [0, 255], Saturation to [0, 255], and Value to [0, 255].

    Returns:
        np.ndarray: 3-channel uint8 array containing Hue, Saturation, Value in [0, 255].
    """
    rgb_norm = rgb_image.astype(np.float32) / 255.0
    r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]

    c_max = np.maximum(np.maximum(r, g), b)
    c_min = np.minimum(np.minimum(r, g), b)
    delta = c_max - c_min

    v = c_max

    s = np.zeros_like(v)
    valid_v = c_max > 1e-7
    s[valid_v] = delta[valid_v] / c_max[valid_v]

    h = np.zeros_like(v)
    non_zero_delta = delta > 1e-7
    r_is_max = non_zero_delta & (c_max == r)
    g_is_max = non_zero_delta & (c_max == g) & (~r_is_max)
    b_is_max = non_zero_delta & (c_max == b) & (~r_is_max) & (~g_is_max)

    h[r_is_max] = ((g[r_is_max] - b[r_is_max]) / delta[r_is_max]) % 6.0
    h[g_is_max] = ((b[g_is_max] - r[g_is_max]) / delta[g_is_max]) + 2.0
    h[b_is_max] = ((r[b_is_max] - g[b_is_max]) / delta[b_is_max]) + 4.0

    h_deg = (h * 60.0) % 360.0

    h_8bit = np.clip(np.round(h_deg / 360.0 * 255.0), 0, 255).astype(np.uint8)
    s_8bit = np.clip(np.round(s * 255.0), 0, 255).astype(np.uint8)
    v_8bit = np.clip(np.round(v * 255.0), 0, 255).astype(np.uint8)

    return np.stack([h_8bit, s_8bit, v_8bit], axis=-1)


def rgb_to_lab(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert an 8-bit RGB image to CIELAB color space using OpenCV standard 8-bit quantization.

    Returns:
        np.ndarray: 3-channel uint8 array containing L* (0..255), a* (0..255), b* (0..255).
    """
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2Lab)


def rgb_to_ycbcr(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert an 8-bit RGB image to YCrCb color space.

    Returns:
        np.ndarray: 3-channel uint8 array containing Y, Cr, Cb in [0, 255].
    """
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)


def compute_soft_map(cmyk: np.ndarray, hls: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    """
    Compute the dimensionless soft-map for leukocyte region segmentation.

    Formulation:
        sftmap = (S_HSV - S_HLS) + (min(M, S_HLS) - min(min(M, S_HLS), K))
    where arithmetic operates under 8-bit saturation [0, 255].

    Returns:
        np.ndarray: 2D uint8 soft-map array.
    """
    m = cmyk[:, :, 1].astype(np.int16)
    k = cmyk[:, :, 3].astype(np.int16)
    s_hls = hls[:, :, 2].astype(np.int16)
    s_hsv = hsv[:, :, 1].astype(np.int16)

    minus = np.maximum(0, s_hsv - s_hls)
    ms = np.minimum(m, s_hls)
    b_temp = np.minimum(ms, k)
    base_sftmap = ms - b_temp

    sftmap = np.clip(minus + base_sftmap, 0, 255).astype(np.uint8)
    return sftmap


def preprocess_image(rgb_image: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> Dict[str, np.ndarray]:
    """
    Execute complete preprocessing pipeline: resize, color balancing, and multi-color-space conversion.
    """
    resized = resize_image(rgb_image, target_size)
    balanced = grey_world_balance(resized)

    cmyk = rgb_to_cmyk(balanced)
    hls = rgb_to_hls(balanced)
    hsv = rgb_to_hsv(balanced)
    lab = rgb_to_lab(balanced)
    ycrcb = rgb_to_ycbcr(balanced)

    sftmap = compute_soft_map(cmyk, hls, hsv)

    return {
        'resized_rgb': resized,
        'balanced_rgb': balanced,
        'cmyk': cmyk,
        'hls': hls,
        'hsv': hsv,
        'lab': lab,
        'ycrcb': ycrcb,
        'sftmap': sftmap
    }
