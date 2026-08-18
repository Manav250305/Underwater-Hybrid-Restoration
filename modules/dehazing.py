"""
Module 4 — Dark-Channel-Prior (DCP) Dehazing with Guided-Filter Refinement
Adapted for the underwater medium: the dark channel is computed only over
the green and blue channels (red is excluded, since red is already
strongly attenuated underwater and would corrupt the dark-channel
estimate), following the Underwater DCP (UDCP) adaptation in the
literature (Drews et al.).
"""

import cv2
import numpy as np


def get_dark_channel(img: np.ndarray, patch_size: int = 15, underwater: bool = True) -> np.ndarray:
    """
    Compute the dark channel. For underwater images, use only the
    green/blue channels (UDCP), since the red channel is attenuated by
    absorption rather than scattering and misleads the standard DCP.
    """
    if underwater:
        channels = img[:, :, 0:2]  # B, G only (OpenCV BGR order)
    else:
        channels = img

    min_channel = np.min(channels, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel


def estimate_atmospheric_light(img: np.ndarray, dark_channel: np.ndarray, top_percent: float = 0.001) -> np.ndarray:
    """
    Estimate atmospheric (background) light from the brightest pixels in
    the dark channel, following He et al.'s original DCP procedure.
    """
    h, w = dark_channel.shape
    num_pixels = h * w
    num_top = max(int(num_pixels * top_percent), 1)

    flat_dark = dark_channel.reshape(num_pixels)
    flat_img = img.reshape(num_pixels, 3)

    indices = np.argsort(flat_dark)[-num_top:]
    brightest_pixels = flat_img[indices]

    atmospheric_light = brightest_pixels.max(axis=0)
    return atmospheric_light.astype(np.float32)


def estimate_transmission(img: np.ndarray, atmospheric_light: np.ndarray,
                           patch_size: int = 15, omega: float = 0.85,
                           underwater: bool = True) -> np.ndarray:
    """
    Estimate the transmission map t(x) = 1 - omega * dark_channel(I/A).
    omega keeps a small amount of haze for a natural look (standard DCP practice).
    """
    normalized = img.astype(np.float32) / (atmospheric_light + 1e-6)
    normalized = np.clip(normalized, 0, 1)
    dark_channel_norm = get_dark_channel(normalized, patch_size, underwater)
    transmission = 1.0 - omega * dark_channel_norm
    return transmission


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 40, eps: float = 1e-3) -> np.ndarray:
    """
    Edge-preserving refinement of the transmission map using the guided
    filter (He et al.), guided by the grayscale input image.
    """
    guide = guide.astype(np.float32) / 255.0
    src = src.astype(np.float32)

    mean_guide = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
    mean_src = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
    mean_guide_src = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
    cov_guide_src = mean_guide_src - mean_guide * mean_src

    mean_guide_sq = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
    var_guide = mean_guide_sq - mean_guide * mean_guide

    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))

    refined = mean_a * guide + mean_b
    return refined


def recover_scene_radiance(img: np.ndarray, transmission: np.ndarray,
                            atmospheric_light: np.ndarray, t_min: float = 0.1) -> np.ndarray:
    """
    Invert the haze model: J = (I - A) / max(t, t_min) + A
    """
    t = np.clip(transmission, t_min, 1.0)
    t = t[:, :, np.newaxis]

    img_f = img.astype(np.float32)
    radiance = (img_f - atmospheric_light) / t + atmospheric_light
    radiance = np.clip(radiance, 0, 255)
    return radiance.astype(np.uint8)


def dcp_dehaze(img_bgr: np.ndarray, patch_size: int = 15, omega: float = 0.85,
                guided_radius: int = 40, guided_eps: float = 1e-3,
                t_min: float = 0.1) -> np.ndarray:
    """
    Full underwater DCP dehazing pipeline: dark channel -> atmospheric
    light -> transmission -> guided-filter refinement -> scene recovery.
    """
    dark_channel = get_dark_channel(img_bgr, patch_size, underwater=True)
    atmospheric_light = estimate_atmospheric_light(img_bgr, dark_channel)
    transmission = estimate_transmission(img_bgr, atmospheric_light, patch_size, omega, underwater=True)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    transmission_refined = guided_filter(gray, transmission * 255.0, guided_radius, guided_eps) / 255.0
    transmission_refined = np.clip(transmission_refined, 0, 1)

    dehazed = recover_scene_radiance(img_bgr, transmission_refined, atmospheric_light, t_min)
    return dehazed
