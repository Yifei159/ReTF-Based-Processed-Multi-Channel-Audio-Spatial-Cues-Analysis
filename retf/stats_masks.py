import numpy as np
from numpy.linalg import eigh
import math
from typing import Tuple
from .config import N_FFT, HOP, EPS_REG, CLEAN_ENERGY_THRESH_DB, MIN_NOISE_FRAC
from .stft_utils import frame_energy, align_frames_to_stft_len

def build_activity_masks(clean: np.ndarray, stft_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    clean_mono = clean.mean(axis=1) if clean.ndim == 2 else clean
    en = frame_energy(clean_mono, frame_len=N_FFT, hop=HOP)
    en = align_frames_to_stft_len(stft_frames, en)
    en_db = 10.0 * np.log10(np.maximum(en, 1e-12))
    peak_db = float(np.max(en_db)) if np.isfinite(np.max(en_db)) else -120.0
    thr_db = peak_db + CLEAN_ENERGY_THRESH_DB
    noise_mask = en_db <= thr_db
    min_needed = max(1, int(MIN_NOISE_FRAC * len(noise_mask)))
    if noise_mask.sum() < min_needed:
        idx = np.argsort(en)
        noise_mask[:] = False
        noise_mask[idx[:min_needed]] = True
    active_mask = ~noise_mask
    return active_mask.astype(bool), noise_mask.astype(bool)

def regularize_cov(phi: np.ndarray, eps_rel: float = EPS_REG) -> np.ndarray:
    Q = phi.shape[0]
    tr = float(np.trace(phi).real)
    reg = eps_rel * (tr / max(Q, 1))
    return phi + reg * np.eye(Q, dtype=np.complex128)

def cov_from_frames(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sel = X[mask, :]
    if sel.shape[0] == 0:
        Q = X.shape[1]
        return 1e-12 * np.eye(Q, dtype=np.complex128)
    return (sel.conj().T @ sel) / sel.shape[0]

def estimate_channel_quality_ratio(X: np.ndarray, active_mask: np.ndarray, noise_mask: np.ndarray) -> np.ndarray:
    Xa = X[:, active_mask, :]
    Xn = X[:, noise_mask, :]
    E_act = np.sum(np.abs(Xa) ** 2, axis=(0, 1)) if Xa.size else np.zeros(X.shape[2])
    E_noise = np.sum(np.abs(Xn) ** 2, axis=(0, 1)) if Xn.size else np.ones(X.shape[2]) * 1e-12
    E_noise = np.maximum(E_noise, 1e-12)
    return E_act / E_noise
