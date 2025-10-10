from typing import Dict, Tuple
import numpy as np
from numpy.linalg import eigh
import math
import warnings

from .config import (BAND_LOW_HZ, BAND_HIGH_HZ, EPS_REG)
from .stft_utils import stft_mc
from .stats_masks import (build_activity_masks, cov_from_frames, regularize_cov, estimate_channel_quality_ratio)

def cw_rtf(phi_pp: np.ndarray, phi_vv: np.ndarray, q0: int) -> np.ndarray:
    phi_vv_r = regularize_cov(phi_vv)
    phi_pp_r = regularize_cov(phi_pp)

    evals_b, evecs_b = eigh(phi_vv_r)
    evals_b = np.maximum(evals_b.real, 1e-12)
    inv_sqrt = np.diag(1.0 / np.sqrt(evals_b))
    B_inv_half = evecs_b @ inv_sqrt @ evecs_b.conj().T

    M = B_inv_half @ phi_pp_r @ B_inv_half.conj().T
    evals, evecs = eigh(M)
    umax_tilde = evecs[:, -1]
    umax = B_inv_half.conj().T @ umax_tilde

    num = phi_vv_r @ umax
    denom = num[q0]
    if np.abs(denom) < 1e-12:
        denom = 1e-12 + 0j
    return num / denom

def hermitian_angle(a: np.ndarray, b: np.ndarray) -> float:
    num = np.abs(np.vdot(a, b))
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < 1e-18:
        return math.pi / 2.0
    val = np.clip(num / den, 0.0, 1.0)
    return float(np.arccos(val))

def compute_theta_for_file(mixture_path, output_path, clean_path) -> Dict:
    from .io_utils import read_audio

    mixture, sr_m = read_audio(mixture_path)
    output, sr_o = read_audio(output_path)
    clean, sr_c = read_audio(clean_path)

    # Basic sanity: all streams must share sampling rate
    if not (sr_m == sr_o == sr_c):
        raise ValueError(f"Sampling rates don't match: mixture={sr_m}, output={sr_o}, clean={sr_c}")
    sr = sr_m

    X_in, freqs = stft_mc(mixture, sr)
    Y_out, _ = stft_mc(output, sr)
    F, Tfrm, Q = X_in.shape

    active_mask, noise_mask = build_activity_masks(clean, Tfrm)
    ratio = estimate_channel_quality_ratio(X_in, active_mask, noise_mask)
    q0 = int(np.argmax(ratio))

    theta = np.zeros(F, dtype=np.float64)
    for f_idx in range(F):
        Xf = X_in[f_idx, :, :]  # (T, Q)
        Yf = Y_out[f_idx, :, :]

        phi_pp = cov_from_frames(Xf, active_mask | noise_mask)
        phi_vv = cov_from_frames(Xf, noise_mask)
        phi_yy = cov_from_frames(Yf, active_mask | noise_mask)
        phi_vhatvhat = cov_from_frames(Yf, noise_mask)

        a_in = cw_rtf(phi_pp, phi_vv, q0)
        a_out = cw_rtf(phi_yy, phi_vhatvhat, q0)
        theta[f_idx] = hermitian_angle(a_in, a_out)


    band_mask = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    mean_theta_full = float(np.mean(theta))
    mean_theta_band = float(np.mean(theta[band_mask])) if np.any(band_mask) else float("nan")
    
    return {
        "ok": True,
        "sr": int(sr),
        "freqs": freqs,
        "theta": theta,
        "mean_theta_full": mean_theta_full,
        "mean_theta_200_1500": mean_theta_band,
        "q0": q0,
        "Q_in": int(mixture.shape[1]),
        "Q_out": int(output.shape[1]),
        "mixture": mixture,
        "output": output,
        "clean": clean,
        "active_mask": active_mask,
    }
