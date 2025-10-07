from typing import Dict, Tuple
import numpy as np
from scipy.signal import stft, get_window
from .config import (N_FFT, HOP, WINDOW, BAND_LOW_HZ, BAND_HIGH_HZ, SPEED_OF_SOUND, MIC_CHANNEL_IDXS)
from .geometry import get_mic_geometry_matrix

def _max_lag_samples(p_i: np.ndarray, p_j: np.ndarray, fs: int) -> int:
    d = np.linalg.norm(p_i - p_j)
    return int(np.ceil(d / SPEED_OF_SOUND * fs))

def gcc_phat_masked(x1: np.ndarray, x2: np.ndarray, fs: int, active_mask: np.ndarray,
                    nfft: int = N_FFT, hop: int = HOP, fmin: float = BAND_LOW_HZ, fmax: float = BAND_HIGH_HZ) -> np.ndarray:
    win = get_window(WINDOW, nfft, fftbins=True)
    _, _, Z1 = stft(x1, fs=fs, nperseg=nfft, noverlap=nfft-hop, window=win, return_onesided=True, boundary="zeros", padded=True)
    _, _, Z2 = stft(x2, fs=fs, nperseg=nfft, noverlap=nfft-hop, window=win, return_onesided=True, boundary="zeros", padded=True)

    L = min(Z1.shape[1], Z2.shape[1], len(active_mask))
    if L <= 0:
        return np.zeros(nfft)
    Z1 = Z1[:, :L] 
    Z2 = Z2[:, :L]
    act = np.asarray(active_mask[:L], dtype=bool)
    if act.sum() == 0:
        return np.zeros(nfft)

    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    if band.sum() == 0:
        band = np.ones_like(freqs, dtype=bool)

    C_acc = None
    idx_act = np.where(act)[0]
    for t in idx_act:
        X1 = Z1[band, t]
        X2 = Z2[band, t]
        C = X1 * np.conj(X2)
        C /= np.maximum(np.abs(C), 1e-12)
        if C_acc is None:
            C_acc = C
        else:
            C_acc += C

    cc = np.fft.irfft(C_acc, n=nfft)
    cc = np.concatenate((cc[-(nfft//2):], cc[:(nfft//2)+1]))
    return cc

def _parabolic_peak_offset(mag: np.ndarray, k: int) -> float:
    if k <= 0 or k >= len(mag) - 1:
        return 0.0
    y_m1, y0, y_p1 = float(mag[k - 1]), float(mag[k]), float(mag[k + 1])
    denom = (y_m1 - 2.0 * y0 + y_p1)
    if abs(denom) < 1e-20:
        return 0.0
    delta = 0.5 * (y_m1 - y_p1) / denom
    return float(np.clip(delta, -1.0, 1.0))

def _tau_from_cc(cc: np.ndarray, fs: int, maxlag: int) -> float:
    mid = len(cc) // 2
    lo = max(0, mid - maxlag)
    hi = min(len(cc) - 1, mid + maxlag)
    mag = np.abs(cc)
    k = int(np.argmax(mag[lo:hi + 1])) + lo
    delta = _parabolic_peak_offset(mag, k)
    k_sub = k + delta
    return (k_sub - mid) / float(fs)

def estimate_u_from_tdoa(tdoas: Dict[Tuple[int,int], float], mic_pos: np.ndarray) -> np.ndarray:
    rows = []
    b = []
    for (i, j), tau in tdoas.items():
        rij = mic_pos[i] - mic_pos[j]
        rows.append(rij / SPEED_OF_SOUND)
        b.append(tau)
    A = np.vstack(rows)
    b = np.array(b)
    u_hat, *_ = np.linalg.lstsq(A, b, rcond=None)
    norm = np.linalg.norm(u_hat)
    if norm < 1e-12:
        return np.array([np.nan, np.nan, np.nan])
    return u_hat / norm

def estimate_doa_gccphat_from_multichannel(x: np.ndarray, sr: int,
                                           ch_idxs=(0,1,2),
                                           mic_pos=None,
                                           active_mask=None) -> np.ndarray:
    """Estimate a unit direction vector using GCC–PHAT from 3 channels."""
    if mic_pos is None:
        mic_pos = get_mic_geometry_matrix()
    a, b, c = ch_idxs
    sA = x[:, a]
    sB = x[:, b]
    sC = x[:, c]

    if active_mask is None:
        from scipy.signal import stft, get_window
        win = get_window(WINDOW, N_FFT, fftbins=True)
        _, tA, _ = stft(sA, fs=sr, nperseg=N_FFT, noverlap=N_FFT-HOP, window=win, return_onesided=True)
        active_mask = np.ones(len(tA), dtype=bool)

    cc_AB = gcc_phat_masked(sA, sB, sr, active_mask, nfft=N_FFT, hop=HOP, fmin=BAND_LOW_HZ, fmax=BAND_HIGH_HZ)
    cc_BC = gcc_phat_masked(sB, sC, sr, active_mask, nfft=N_FFT, hop=HOP, fmin=BAND_LOW_HZ, fmax=BAND_HIGH_HZ)
    cc_CA = gcc_phat_masked(sC, sA, sr, active_mask, nfft=N_FFT, hop=HOP, fmin=BAND_LOW_HZ, fmax=BAND_HIGH_HZ)

    maxlag_AB = _max_lag_samples(mic_pos[0], mic_pos[1], sr)
    maxlag_BC = _max_lag_samples(mic_pos[1], mic_pos[2], sr)
    maxlag_CA = _max_lag_samples(mic_pos[2], mic_pos[0], sr)

    tau_AB = _tau_from_cc(cc_AB, sr, maxlag_AB)
    tau_BC = _tau_from_cc(cc_BC, sr, maxlag_BC)
    tau_CA = _tau_from_cc(cc_CA, sr, maxlag_CA)

    tdoas = {(0,1): tau_AB, (1,2): tau_BC, (2,0): tau_CA}
    u = estimate_u_from_tdoa(tdoas, mic_pos)
    return u

def angle_between_unit_vectors_deg(u: np.ndarray, v: np.ndarray) -> float:
    if np.any(np.isnan(u)) or np.any(np.isnan(v)):
        return float("nan")
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))

def align_to_reference_hemisphere(u: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
    if u is None or u_ref is None or np.any(np.isnan(u)) or np.any(np.isnan(u_ref)):
        return u
    return u if float(np.dot(u, u_ref)) >= 0.0 else -u
