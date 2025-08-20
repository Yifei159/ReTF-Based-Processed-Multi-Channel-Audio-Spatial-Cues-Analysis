import numpy as np
from typing import Tuple
from scipy.signal import stft, get_window
from .config import N_FFT, HOP, WINDOW

def stft_mc(x: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    win = get_window(WINDOW, N_FFT, fftbins=True)
    X_list = []
    for q in range(x.shape[1]):
        f, t, Z = stft(x[:, q], fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP,
                       window=win, return_onesided=True, boundary="zeros", padded=True)
        X_list.append(Z)
    X = np.stack(X_list, axis=2)  # (F, T, Q)
    return X, f

def frame_energy(sig: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    T = len(sig)
    n_frames = 1 + max(0, (T - frame_len) // hop) if T > 0 else 1
    energies = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        s = i * hop
        e = min(s + frame_len, T)
        frame = sig[s:e]
        if len(frame) < frame_len:
            pad = np.zeros(frame_len, dtype=frame.dtype)
            pad[: len(frame)] = frame
            frame = pad
        energies[i] = np.mean(frame ** 2)
    return energies

def align_frames_to_stft_len(len_stft_frames: int, arr: np.ndarray) -> np.ndarray:
    out = np.zeros(len_stft_frames, dtype=arr.dtype)
    L = min(len_stft_frames, len(arr))
    out[:L] = arr[:L]
    return out
