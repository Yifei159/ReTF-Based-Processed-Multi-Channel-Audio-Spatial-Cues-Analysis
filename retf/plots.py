import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from .config import (PLOT_FREQ_MAX_HZ, PLOT_MAX_THETA_RAD, BAND_LOW_HZ, BAND_HIGH_HZ, TARGET_SNR_LIST)

def plot_theta_curve(freqs: np.ndarray, theta: np.ndarray, title: str, save_path: Path):
    plt.figure(figsize=(9, 4.5))
    mask = freqs <= PLOT_FREQ_MAX_HZ
    plt.plot(freqs[mask], theta[mask], linewidth=1.2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Hermitian angle Θ (rad)")
    plt.title(title)
    if PLOT_MAX_THETA_RAD is not None:
        plt.ylim(0.0, PLOT_MAX_THETA_RAD)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def choose_nearest_snr_groups(available: List[float], targets: List[float]) -> List[float]:
    chosen = []
    remaining = available.copy()
    for t in targets:
        if len(remaining) == 0:
            break
        idx = int(np.argmin([abs(a - t) for a in remaining]))
        val = remaining.pop(idx)
        chosen.append(val)
    return chosen

def make_fig2_selected_lines(freqs: np.ndarray,
                             snrplot_to_curve: Dict[float, np.ndarray],
                             targets: List[float],
                             save_path_full: Path,
                             save_path_band: Path):
    available = sorted(snrplot_to_curve.keys())
    picked = choose_nearest_snr_groups(available, targets)

    plt.figure(figsize=(10, 5.2))
    for snr in picked:
        curve = snrplot_to_curve[snr]
        mask = freqs <= PLOT_FREQ_MAX_HZ
        plt.plot(freqs[mask], curve[mask], linewidth=1.1, marker='o', markersize=2.3, label=f"{snr:.0f} dB")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Θ (rad)")
    plt.title("Hermitian angle Θ vs frequency")
    if PLOT_MAX_THETA_RAD is not None:
        plt.ylim(0.0, PLOT_MAX_THETA_RAD)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Input SNR", loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path_full, dpi=300)
    plt.close()

    # 200–1500 Hz
    band_mask = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    plt.figure(figsize=(10, 5.2))
    for snr in picked:
        curve = snrplot_to_curve[snr]
        plt.plot(freqs[band_mask], curve[band_mask], linewidth=1.2, marker='o', markersize=2.8, label=f"{snr:.0f} dB")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Θ (rad)")
    plt.title(f"Hermitian angle Θ vs frequency ({BAND_LOW_HZ:.0f}-{BAND_HIGH_HZ:.0f} Hz)")
    if PLOT_MAX_THETA_RAD is not None:
        plt.ylim(0.0, PLOT_MAX_THETA_RAD)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Input SNR", loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path_band, dpi=300)
    plt.close()

def make_fig3_avgTheta_vs_SNR(
    snr_plot_vals, avg_full, avg_band, save_full: Path, save_band: Path,
):
    order = np.argsort(snr_plot_vals)
    x = np.array(snr_plot_vals)[order]
    y_full = np.array(avg_full)[order]
    y_band = np.array(avg_band)[order]

    plt.figure(figsize=(8.2, 5.2))
    plt.plot(x, y_full, marker='o', linewidth=1.6)
    plt.xlabel("Avg. Input SNR (dB)")
    plt.ylabel("Θ (rad)")
    plt.title("Hermitian angle Θ over full band vs input SNR")
    if PLOT_MAX_THETA_RAD is not None:
        plt.ylim(0.0, PLOT_MAX_THETA_RAD)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_full, dpi=300)
    plt.close()

    plt.figure(figsize=(8.2, 5.2))
    plt.plot(x, y_band, marker='o', linewidth=1.6)
    plt.xlabel("Avg. Input SNR (dB)")
    plt.ylabel("Θ (rad)")
    plt.title("Hermitian angle Θ over 200–1500 Hz vs input SNR")
    if PLOT_MAX_THETA_RAD is not None:
        plt.ylim(0.0, PLOT_MAX_THETA_RAD)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_band, dpi=300)
    plt.close()
