from pathlib import Path
from typing import Dict, List
import math, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from .config import (DATA_DIR, OUT_DIR, PER_FILE_DIR, SUM_DIR, BAND_LOW_HZ, BAND_HIGH_HZ, TARGET_SNR_LIST)
from .io_utils import (find_leaf_dirs, extract_distance_from_parts, extract_snr_from_parts, read_audio)
from .retf import compute_theta_for_file
from .plots import (plot_theta_curve, make_fig2_selected_lines, make_fig3_avgTheta_vs_SNR)
from .gcc_phat import (estimate_doa_gccphat_from_multichannel, angle_between_unit_vectors_deg, align_to_reference_hemisphere)
from .geometry import get_mic_geometry_matrix

def ensure_dirs():
    PER_FILE_DIR.mkdir(parents=True, exist_ok=True)
    SUM_DIR.mkdir(parents=True, exist_ok=True)

def run_analysis(data_dir: Path = DATA_DIR, out_dir: Path = OUT_DIR) -> None:
    global PER_FILE_DIR, SUM_DIR
    # rebind output dirs if the user passed a different out_dir
    PER_FILE_DIR = out_dir / "per_file"
    SUM_DIR = out_dir / "summaries"
    ensure_dirs()

    leaf_dirs = find_leaf_dirs(data_dir)
    if not leaf_dirs:
        print(f"No valid leaf directories found under {data_dir}")
        return

    records = []
    snr_to_curves: Dict[float, List[np.ndarray]] = {}
    snr_to_freqs: Dict[float, np.ndarray] = {}
    snr_to_full_means: Dict[float, List[float]] = {}
    snr_to_band_means: Dict[float, List[float]] = {}
    gcc_rows: List[Dict] = []

    for leaf in tqdm(leaf_dirs, desc="Processing"):
        mixture = leaf / "mixture_input.wav"
        output = leaf / "vocals_pred.wav"
        clean = leaf / "vocals_clean.wav"

        parts_full = [p.name for p in leaf.parents] + [leaf.name]
        distance = extract_distance_from_parts(parts_full)
        snr_db_raw = extract_snr_from_parts(parts_full)
        snr_plot = -abs(snr_db_raw) if not math.isnan(snr_db_raw) else float("nan" )

        # -------------- ReTF distance --------------
        res = compute_theta_for_file(mixture, output, clean)

        # -------------- Naming --------------
        outer_name = None
        for p in leaf.parents:
            if p == data_dir:
                break
            if 'D' in p.name and 'm' in p.name:  # crude distance token check
                outer_name = p.name
                break
        if outer_name is None:
            outer_name = leaf.parent.name

        if res.get("ok"):
            freqs = res["freqs"]
            theta = res["theta"]

            df_curve = pd.DataFrame({"freq_hz": freqs, "theta_rad": theta})
            df_curve.to_csv(PER_FILE_DIR / f"{outer_name}_theta_curve.csv", index=False)
            plot_theta_curve(freqs, theta, outer_name, PER_FILE_DIR / f"{outer_name}_theta_curve.png")  # noqa: E501

            # -------------- GCC–PHAT DOA errors --------------
            try:
                u_clean = estimate_doa_gccphat_from_multichannel(res["clean"], res["sr"], active_mask=res["active_mask"])  # noqa: E501
                u_mix   = estimate_doa_gccphat_from_multichannel(res["mixture"], res["sr"], active_mask=res["active_mask"])  # noqa: E501
                u_mwf   = estimate_doa_gccphat_from_multichannel(res["output"],  res["sr"], active_mask=res["active_mask"])  # noqa: E501

                u_mix = align_to_reference_hemisphere(u_mix, u_clean)
                u_mwf = align_to_reference_hemisphere(u_mwf, u_clean)

                err_mix_deg = angle_between_unit_vectors_deg(u_mix, u_clean)
                err_mwf_deg = angle_between_unit_vectors_deg(u_mwf, u_clean)
            except Exception as e:
                warnings.warn(f"GCC-PHAT estimation failed for {leaf}: {e}")
                err_mix_deg = err_mwf_deg = float("nan")  # keep pipeline going

            metrics = {
                "folder": str(leaf),
                "outer_name": outer_name,
                "inner_name": leaf.name,
                "distance_m": distance,
                "snr_db_raw": snr_db_raw,
                "snr_db_plot": snr_plot,
                "sr": res["sr"],
                "Q_in": res["Q_in"],
                "Q_out": res["Q_out"],
                "q0": res["q0"],
                "mean_theta_full": res["mean_theta_full"],
                "mean_theta_200_1500": res["mean_theta_200_1500"],
                "gccphat_err_mix_deg": err_mix_deg,
                "gccphat_err_mwf_deg": err_mwf_deg,
            }
            with open(PER_FILE_DIR / f"{outer_name}_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            records.append(metrics)

            if not math.isnan(snr_plot):
                if snr_plot not in snr_to_curves:
                    snr_to_curves[snr_plot] = []
                    snr_to_freqs[snr_plot] = freqs
                    snr_to_full_means[snr_plot] = []
                    snr_to_band_means[snr_plot] = []
                ref_f = snr_to_freqs[snr_plot]
                if len(freqs) != len(ref_f) or not np.allclose(freqs, ref_f):
                    theta = np.interp(ref_f, freqs, theta)
                    freqs = ref_f
                snr_to_curves[snr_plot].append(theta)
                snr_to_full_means[snr_plot].append(res["mean_theta_full"])
                snr_to_band_means[snr_plot].append(res["mean_theta_200_1500"])

            gcc_rows.append({
                "folder": str(leaf),
                "outer_name": outer_name,
                "inner_name": leaf.name,
                "distance_m": distance,
                "snr_db_raw": snr_db_raw,
                "snr_db_plot": snr_plot,
                "gccphat_err_mix_deg": err_mix_deg,
                "gccphat_err_mwf_deg": err_mwf_deg,
            })
        else:
            warnings.warn(f"Skipping {leaf}: reason={res.get('reason','unknown')}")

    if not records:
        print("No valid results.")
        return

    df_all = pd.DataFrame(records)
    df_all.to_csv(SUM_DIR / "all_files_metrics.csv", index=False)

    agg_dist = (
        df_all.groupby("distance_m")
        .agg(
            n_files=("mean_theta_full", "count"),
            mean_theta_full=("mean_theta_full", "mean"),
            std_theta_full=("mean_theta_full", "std"),
            mean_theta_200_1500=("mean_theta_200_1500", "mean"),
            std_theta_200_1500=("mean_theta_200_1500", "std"),
        )
        .reset_index()
        .sort_values("distance_m")
    )
    agg_dist.to_csv(SUM_DIR / "summary_by_distance.csv", index=False)

    agg_snr = (
        df_all.groupby("snr_db_plot")
        .agg(
            n_files=("mean_theta_200_1500", "count"),
            mean_theta_200_1500=("mean_theta_200_1500", "mean"),
            std_theta_200_1500=("mean_theta_200_1500", "std"),
            mean_theta_full=("mean_theta_full", "mean"),
            std_theta_full=("mean_theta_full", "std"),
        )
        .reset_index()
        .sort_values("snr_db_plot")
    )
    agg_snr.to_csv(SUM_DIR / "summary_by_snr.csv", index=False)

    # --- Mean curves by SNR ---
    if snr_to_curves:
        snrplot_to_mean_curve: Dict[float, np.ndarray] = {}
        rows_curves = []
        ref_f_any = None

        for snr_plot, curves in snr_to_curves.items():
            ref_f = snr_to_freqs[snr_plot]
            if ref_f_any is None:
                ref_f_any = ref_f
            M = np.vstack(curves)
            mean_curve = np.mean(M, axis=0)
            snrplot_to_mean_curve[snr_plot] = mean_curve
            for f_val, th_val in zip(ref_f, mean_curve):
                rows_curves.append({"snr_db_plot": snr_plot, "freq_hz": float(f_val), "theta_rad": float(th_val)})

        pd.DataFrame(rows_curves).sort_values(["snr_db_plot", "freq_hz"]).to_csv(
            SUM_DIR / "snr_avg_curves.csv", index=False
        )

        make_fig2_selected_lines(
            ref_f_any,
            snrplot_to_mean_curve,
            TARGET_SNR_LIST,
            SUM_DIR / "fig2_full_theta_vs_freq_selectedSNR.png",
            SUM_DIR / "fig2_band_theta_vs_freq_selectedSNR.png",
        )

        snr_vals = sorted(snr_to_curves.keys())
        avg_full = [float(np.mean(snr_to_full_means[s])) for s in snr_vals]
        avg_band = [float(np.mean(snr_to_band_means[s])) for s in snr_vals]

        make_fig3_avgTheta_vs_SNR(
            snr_vals, avg_full, avg_band,
            SUM_DIR / "fig3_full_avgTheta_vs_SNR.png",
            SUM_DIR / "fig3_band_avgTheta_vs_SNR.png",
        )

    # --- GCC–PHAT error plots per-file ---
    if gcc_rows:
        df_gcc = pd.DataFrame(gcc_rows)
        df_gcc.to_csv(SUM_DIR / "gccphat_per_file_errors.csv", index=False)

        plt.figure(figsize=(8.4, 5.4))
        df_plot = df_gcc[["snr_db_plot", "gccphat_err_mix_deg"]].dropna()
        x = df_plot["snr_db_plot"].values
        y = df_plot["gccphat_err_mix_deg"].values
        order = np.argsort(x)
        plt.plot(x[order], y[order], marker='o', linewidth=1.8)
        plt.xlabel("Input SNR (dB)")
        plt.ylabel("Angle error (deg)")
        plt.title("GCC-PHAT angle error vs input SNR (per file, line plot)")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.savefig(SUM_DIR / "fig_gccphat_error_vs_snr_mix.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8.4, 5.4))
        df_plot = df_gcc[["snr_db_plot", "gccphat_err_mwf_deg"]].dropna()
        x = df_plot["snr_db_plot"].values
        y = df_plot["gccphat_err_mwf_deg"].values
        order = np.argsort(x)
        plt.plot(x[order], y[order], marker='o', linewidth=1.8)
        plt.xlabel("Input SNR (dB)")
        plt.ylabel("Angle error (deg)")
        plt.title("GCC-PHAT$_{MWF}$ angle error vs input SNR (per file, line plot)")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.savefig(SUM_DIR / "fig_gccphat_error_vs_snr_mwf.png", dpi=300)
        plt.close()

        plt.figure(figsize=(9.2, 5.6))
        df_mix = df_gcc[["snr_db_plot", "gccphat_err_mix_deg"]].dropna()
        df_mwf = df_gcc[["snr_db_plot", "gccphat_err_mwf_deg"]].dropna()
        xm, ym = df_mix["snr_db_plot"].values, df_mix["gccphat_err_mix_deg"].values
        xo, yo = df_mwf["snr_db_plot"].values, df_mwf["gccphat_err_mwf_deg"].values
        om, oo = np.argsort(xm), np.argsort(xo)
        plt.plot(xm[om], ym[om], marker='o', linewidth=1.8, label="GCC-PHAT (mixture)")
        plt.plot(xo[oo], yo[oo], marker='x', linewidth=1.8, label="GCC-PHAT (output)")
        plt.xlabel("Input SNR (dB)")
        plt.ylabel("Angle error (deg)")
        plt.title("Angle error vs input SNR")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(SUM_DIR / "fig_gccphat_error_vs_snr_both.png", dpi=300)
        plt.close()

    # --- Parameter dump ---
    from .config import (N_FFT, HOP, WINDOW, BAND_LOW_HZ, BAND_HIGH_HZ, PLOT_FREQ_MAX_HZ, CLEAN_ENERGY_THRESH_DB,
                         MIN_NOISE_FRAC)
    from .geometry import get_mic_geometry_matrix
    with open(SUM_DIR / "example_params.txt", "w", encoding="utf-8") as f:
        f.write("\n".join([
            f"DATA_DIR={data_dir}",
            f"OUT_DIR={out_dir}",
            f"N_FFT={N_FFT}",
            f"HOP={HOP}",
            f"WINDOW={WINDOW}",
            f"BAND_LOW_HZ={BAND_LOW_HZ}",
            f"BAND_HIGH_HZ={BAND_HIGH_HZ}",
            f"PLOT_FREQ_MAX_HZ={PLOT_FREQ_MAX_HZ}",
            f"CLEAN_ENERGY_THRESH_DB={CLEAN_ENERGY_THRESH_DB}",
            f"MIN_NOISE_FRAC={MIN_NOISE_FRAC}",
            f"MIC_POSITIONS (m) rows=[Mic2, Mic3, Mic4]:\n{get_mic_geometry_matrix()}",
        ]))
