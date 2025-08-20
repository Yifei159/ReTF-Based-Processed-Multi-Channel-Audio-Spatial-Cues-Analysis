# ReTF Based Processed Multi-Channel Audio Spatial Cues Analysis

This repository reproduces and organizes the experimental workflow of "Drone Audition: Analysis of the Preservation of
Spatial Cues in Multichannel Wiener Filtering Session: Drone Audition - Listening with Drones" with python.

> This implementation follows the methodology in the original paper: estimating the Relative Transfer Function (ReTF) between input and output, measuring spatial cue preservation by the Hermitian angle Θ(f), and validating localization consistency using GCC‑PHAT (the original paper demonstrates how fidelity aids localization).

## Directory Structure
```text
ReTF-Based-Processed-Multi-Channel-Audio-Spatial-Cues-Analysis/
├── retf/
│   ├── __init__.py
│   ├── analysis.py       # Top-level pipeline: dataset traversal, metric computation, plotting, summary
│   ├── config.py         # All hyperparameters/constants and default paths
│   ├── geometry.py       # Microphone array coordinates (example: 3-mic layout)
│   ├── gcc_phat.py       # GCC‑PHAT and DOA estimation, error comparison
│   ├── io_utils.py       # Data loading, leaf directory discovery, SNR/distance parsing
│   ├── plots.py          # Unified Matplotlib plotting functions
│   ├── retf.py           # ReTF (covariance whitening) and Hermitian angle computation
│   └── stft_utils.py     # STFT and framewise energy/alignment, etc.
├── scripts/
│   └── run_analysis.py   # Command-line entry point
└── README.md
```

## Dependencies
- Python ≥ 3.9
- numpy, scipy, soundfile, pandas, matplotlib, tqdm

Install via:
```bash
pip install -r requirements.txt
```

## Data Organization Requirements
- Recursively search for **leaf directories** under the root specified by `--data`. Each leaf directory must contain three WAV files (case-insensitive filenames):
  - `mixture_input.wav` (noisy multichannel input)
  - `vocals_pred.wav` (the denoised output)
  - `vocals_clean.wav` (target speech)
- The code will attempt to parse distance `D<meters>m` and SNR `SNR<dB>` tags from directory names (e.g., `.../D1.0m/.../SNR-20/...`).

## Running
```bash
## Method 1: Use package default paths (DATA_DIR and OUT_DIR in retf/config.py)
python -m scripts.run_analysis

## Method 2: Explicitly specify data and output directories
python -m scripts.run_analysis --data ./Conformer-GSE --out ./analysis_output
```

After running, the following will be generated in the `out/` directory:
- `per_file/`: Per-sample curves and JSON metrics
  - `<outer_name>_theta_curve.csv|.png`
  - `<outer_name>_metrics.json`
- `summaries/`: Aggregated statistics and summary plots
  - `all_files_metrics.csv`, `summary_by_distance.csv`, `summary_by_snr.csv`
  - `snr_avg_curves.csv`
  - `fig2_*theta_vs_freq*.png`, `fig3_*avgTheta_vs_SNR*.png`
  - `fig_gccphat_error_vs_snr_*.png`
  - `example_params.txt` (snapshot of key parameters)

## Changes/Fixes Compared to the Original Script
- **Modularization & Documentation**: Computation, plotting, DOA, and IO are separated; all functions include concise docstrings.
- **Sample Rate Consistency Check**: `compute_theta_for_file` checks that `mixture/output/clean` all have the same sample rate, otherwise raises an error (in the original script, `sr` was undefined).
- **GCC‑PHAT**: Original algorithm flow retained, encapsulated in `gcc_phat.py`, with error analysis aligned to ReTF results.
- **Robustness**: If noise frames are insufficient, at least `MIN_NOISE_FRAC` is always kept as noise frames; covariance matrices are regularized with a trace-related diagonal term.
- **Configurable**: All thresholds, FFT parameters, frequency bands, etc. are centralized in `config.py`.

### Run
```bash
python -m scripts.run_analysis --data ./Conformer-GSE --out ./analysis_output
```

### Outputs
Per‑file curves/metrics, aggregated CSVs, and summary plots replicating the figures described
in the original workflow (Θ(f) vs frequency; averaged Θ vs SNR; GCC‑PHAT error vs SNR).

## Reference

Manamperi, W. N., Abhayapala, T. D., Samarasinghe, P. N., & Zhang, J. (2024). Drone audition: Audio signal enhancement from drone embedded microphones using multichannel Wiener filtering and Gaussian‑mixture based post‑filtering. Applied Acoustics, 216, 109818.
