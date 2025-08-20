from pathlib import Path
import numpy as np

THIS_FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_FILE_DIR.parent
DATA_DIR = PROJECT_ROOT / "Spatial-U-Net"   # This is just the project that I'm working on
OUT_DIR = PROJECT_ROOT / "analysis_spatial_cues"
PER_FILE_DIR = OUT_DIR / "per_file"
SUM_DIR = OUT_DIR / "summaries"

# STFT params
N_FFT = 2048
HOP = 512
WINDOW = "hann"

# Analysis band for plotting/statistics
BAND_LOW_HZ = 200.0
BAND_HIGH_HZ = 1500.0

# Plot controls
PLOT_FREQ_MAX_HZ = 10000.0  # full band upper limit for line plots
PLOT_MAX_THETA_RAD = None   # e.g., set to 1.0 to clamp y-axis

# Numerical safety
EPS_REG = 1e-6
MIN_NOISE_FRAC = 0.10
CLEAN_ENERGY_THRESH_DB = -40.0

# Display SNR lines to highlight in figures
TARGET_SNR_LIST = [-30.0, -25.0, -20.0, -15.0, -10.0]

# GCC–PHAT
SPEED_OF_SOUND = 343.0
MIC_CHANNEL_IDXS = (0, 1, 2)   # channels used to form pairs
RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)
