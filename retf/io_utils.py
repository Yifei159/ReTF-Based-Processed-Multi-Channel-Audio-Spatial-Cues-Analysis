import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple
import numpy as np
import soundfile as sf

RE_DISTANCE = re.compile(r"D(\d+(?:\.\d+)?)m", re.IGNORECASE)
RE_SNR = re.compile(r"SNR[-_]?(-?\d+(?:\.\d+)?)", re.IGNORECASE)

def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=True)
    return x.astype(np.float64), sr

def find_leaf_dirs(root: Path) -> List[Path]:
    leaves = []
    for dirpath, _, filenames in os.walk(root):
        fn_set = set(fn.lower() for fn in filenames)
        need = {"mixture_input.wav", "vocals_pred.wav", "vocals_clean.wav"}
        if need.issubset(fn_set):
            leaves.append(Path(dirpath))
    return leaves

def extract_distance_from_parts(parts: Iterable[str]) -> float:
    for name in parts:
        m = RE_DISTANCE.search(name)
        if m:
            return float(m.group(1))
    return float("nan")

def extract_snr_from_parts(parts: Iterable[str]) -> float:
    for name in parts:
        m = RE_SNR.search(name)
        if m:
            return float(m.group(1))
    return float("nan")
