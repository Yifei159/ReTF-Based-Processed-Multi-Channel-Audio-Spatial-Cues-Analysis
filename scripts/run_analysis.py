import argparse
from pathlib import Path
from retf.analysis import run_analysis

def parse_args():
    p = argparse.ArgumentParser(description="ReTF spatial cue preservation analysis")
    p.add_argument("--data", type=Path, default=None, help="Root folder that contains leaf dirs with 3 WAVs.")
    p.add_argument("--out", type=Path, default=None, help="Output root folder for figures & CSVs.")
    return p.parse_args()

def main():
    args = parse_args()
    if args.data is None and args.out is None:
        # fall back to package defaults
        from retf.config import DATA_DIR, OUT_DIR
        run_analysis(DATA_DIR, OUT_DIR)
    else:
        from retf.config import DATA_DIR, OUT_DIR
        data_dir = args.data if args.data is not None else DATA_DIR
        out_dir = args.out if args.out is not None else OUT_DIR
        run_analysis(data_dir, out_dir)

if __name__ == "__main__":
    main()
