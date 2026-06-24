#!/usr/bin/env python3
"""
run_pipeline.py
================
Scrape Mathlib4 -> embed statements -> train models -> print results.

Usage:
  python run_pipeline.py                    # full run
  python run_pipeline.py --only_train       # skip scraping (data exists)
  python run_pipeline.py --embed_mode api   # use OpenAI embeddings (set OPENAI_API_KEY)
  python run_pipeline.py --n_theorems 500   # larger dataset
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Auto-install numpy if missing
try:
    import numpy
except ModuleNotFoundError:
    print("numpy not found — installing...")
    for cmd in [
        [sys.executable, "-m", "pip", "install", "numpy"],
        [sys.executable, "-m", "pip", "install", "--user", "numpy"],
        [sys.executable, "-m", "pip", "install", "--break-system-packages", "numpy"],
        ["pip3", "install", "--user", "numpy"],
    ]:
        try:
            subprocess.check_call(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            print(f"  installed via: {' '.join(cmd)}")
            break
        except Exception:
            continue
    else:
        print("\nCould not install numpy automatically. Run one of:")
        print("  pip3 install --user numpy")
        print("  pip3 install --break-system-packages numpy")
        sys.exit(1)
    import importlib
    importlib.invalidate_caches()
    print()

from scrape_mathlib import scrape
from embed_statements import embed
from train_predictor import main as train_main

DATA_PATH   = "data/mathlib_raw.jsonl"
EMBED_PATH  = "data/embeddings.npz"
RESULTS_DIR = "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_theorems", type=int, default=400)
    parser.add_argument("--embed_mode", default="tfidf", choices=["tfidf", "api", "bow"])
    parser.add_argument("--only_train", action="store_true")
    parser.add_argument("--only_embed", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║      Lean Proof Difficulty Predictor                         ║
║      Can theorem statements predict proof complexity?        ║
╚══════════════════════════════════════════════════════════════╝
""")

    Path("data").mkdir(exist_ok=True)
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    if not args.only_train and not args.only_embed:
        if Path(DATA_PATH).exists():
            n = sum(1 for _ in open(DATA_PATH))
            print(f"[1/3] Dataset exists ({n} theorems). Delete {DATA_PATH} to re-scrape.\n")
        else:
            print("[1/3] Scraping Mathlib4...")
            scrape(DATA_PATH, max_per_file=args.n_theorems, seed=args.seed)
            print()

    if args.only_embed or not args.only_train:
        print(f"[2/3] Embedding statements (mode={args.embed_mode})...")
        embed(DATA_PATH, EMBED_PATH, mode=args.embed_mode)
        print()

    print("[3/3] Training models + ablation study...")
    sys.argv = [
        "train_predictor.py",
        f"--embeddings={EMBED_PATH}",
        f"--data={DATA_PATH}",
        f"--output_dir={RESULTS_DIR}",
        f"--seed={args.seed}",
    ]
    train_main()

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()