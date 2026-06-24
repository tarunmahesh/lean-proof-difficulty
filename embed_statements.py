"""
embed_statements.py
====================
Encodes theorem statements into dense embeddings for the difficulty predictor.

Two embedding modes:
  1. "api"    -- calls OpenAI text-embedding-3-small (best quality, requires key)
  2. "tfidf"  -- TF-IDF over Lean tokens (no API, reproducible baseline)
  3. "bow"    -- Simple bag-of-tactics (interpretable ablation)

The embeddings are written to data/embeddings.npz alongside a manifest.

Usage:
  python embed_statements.py --mode tfidf --input data/mathlib_raw.jsonl
  python embed_statements.py --mode api   --input data/mathlib_raw.jsonl
"""

import os
import re
import json
import math
import argparse
import urllib.request
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Optional


# ── Lean token vocabulary ─────────────────────────────────────────────────────

LEAN_KEYWORDS = [
    "theorem", "lemma", "def", "instance", "structure", "class", "where",
    "forall", "exists", "fun", "let", "have", "show", "from", "by",
    "match", "with", "if", "then", "else", "do", "return",
    # Types
    "Nat", "Int", "Real", "Rat", "Bool", "Prop", "Type", "Sort",
    "List", "Finset", "Set", "Multiset", "Array", "Option", "Prod",
    "Sum", "Sigma", "Subtype", "Quotient", "Equiv",
    # Algebra
    "Group", "Ring", "Field", "Module", "Algebra", "Monoid", "Semiring",
    "OrderedField", "LinearOrder", "CommRing", "CommGroup",
    # Tactics (appear in statements via type names)
    "le", "lt", "ge", "gt", "eq", "ne", "dvd", "Coprime",
    # Unicode math (ASCII-ized for tokenization)
    "forall", "exists", "not", "and", "or", "iff", "implies",
    "add", "mul", "sub", "div", "pow", "mod", "abs", "max", "min",
    "sup", "inf", "sum", "prod",
]

UNICODE_MAP = {
    "∀": "forall", "∃": "exists", "¬": "not", "∧": "and", "∨": "or",
    "↔": "iff", "→": "implies", "←": "leftarrow",
    "≤": "le", "≥": "ge", "≠": "ne", "≡": "equiv",
    "∈": "mem", "∉": "notmem", "⊆": "subset", "⊂": "ssubset",
    "∪": "union", "∩": "inter", "∅": "empty",
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "μ": "mu", "φ": "phi", "ψ": "psi",
    "·": "cdot", "×": "times", "⊕": "oplus", "⊗": "otimes",
}


def normalize_statement(s: str) -> str:
    """Normalize unicode math symbols and clean whitespace."""
    for sym, name in UNICODE_MAP.items():
        s = s.replace(sym, f" {name} ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> list[str]:
    """Simple Lean-aware tokenizer."""
    s = normalize_statement(s)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_']*|[0-9]+|[+\-*/^<>=!|&%]", s)
    return [t.lower() for t in tokens]


# ── TF-IDF Embedding ──────────────────────────────────────────────────────────

class TFIDFEmbedder:
    """
    Lean-aware TF-IDF over theorem statement tokens.
    Vocabulary built from corpus; produces dense L2-normalized vectors.
    """

    def __init__(self, max_features: int = 2048, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = None

    def fit(self, statements: list[str]) -> "TFIDFEmbedder":
        n = len(statements)
        df = Counter()
        tokenized = []
        for s in statements:
            toks = set(tokenize(s))
            tokenized.append(toks)
            for t in toks:
                df[t] += 1

        # Filter by min_df, sort by df desc, truncate
        vocab_terms = sorted(
            [t for t, c in df.items() if c >= self.min_df],
            key=lambda t: -df[t]
        )[:self.max_features]

        self.vocab = {t: i for i, t in enumerate(vocab_terms)}
        # IDF with smoothing
        self.idf = np.array([
            math.log((n + 1) / (df[t] + 1)) + 1.0
            for t in vocab_terms
        ])
        return self

    def transform(self, statements: list[str]) -> np.ndarray:
        V = len(self.vocab)
        X = np.zeros((len(statements), V), dtype=np.float32)
        for i, s in enumerate(statements):
            toks = tokenize(s)
            tf = Counter(toks)
            for tok, cnt in tf.items():
                if tok in self.vocab:
                    j = self.vocab[tok]
                    X[i, j] = (1 + math.log(cnt)) * self.idf[j]
        # L2 normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return X / norms

    def fit_transform(self, statements: list[str]) -> np.ndarray:
        return self.fit(statements).transform(statements)


# ── Bag-of-Tactics Embedding ──────────────────────────────────────────────────

TACTIC_VOCAB = [
    "simp", "ring", "omega", "linarith", "norm_num", "exact", "apply",
    "intro", "intros", "constructor", "left", "right", "rfl", "trivial",
    "tauto", "contradiction", "assumption", "aesop", "decide", "positivity",
    "have", "obtain", "use", "refine", "rw", "rewrite", "calc",
    "induction", "cases", "rcases", "ext", "funext", "congr", "conv",
    "push_neg", "norm_cast", "field_simp", "ring_nf", "simp_rw",
    "gcongr", "nlinarith", "polyrith", "fin_cases", "interval_cases",
    "contrapose", "by_contra", "by_cases", "suffices", "show", "change",
    "unfold", "group", "abel",
]

TACTIC_INDEX = {t: i for i, t in enumerate(TACTIC_VOCAB)}


def bag_of_tactics(proof_body: str) -> np.ndarray:
    vec = np.zeros(len(TACTIC_VOCAB), dtype=np.float32)
    tokens = re.findall(r"\b\w+\b", proof_body)
    for tok in tokens:
        if tok in TACTIC_INDEX:
            vec[TACTIC_INDEX[tok]] += 1
    # L2 normalize
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ── OpenAI API Embedding ──────────────────────────────────────────────────────

def embed_via_api(
    statements: list[str],
    api_key: Optional[str] = None,
    model: str = "text-embedding-3-small",
    batch_size: int = 64,
) -> np.ndarray:
    import time
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY not set. Use --mode tfidf for no-API embedding.")

    all_embeddings = []
    for i in range(0, len(statements), batch_size):
        batch = statements[i:i+batch_size]
        # Normalize unicode for API
        batch_clean = [normalize_statement(s)[:8000] for s in batch]
        payload = json.dumps({
            "model": model,
            "input": batch_clean,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        batch_embs = [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]
        all_embeddings.extend(batch_embs)
        print(f"  Embedded {min(i+batch_size, len(statements))}/{len(statements)}")
        time.sleep(0.1)

    return np.array(all_embeddings, dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def embed(
    input_path: str,
    output_path: str,
    mode: str = "tfidf",
    api_key: Optional[str] = None,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        records = [json.loads(l) for l in f]

    ids = [r["id"] for r in records]
    statements = [r["full_statement"] for r in records]
    proofs = [r["proof_body"] for r in records]

    print(f"Embedding {len(records)} theorems with mode='{mode}'...")

    if mode == "api":
        X_stmt = embed_via_api(statements, api_key=api_key)
        X_tac = np.stack([bag_of_tactics(p) for p in proofs])
        X = np.concatenate([X_stmt, X_tac], axis=1)
        embed_dim = X_stmt.shape[1]

    elif mode == "tfidf":
        embedder = TFIDFEmbedder(max_features=1024, min_df=2)
        X_stmt = embedder.fit_transform(statements)
        X_tac = np.stack([bag_of_tactics(p) for p in proofs])
        X = np.concatenate([X_stmt, X_tac], axis=1)
        embed_dim = X_stmt.shape[1]

        # Save vocab for interpretability
        vocab_path = Path(output_path).parent / "tfidf_vocab.json"
        with open(vocab_path, "w") as f:
            json.dump({
                "vocab": {t: int(i) for t, i in embedder.vocab.items()},
                "idf": embedder.idf.tolist(),
            }, f)
        print(f"  TF-IDF vocab ({len(embedder.vocab)} terms) saved to {vocab_path}")

    elif mode == "bow":
        X = np.stack([bag_of_tactics(p) for p in proofs])
        embed_dim = 0

    else:
        raise ValueError(f"Unknown mode: {mode}")

    np.savez_compressed(
        output_path,
        X=X,
        ids=np.array(ids),
        embed_dim=embed_dim,
        mode=mode,
    )
    print(f"Embeddings saved: shape={X.shape} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/mathlib_raw.jsonl")
    parser.add_argument("--output", default="data/embeddings.npz")
    parser.add_argument("--mode", default="tfidf", choices=["tfidf", "api", "bow"])
    parser.add_argument("--api_key", default=None)
    args = parser.parse_args()
    embed(args.input, args.output, args.mode, args.api_key)
