"""
train_predictor.py
===================
Trains proof difficulty predictors and runs the ablation study.

Task: Given only the theorem statement, predict proof_length (regression)
or difficulty_bucket (classification: easy / medium / hard).

Models:
  - Linear regression / Ridge (interpretable baseline)
  - Random forest (non-linear, handles feature interactions)
  - Gradient boosting (best performance expected)
  - MLP (small neural net on embeddings)

Feature sets (ablation):
  A. Syntactic only    -- statement token count, quantifier depth, etc.
  B. Bag-of-tactics    -- tactic frequencies from embedding
  C. TF-IDF statement  -- semantic statement embedding
  D. A + B             -- syntactic + BoT
  E. A + C             -- syntactic + TF-IDF (main comparison)
  F. A + B + C         -- full feature set

Outputs:
  results/model_results.json    -- metrics per model per feature set
  results/feature_importance.json
  results/predictions_test.jsonl -- per-sample predictions on test set

Usage:
  python train_predictor.py --embeddings data/embeddings.npz \\
                            --data data/mathlib_raw.jsonl \\
                            --task regression
"""

import json
import math
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# ── Pure-numpy ML primitives ──────────────────────────────────────────────────

def train_test_split(X, y, ids, test_ratio=0.2, seed=42):
    rng = random.Random(seed)
    n = len(y)
    indices = list(range(n))
    rng.shuffle(indices)
    split = int(n * (1 - test_ratio))
    tr, te = indices[:split], indices[split:]
    return (X[tr], y[tr], [ids[i] for i in tr],
            X[te], y[te], [ids[i] for i in te])


def standardize(X_tr, X_te):
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0) + 1e-8
    return (X_tr - mu) / sigma, (X_te - mu) / sigma, mu, sigma


# ── Ridge Regression ──────────────────────────────────────────────────────────

def ridge_fit(X, y, alpha=1.0):
    n, d = X.shape
    A = X.T @ X + alpha * np.eye(d)
    b = X.T @ y
    # Solve via Cholesky (X is likely well-conditioned after standardization)
    try:
        L = np.linalg.cholesky(A)
        w = np.linalg.solve(L.T, np.linalg.solve(L, b))
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w


def ridge_predict(X, w):
    return X @ w


# ── Random Forest (numpy, no sklearn) ────────────────────────────────────────

class DecisionTree:
    def __init__(self, max_depth=8, min_samples_split=5, max_features=None, seed=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.tree = None
        self._rng = np.random.RandomState(seed)

    def fit(self, X, y):
        self.tree = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        n, d = X.shape
        if depth >= self.max_depth or n < self.min_samples_split:
            return {"leaf": True, "value": float(np.mean(y))}

        max_feats = self.max_features or max(1, int(math.sqrt(d)))
        feat_idx = self._rng.choice(d, size=min(max_feats, d), replace=False)

        best_feat, best_thresh, best_score = None, None, float("inf")
        for f in feat_idx:
            vals = np.unique(X[:, f])
            if len(vals) < 2:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2
            # Sample thresholds for speed on large d
            if len(thresholds) > 20:
                thresholds = thresholds[self._rng.choice(len(thresholds), 20, replace=False)]
            for t in thresholds:
                left = y[X[:, f] <= t]
                right = y[X[:, f] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                score = len(left) * np.var(left) + len(right) * np.var(right)
                if score < best_score:
                    best_score = score
                    best_feat = f
                    best_thresh = t

        if best_feat is None:
            return {"leaf": True, "value": float(np.mean(y))}

        mask = X[:, best_feat] <= best_thresh
        return {
            "leaf": False,
            "feat": best_feat,
            "thresh": best_thresh,
            "left": self._build(X[mask], y[mask], depth + 1),
            "right": self._build(X[~mask], y[~mask], depth + 1),
        }

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feat"]] <= node["thresh"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


class RandomForest:
    def __init__(self, n_trees=80, max_depth=8, min_samples_split=5,
                 max_features=None, seed=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        n = len(y)
        self.trees = []
        for i in range(self.n_trees):
            idx = rng.choice(n, n, replace=True)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                seed=self.seed + i,
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict(self, X):
        preds = np.stack([t.predict(X) for t in self.trees])
        return preds.mean(axis=0)

    def feature_importances(self, X, y):
        """Permutation importance on training set."""
        baseline = rmse(self.predict(X), y)
        importances = []
        rng = np.random.RandomState(self.seed)
        for j in range(X.shape[1]):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            importances.append(rmse(self.predict(X_perm), y) - baseline)
        return np.array(importances)


# ── Gradient Boosting (simple GBRT) ──────────────────────────────────────────

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=4,
                 min_samples_split=5, seed=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = seed
        self.trees = []
        self.init_pred = 0.0

    def fit(self, X, y):
        self.init_pred = float(np.mean(y))
        residuals = y - self.init_pred
        rng = np.random.RandomState(self.seed)
        n, d = X.shape

        for i in range(self.n_estimators):
            # Subsample rows (stochastic GB)
            idx = rng.choice(n, int(0.8 * n), replace=False)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max(1, int(math.sqrt(d))),
                seed=self.seed + i,
            )
            tree.fit(X[idx], residuals[idx])
            update = tree.predict(X)
            residuals -= self.learning_rate * update
            self.trees.append(tree)
        return self

    def predict(self, X):
        pred = np.full(len(X), self.init_pred)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred


# ── MLP (simple 2-layer) ──────────────────────────────────────────────────────

class MLP:
    def __init__(self, hidden=128, lr=1e-3, epochs=200, seed=42):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.W1 = self.b1 = self.W2 = self.b2 = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        d = X.shape[1]
        scale = math.sqrt(2.0 / d)
        self.W1 = rng.randn(d, self.hidden) * scale
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.randn(self.hidden, 1) * math.sqrt(2.0 / self.hidden)
        self.b2 = np.zeros(1)
        y = y.reshape(-1, 1)
        n = len(X)
        best_loss = float("inf")
        best_params = None

        for epoch in range(self.epochs):
            # Forward
            h = np.maximum(0, X @ self.W1 + self.b1)  # ReLU
            out = h @ self.W2 + self.b2
            loss = np.mean((out - y) ** 2)

            if loss < best_loss:
                best_loss = loss
                best_params = (self.W1.copy(), self.b1.copy(),
                               self.W2.copy(), self.b2.copy())

            # Backward
            d_out = 2 * (out - y) / n
            dW2 = h.T @ d_out
            db2 = d_out.sum(axis=0)
            d_h = d_out @ self.W2.T * (h > 0)
            dW1 = X.T @ d_h
            db1 = d_h.sum(axis=0)

            # Gradient clipping to prevent exploding gradients on skewed targets
            clip = 1.0
            for g in [dW2, db2, dW1, db1]:
                np.clip(g, -clip, clip, out=g)
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

        # Restore best
        self.W1, self.b1, self.W2, self.b2 = best_params
        return self

    def predict(self, X):
        h = np.maximum(0, X @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2).flatten()


# ── Metrics ───────────────────────────────────────────────────────────────────

def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def r2(pred, true):
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))


def spearman_r(pred, true):
    """Spearman rank correlation (no scipy)."""
    n = len(pred)
    rank_pred = np.argsort(np.argsort(pred)).astype(float)
    rank_true = np.argsort(np.argsort(true)).astype(float)
    d = rank_pred - rank_true
    return float(1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1)))


def log_rmse(pred, true):
    """RMSE in log space — robust to heavy-tailed proof length distribution."""
    return float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(true)) ** 2)))


def bucket_accuracy_q1q4(pred, true, q25, q75):
    """
    Q1-vs-Q4 binary accuracy: only evaluate on easy (<=Q25) and hard (>=Q75).
    Drops ambiguous middle 50%, making the task well-separated and meaningful.
    """
    mask = (true <= q25) | (true >= q75)
    if mask.sum() == 0:
        return 0.0
    pred_m, true_m = pred[mask], true[mask]
    pred_hard = pred_m >= q75
    true_hard = true_m >= q75
    return float(np.mean(pred_hard == true_hard))


# ── Feature set construction ──────────────────────────────────────────────────

# Core syntactic features from statement text
SYNTACTIC_FEATURES = [
    "stmt_token_count", "has_forall", "has_exists", "has_iff", "has_neg",
    "quantifier_depth",
]

# Dependency graph features (computed post-scrape)
DEP_FEATURES = ["dep_depth", "dep_count"]

# All known domains (must match scrape_mathlib.DOMAIN_MAP values)
ALL_DOMAINS = [
    "algebra", "analysis", "calculus", "category_theory", "combinatorics",
    "data_structures", "geometry", "linear_algebra", "logic", "measure_theory",
    "number_theory", "order_theory", "set_theory", "topology", "other",
]

def domain_onehot(records: list[dict]) -> tuple[np.ndarray, list[str]]:
    """One-hot encode domain labels. Unknown domains map to 'other'."""
    feat_names = [f"domain_{d}" for d in ALL_DOMAINS]
    X = np.zeros((len(records), len(ALL_DOMAINS)), dtype=np.float32)
    for i, r in enumerate(records):
        dom = r.get("domain", "other")
        if dom not in ALL_DOMAINS:
            dom = "other"
        X[i, ALL_DOMAINS.index(dom)] = 1.0
    return X, feat_names


def build_feature_sets(records, X_embed, embed_dim):
    """
    Returns dict of feature_set_name -> (feature_matrix, feature_names)

    Feature sets:
      A  - Syntactic statement features only
      B  - Bag-of-tactics (oracle: uses proof)
      C  - TF-IDF statement embedding
      D  - Syn + BoT
      E  - Syn + TF-IDF
      F  - Full (A+B+C)
      G  - Syn + Domain one-hot
      H  - Syn + Domain + Dep features (dep_depth, dep_count)
      I  - Full + Domain + Dep (kitchen sink)
    """
    n = len(records)

    syn_vals = np.array([[r[f] for f in SYNTACTIC_FEATURES] for r in records], dtype=np.float32)
    X_domain, domain_names = domain_onehot(records)

    # Dependency features — structural difficulty signal
    dep_vals = np.zeros((n, len(DEP_FEATURES)), dtype=np.float32)
    for i, r in enumerate(records):
        for j, f in enumerate(DEP_FEATURES):
            dep_vals[i, j] = float(r.get(f, 0))

    if embed_dim > 0:
        X_stmt = X_embed[:, :embed_dim]
        X_bot = X_embed[:, embed_dim:]
    else:
        X_stmt = None
        X_bot = X_embed

    syn_dom = np.concatenate([syn_vals, X_domain], axis=1)
    syn_dom_names = SYNTACTIC_FEATURES + domain_names

    feature_sets = {
        "A_syntactic":    (syn_vals,  SYNTACTIC_FEATURES),
        "G_syn_domain":   (syn_dom,   syn_dom_names),
        "H_syn_dom_dep":  (
            np.concatenate([syn_dom, dep_vals], axis=1),
            syn_dom_names + DEP_FEATURES,
        ),
    }

    if X_bot is not None and X_bot.shape[1] > 0:
        bot_names = [f"tac_{i}" for i in range(X_bot.shape[1])]
        feature_sets["B_bag_of_tactics"] = (X_bot, bot_names)
        feature_sets["D_syn_plus_bot"] = (
            np.concatenate([syn_vals, X_bot], axis=1),
            SYNTACTIC_FEATURES + bot_names,
        )

    if X_stmt is not None:
        tfidf_names = [f"tfidf_{i}" for i in range(X_stmt.shape[1])]
        feature_sets["C_tfidf"] = (X_stmt, tfidf_names)
        feature_sets["E_syn_plus_tfidf"] = (
            np.concatenate([syn_vals, X_stmt], axis=1),
            SYNTACTIC_FEATURES + tfidf_names,
        )
        if X_bot is not None:
            bot_names = [f"tac_{i}" for i in range(X_bot.shape[1])]
            feature_sets["F_full"] = (
                np.concatenate([syn_vals, X_bot, X_stmt], axis=1),
                SYNTACTIC_FEATURES + bot_names + tfidf_names,
            )
            feature_sets["I_kitchen_sink"] = (
                np.concatenate([syn_dom, dep_vals, X_bot, X_stmt], axis=1),
                syn_dom_names + DEP_FEATURES + bot_names + tfidf_names,
            )

    return feature_sets


# ── Training loop ─────────────────────────────────────────────────────────────

def train_and_evaluate(records, X_embed, embed_dim, seed=42):
    ids = [r["id"] for r in records]
    y_raw = np.array([float(r["proof_length"]) for r in records])

    # Log-transform target: log(proof_length + 1)
    # Proof length is heavy-tailed (min=1, median=2, max=67); log scale
    # prevents large proofs from dominating the loss and stabilizes the MLP.
    y = np.log1p(y_raw)

    # Q1/Q4 bucketing on raw scale: only compare bottom 25% vs top 25%.
    # P33/P67 thresholds collapsed to 2 and 3 lines — not meaningfully separable.
    q25, q75 = np.percentile(y_raw, 25), np.percentile(y_raw, 75)
    p33, p67 = q25, q75  # keep variable names for return compat
    print(f"\nProof length (raw): min={y_raw.min():.0f} Q25={q25:.1f} "
          f"median={float(np.median(y_raw)):.1f} Q75={q75:.1f} max={y_raw.max():.0f}")
    print(f"Log-transformed: min={y.min():.2f} mean={y.mean():.2f} max={y.max():.2f}")
    print(f"Buckets: easy (raw<=Q25={q25:.0f}) vs hard (raw>=Q75={q75:.0f}), middle dropped")

    feature_sets = build_feature_sets(records, X_embed, embed_dim)
    all_results = {}
    all_predictions = {}

    for fs_name, (X_fs, feat_names) in feature_sets.items():
        print(f"\n{'='*55}")
        print(f"Feature set: {fs_name}  (dim={X_fs.shape[1]})")

        X_tr, y_tr, ids_tr, X_te, y_te, ids_te = train_test_split(
            X_fs, y, ids, test_ratio=0.2, seed=seed)

        X_tr_s, X_te_s, mu, sigma = standardize(X_tr, X_te)

        fs_results = {}

        models = {
            "ridge": (
                lambda: ridge_predict(X_te_s, ridge_fit(X_tr_s, y_tr, alpha=10.0)),
                "Ridge"
            ),
            "random_forest": (
                lambda: (lambda rf: rf.predict(X_te))(
                    RandomForest(n_trees=60, max_depth=7, seed=seed).fit(X_tr, y_tr)
                ),
                "Random Forest"
            ),
            "gradient_boosting": (
                lambda: (lambda gb: gb.predict(X_te))(
                    GradientBoostingRegressor(n_estimators=80, learning_rate=0.1,
                                             max_depth=4, seed=seed).fit(X_tr, y_tr)
                ),
                "Gradient Boosting"
            ),
        }
        # Only run MLP on reasonably sized feature sets (avoid OOM on large TF-IDF)
        if X_fs.shape[1] <= 200:
            models["mlp"] = (
                lambda: (lambda mlp: mlp.predict(X_te_s))(
                    MLP(hidden=64, lr=1e-3, epochs=400, seed=seed).fit(X_tr_s, y_tr)
                ),
                "MLP"
            )

        for model_key, (predict_fn, model_name) in models.items():
            print(f"  Training {model_name}...", end=" ", flush=True)
            try:
                pred_log = predict_fn()
                pred_log = np.clip(pred_log, 0, 10.0)  # cap before expm1 to avoid overflow
                # Back-transform to raw scale for RMSE/MAE reporting
                pred = np.expm1(pred_log)
                pred = np.clip(pred, 0, None)
                y_te_raw = np.expm1(y_te)
                metrics = {
                    "rmse": rmse(pred, y_te_raw),
                    "mae": mae(pred, y_te_raw),
                    "log_rmse": log_rmse(pred, y_te_raw),
                    "r2": r2(pred_log, y_te),
                    "spearman_r": spearman_r(pred_log, y_te),
                    "bucket_acc": bucket_accuracy_q1q4(pred, y_te_raw, q25, q75),
                }
                print(f"LogRMSE={metrics['log_rmse']:.3f} R2={metrics['r2']:.3f} "
                      f"Spearman={metrics['spearman_r']:.3f} Q1Q4Acc={metrics['bucket_acc']:.3f}")
                fs_results[model_key] = {
                    "model_name": model_name,
                    "metrics": metrics,
                    "n_train": len(y_tr),
                    "n_test": len(y_te),
                    "feature_dim": X_fs.shape[1],
                }
                if model_key == "gradient_boosting":
                    all_predictions[fs_name] = {
                        "ids": ids_te,
                        "true": y_te_raw.tolist(),
                        "pred": pred.tolist(),
                    }
            except Exception as e:
                print(f"FAILED: {e}")
                fs_results[model_key] = {"error": str(e)}

        all_results[fs_name] = fs_results

    return all_results, all_predictions, (q25, q75)


def extract_feature_importance(records, X_embed, embed_dim, seed=42):
    """
    Permutation importance on syntactic + domain + dep features using RF.
    Runs on log-transformed target. Returns list of (feature_name, importance).
    """
    y_raw = np.array([float(r["proof_length"]) for r in records])
    y = np.log1p(y_raw)

    X_domain, domain_names = domain_onehot(records)
    dep_vals = np.zeros((len(records), len(DEP_FEATURES)), dtype=np.float32)
    for i, r in enumerate(records):
        for j, f in enumerate(DEP_FEATURES):
            dep_vals[i, j] = float(r.get(f, 0))

    syn_vals = np.array([[r[f] for f in SYNTACTIC_FEATURES] for r in records], dtype=np.float32)
    X = np.concatenate([syn_vals, X_domain, dep_vals], axis=1)
    feat_names = SYNTACTIC_FEATURES + domain_names + DEP_FEATURES

    X_tr, y_tr, _, X_te, y_te, _ = train_test_split(X, y, list(range(len(y))), seed=seed)

    rf = RandomForest(n_trees=60, max_depth=7, seed=seed).fit(X_tr, y_tr)
    importances = rf.feature_importances(X_te, y_te)

    return sorted(zip(feat_names, importances.tolist()), key=lambda x: -x[1])


def train_dep_depth_target(records, X_embed, embed_dim, seed=42):
    """
    Secondary experiment: predict dep_depth instead of proof_length.
    dep_depth is a structural difficulty signal independent of proof style.

    IMPORTANT: restricted to records with dep_depth > 0 only.
    Including the zero-dep records inflates R² artificially — the model
    trivially learns to predict 0 for most inputs (the mode), which looks
    like high R² but carries no ranking signal (Spearman stays low).
    Filtering to non-zero gives an honest regression on structurally deep
    theorems, where the question is meaningful.
    """
    y_all = np.array([float(r.get("dep_depth", 0)) for r in records])
    nonzero_mask = y_all > 0
    n_nonzero = int(nonzero_mask.sum())

    if n_nonzero < 50:
        print(f"  [dep_depth target] Only {n_nonzero} non-zero records, skipping.")
        return {}

    # Filter everything to non-zero subset
    nz_idx = np.where(nonzero_mask)[0]
    records_nz = [records[i] for i in nz_idx]
    y = y_all[nonzero_mask]
    X_embed_nz = X_embed[nonzero_mask] if X_embed is not None else None

    print(f"\n{'='*55}")
    print(f"SECONDARY TARGET: dep_depth (non-zero only)")
    print(f"  n={n_nonzero} / {len(y_all)} total  |  "
          f"mean={y.mean():.2f}  median={float(np.median(y)):.1f}  max={y.max():.0f}")

    X_domain, domain_names = domain_onehot(records_nz)
    syn_vals = np.array([[r[f] for f in SYNTACTIC_FEATURES] for r in records_nz], dtype=np.float32)

    feature_sets_dep = {
        "A_syntactic":  syn_vals,
        "G_syn_domain": np.concatenate([syn_vals, X_domain], axis=1),
    }
    if embed_dim > 0 and X_embed_nz is not None:
        X_stmt_nz = X_embed_nz[:, :embed_dim]
        feature_sets_dep["E_syn_plus_tfidf"] = np.concatenate([syn_vals, X_stmt_nz], axis=1)

    results = {}
    ids_nz = list(range(n_nonzero))
    for fs_name, X_fs in feature_sets_dep.items():
        X_tr, y_tr, _, X_te, y_te, _ = train_test_split(X_fs, y, ids_nz, seed=seed)
        X_tr_s, X_te_s, _, _ = standardize(X_tr, X_te)
        gb = GradientBoostingRegressor(n_estimators=80, learning_rate=0.1, max_depth=4, seed=seed)
        gb.fit(X_tr_s, y_tr)
        pred = np.clip(gb.predict(X_te_s), 0, None)
        sr = spearman_r(pred, y_te)
        r2v = r2(pred, y_te)
        print(f"  {fs_name:<25} Spearman={sr:.3f} R2={r2v:.3f}")
        results[fs_name] = {"spearman_r": sr, "r2": r2v}

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="data/embeddings.npz")
    parser.add_argument("--data", default="data/mathlib_raw.jsonl")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.data) as f:
        records = [json.loads(l) for l in f]
    print(f"Loaded {len(records)} theorems")

    emb = np.load(args.embeddings, allow_pickle=True)
    X_embed = emb["X"].astype(np.float32)
    embed_dim = int(emb["embed_dim"])
    mode = str(emb["mode"])
    print(f"Embeddings: shape={X_embed.shape}, embed_dim={embed_dim}, mode={mode}")

    # Align records with embeddings by id
    id_to_record = {r["id"]: r for r in records}
    emb_ids = [str(i) for i in emb["ids"]]
    aligned_records = []
    aligned_X = []
    for i, eid in enumerate(emb_ids):
        if eid in id_to_record:
            aligned_records.append(id_to_record[eid])
            aligned_X.append(X_embed[i])
    X_embed = np.stack(aligned_X)
    records = aligned_records
    print(f"Aligned {len(records)} records with embeddings")

    print("\nRunning ablation study...")
    results, predictions, (p33, p67) = train_and_evaluate(
        records, X_embed, embed_dim, seed=args.seed)

    print("\nExtracting feature importances (syntactic + domain + dep)...")
    importances = extract_feature_importance(records, X_embed, embed_dim, seed=args.seed)
    print("  Top features:")
    for feat, imp in importances[:12]:
        print(f"    {feat:<30}: {imp:+.4f}")

    dep_results = train_dep_depth_target(records, X_embed, embed_dim, seed=args.seed)

    # Save results
    summary = {
        "n_theorems": len(records),
        "embed_mode": mode,
        "proof_length_q25": float(p33),
        "proof_length_q75": float(p67),
        "feature_sets": results,
        "feature_importance": {k: v for k, v in importances},
        "dep_depth_results": dep_results,
    }
    out_json = Path(args.output_dir) / "model_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Save test predictions
    pred_path = Path(args.output_dir) / "predictions_test.jsonl"
    with open(pred_path, "w") as f:
        for fs_name, pdata in predictions.items():
            for rec_id, true_val, pred_val in zip(pdata["ids"], pdata["true"], pdata["pred"]):
                f.write(json.dumps({
                    "feature_set": fs_name,
                    "id": rec_id,
                    "true_length": true_val,
                    "pred_length": pred_val,
                    "bucket_true": "easy" if true_val <= p33 else ("hard" if true_val >= p67 else "middle"),
                    "bucket_pred": "easy" if pred_val <= p33 else ("hard" if pred_val >= p67 else "middle"),
                }) + "\n")
    print(f"Test predictions saved to {pred_path}")


if __name__ == "__main__":
    main()