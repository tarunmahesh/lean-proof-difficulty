# lean-proof-difficulty

Predicts how hard a Lean 4 theorem is to prove — from the statement alone.

Scrapes ~1400 theorems from [Mathlib4](https://github.com/leanprover-community/mathlib4), builds a lemma dependency graph, classifies each theorem by mathematical domain, and trains regressors to predict proof complexity.

## What it does

Given just the text of a theorem statement like:

```
theorem mul_comm (a b : α) : a * b = b * a
```

it predicts how long the proof will be.

Three ways to represent a theorem are compared:

**Syntactic** — hand-crafted counts from the statement: token count, whether it contains ∀/∃/↔/¬, quantifier nesting depth, which area of math it's from (algebra/calculus/etc.), and where it sits in the lemma dependency graph (how many other lemmas lead up to it).

**TF-IDF** — treats the statement as a bag of tokens and learns which Lean identifiers and type names statistically predict proof length across the corpus. No hand-crafting; weights learned from data.

**Bag-of-tactics** — counts tactic frequencies from the proof body itself (`simp`, `induction`, `calc`, etc.). This is an oracle since it requires the proof — included as an upper bound.

## Results

**1376 theorems, 13 domains. Target: `log(proof_length + 1)`. Main metric: Spearman ρ.**

| Feature Set | Dim | Model | Spearman ρ | LogRMSE | R² | Q1Q4 Acc |
|---|---|---|---|---|---|---|
| B · Bag-of-tactics† | 53 | Gradient Boosting | **0.764** | 0.316 | **0.682** | **90.5%** |
| D · Syn + BoT† | 59 | Gradient Boosting | 0.740 | 0.330 | 0.654 | 90.5% |
| I · Kitchen sink† | 1100 | Gradient Boosting | 0.744 | **0.334** | 0.646 | 84.9% |
| F · Full† | 1083 | Gradient Boosting | 0.740 | 0.335 | 0.642 | 87.3% |
| H · Syn + Domain + Dep | 23 | Random Forest | 0.423 | 0.496 | 0.219 | 66.7% |
| E · Syn + TF-IDF | 1030 | Gradient Boosting | 0.366 | 0.511 | 0.169 | 65.1% |
| C · TF-IDF stmt | 1024 | Gradient Boosting | 0.330 | 0.516 | 0.155 | 62.7% |
| G · Syn + Domain | 21 | Gradient Boosting | 0.268 | 0.515 | 0.157 | 68.3% |
| A · Syntactic only | 6 | Random Forest | 0.202 | 0.522 | 0.133 | 67.5% |

† Uses proof-side information — oracle upper bound. Statement-only sets are A, C, G, H, E.

Q1Q4 accuracy: binary easy (≤2 tactic lines) vs hard (≥4 lines), middle 50% dropped.

<details>
<summary>Full ablation (all models)</summary>

```
A_syntactic  (dim=6)
  Ridge              Spearman=0.225  LogRMSE=1.367  R2=-4.939  Q1Q4Acc=0.571
  Random Forest      Spearman=0.202  LogRMSE=0.522  R2= 0.133  Q1Q4Acc=0.675
  Gradient Boosting  Spearman=0.171  LogRMSE=0.529  R2= 0.111  Q1Q4Acc=0.690
  MLP                Spearman=0.124  LogRMSE=0.562  R2=-0.003  Q1Q4Acc=0.690

G_syn_domain  (dim=21)
  Ridge              Spearman=0.206  LogRMSE=1.349  R2=-4.781  Q1Q4Acc=0.571
  Random Forest      Spearman=0.251  LogRMSE=0.524  R2= 0.127  Q1Q4Acc=0.651
  Gradient Boosting  Spearman=0.268  LogRMSE=0.515  R2= 0.157  Q1Q4Acc=0.683
  MLP                Spearman=0.119  LogRMSE=0.610  R2=-0.182  Q1Q4Acc=0.595

H_syn_dom_dep  (dim=23)
  Ridge              Spearman=0.337  LogRMSE=1.330  R2=-4.619  Q1Q4Acc=0.571
  Random Forest      Spearman=0.423  LogRMSE=0.496  R2= 0.219  Q1Q4Acc=0.667
  Gradient Boosting  Spearman=0.393  LogRMSE=0.490  R2= 0.238  Q1Q4Acc=0.706
  MLP                Spearman=0.148  LogRMSE=0.660  R2=-0.383  Q1Q4Acc=0.635

B_bag_of_tactics  (dim=53)
  Ridge              Spearman=0.660  LogRMSE=1.180  R2=-3.423  Q1Q4Acc=0.579
  Random Forest      Spearman=0.729  LogRMSE=0.356  R2= 0.597  Q1Q4Acc=0.873
  Gradient Boosting  Spearman=0.764  LogRMSE=0.316  R2= 0.682  Q1Q4Acc=0.905
  MLP                Spearman=0.431  LogRMSE=0.736  R2=-0.719  Q1Q4Acc=0.667

D_syn_plus_bot  (dim=59)
  Ridge              Spearman=0.653  LogRMSE=1.177  R2=-3.401  Q1Q4Acc=0.579
  Random Forest      Spearman=0.703  LogRMSE=0.359  R2= 0.592  Q1Q4Acc=0.873
  Gradient Boosting  Spearman=0.740  LogRMSE=0.330  R2= 0.654  Q1Q4Acc=0.905
  MLP                Spearman=0.419  LogRMSE=0.682  R2=-0.479  Q1Q4Acc=0.730

C_tfidf  (dim=1024)
  Ridge              Spearman=0.222  LogRMSE=1.262  R2=-4.056  Q1Q4Acc=0.556
  Random Forest      Spearman=0.294  LogRMSE=0.533  R2= 0.098  Q1Q4Acc=0.587
  Gradient Boosting  Spearman=0.330  LogRMSE=0.516  R2= 0.155  Q1Q4Acc=0.627

E_syn_plus_tfidf  (dim=1030)
  Ridge              Spearman=0.224  LogRMSE=1.257  R2=-4.018  Q1Q4Acc=0.563
  Random Forest      Spearman=0.313  LogRMSE=0.528  R2= 0.116  Q1Q4Acc=0.619
  Gradient Boosting  Spearman=0.366  LogRMSE=0.511  R2= 0.169  Q1Q4Acc=0.651

F_full  (dim=1083)
  Ridge              Spearman=0.464  LogRMSE=1.132  R2=-3.073  Q1Q4Acc=0.643
  Random Forest      Spearman=0.673  LogRMSE=0.430  R2= 0.412  Q1Q4Acc=0.754
  Gradient Boosting  Spearman=0.740  LogRMSE=0.335  R2= 0.642  Q1Q4Acc=0.873

I_kitchen_sink  (dim=1100)
  Ridge              Spearman=0.442  LogRMSE=1.133  R2=-3.079  Q1Q4Acc=0.635
  Random Forest      Spearman=0.695  LogRMSE=0.430  R2= 0.412  Q1Q4Acc=0.762
  Gradient Boosting  Spearman=0.744  LogRMSE=0.334  R2= 0.646  Q1Q4Acc=0.849
```
</details>

**Secondary target: dep_depth** (restricted to 726/1376 theorems with non-zero in-corpus depth)

| Feature Set | Spearman ρ | R² |
|---|---|---|
| G · Syn + Domain | 0.316 | 0.127 |
| E · Syn + TF-IDF | 0.312 | 0.612 |
| A · Syntactic only | 0.156 | -0.076 |

G and E have similar Spearman but E has much higher R² — domain labels help rank theorems by graph depth, TF-IDF additionally gets the magnitude right.

**Feature importances** (permutation, RF on syntactic + domain + dep):

| Feature | Importance |
|---|---|
| `dep_depth` | +0.0257 |
| `stmt_token_count` | +0.0228 |
| `dep_count` | +0.0198 |
| `quantifier_depth` | +0.0128 |
| `has_exists` | +0.0065 |
| `has_iff` | +0.0055 |
| `has_forall` | +0.0048 |
| `domain_data_structures` | +0.0047 |
| `domain_measure_theory` | +0.0040 |
| `domain_logic` | +0.0039 |
| `domain_set_theory` | -0.0018 |
| `domain_linear_algebra` | -0.0000 |

`dep_depth` is the strongest single predictor — where a theorem sits in the lemma dependency chain matters more than how long its statement is. Domains with zero importance (geometry, topology, category theory) have too few examples (<15) to learn from.

## Feature sets

| ID | Description | Uses proof? |
|---|---|---|
| A | Syntactic: token count, quantifier depth, ∀/∃/↔/¬ flags | No |
| B | Bag-of-tactics: tactic frequencies from proof body | **Yes** |
| C | TF-IDF over theorem statement tokens | No |
| D | A + B | **Yes** |
| E | A + C | No |
| F | A + B + C | **Yes** |
| G | A + domain one-hot (15 categories) | No |
| H | A + domain + dep_depth + dep_count | No |
| I | All of the above (kitchen sink) | **Yes** |

Statement-only (A, C, G, H, E) is the honest benchmark. B/D/F/I are oracle upper bounds.

**Models:** Ridge, Random Forest, Gradient Boosting, MLP — all from scratch in numpy, no sklearn.

## Difficulty proxies

`proof_length` — tactic line count. Noisy: a deep theorem might close in one `simp` if Mathlib already has the machinery; a verbose proof isn't necessarily hard. Style-dependent.

`dep_depth` — longest chain of lemma dependencies reaching this theorem within the scraped corpus. Style-independent and structural. Noisier as ground truth since only 53% of theorems have in-corpus deps (the rest depend on lemmas outside the scraped set), but a more honest measure of mathematical depth.

## Usage

```bash
pip install numpy

python3 run_pipeline.py                   # full run (~3 min)
python3 run_pipeline.py --only_train      # skip re-scraping
python3 run_pipeline.py --embed_mode api  # OpenAI embeddings (set OPENAI_API_KEY)
```

Results written to `results/model_results.json` and `results/predictions_test.jsonl`.

## Structure

```
scrape_mathlib.py    # scrapes Mathlib4, extracts features + lemma dependency graph
embed_statements.py  # TF-IDF, bag-of-tactics, or OpenAI embeddings
train_predictor.py   # all models + ablation, pure numpy
run_pipeline.py      # orchestrator
```

## Data

1376 theorems from 38 Mathlib4 files across 13 domains. Proof length: min=1, Q25=2, median=2, Q75=4, max=71. Dependency graph: 53% of theorems have ≥1 in-corpus dependency, depth up to 60.

Domain distribution: calculus (366), combinatorics (306), algebra (205), data structures (99), analysis (92), measure theory (82), order theory (61), set theory (54), logic (48), number theory (36), topology (15), linear algebra (6), geometry (6).

## License

MIT. Data from [Mathlib4](https://github.com/leanprover-community/mathlib4) (Apache 2.0).