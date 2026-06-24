"""
scrape_mathlib.py
==================
Scrapes Lean 4 theorem/proof pairs from Mathlib4 on GitHub.

For each theorem, extracts:
  - The natural language statement
  - The proof body
  - Structural/syntactic features
  - Direct lemma dependencies (named Mathlib identifiers invoked in the proof)
  - Mathematical domain (algebra, analysis, topology, etc.) from file path
  - Dependency depth: longest chain of transitive lemma dependencies within
    the scraped set (a structural measure of difficulty independent of proof
    length or author style)

Difficulty proxies:
  - proof_length    : tactic line count (noisy, style-dependent)
  - dep_depth       : longest dependency chain reaching this theorem (structural)
  - dep_count       : total direct + transitive dependency count (fanout)

Usage:
  python scrape_mathlib.py --output data/mathlib_raw.jsonl --max_per_file 80
"""

import re
import json
import time
import random
import argparse
import urllib.request
from pathlib import Path
from typing import Optional
from collections import defaultdict, deque

RAW_BASE = "https://raw.githubusercontent.com/leanprover-community/mathlib4/master"
HEADERS = {
    "Accept": "text/plain",
    "User-Agent": "LeanDifficultyPredictor-Research/1.0",
}

# ── Domain taxonomy ───────────────────────────────────────────────────────────
# Maps Mathlib path prefixes to a canonical domain label used as a feature.
# Order matters: first match wins.
DOMAIN_MAP = [
    ("Mathlib/NumberTheory",              "number_theory"),
    ("Mathlib/Algebra/BigOperators",      "combinatorics"),
    ("Mathlib/Algebra/Order",             "order_theory"),
    ("Mathlib/Algebra/Module",            "linear_algebra"),
    ("Mathlib/Algebra/Ring",              "algebra"),
    ("Mathlib/Algebra/Field",             "algebra"),
    ("Mathlib/Algebra/Group",             "algebra"),
    ("Mathlib/Algebra",                   "algebra"),
    ("Mathlib/LinearAlgebra",             "linear_algebra"),
    ("Mathlib/Analysis/SpecialFunctions", "calculus"),
    ("Mathlib/Analysis/Calculus",         "calculus"),
    ("Mathlib/Analysis/Normed",           "analysis"),
    ("Mathlib/Analysis",                  "analysis"),
    ("Mathlib/Topology",                  "topology"),
    ("Mathlib/Geometry",                  "geometry"),
    ("Mathlib/MeasureTheory",             "measure_theory"),
    ("Mathlib/Combinatorics",             "combinatorics"),
    ("Mathlib/Data/Nat",                  "number_theory"),
    ("Mathlib/Data/Int",                  "number_theory"),
    ("Mathlib/Data/Rat",                  "number_theory"),
    ("Mathlib/Data/Real",                 "analysis"),
    ("Mathlib/Data/Complex",              "analysis"),
    ("Mathlib/Data/List",                 "data_structures"),
    ("Mathlib/Data/Finset",               "combinatorics"),
    ("Mathlib/Data/Multiset",             "combinatorics"),
    ("Mathlib/Data/Set",                  "set_theory"),
    ("Mathlib/Data",                      "data_structures"),
    ("Mathlib/Order",                     "order_theory"),
    ("Mathlib/Logic",                     "logic"),
    ("Mathlib/SetTheory",                 "set_theory"),
    ("Mathlib/CategoryTheory",            "category_theory"),
    ("Mathlib/RingTheory",                "algebra"),
    ("Mathlib/FieldTheory",               "algebra"),
    ("Mathlib/GroupTheory",               "algebra"),
]

ALL_DOMAINS = sorted(set(d for _, d in DOMAIN_MAP))

def infer_domain(filepath: str) -> str:
    for prefix, domain in DOMAIN_MAP:
        if filepath.startswith(prefix):
            return domain
    return "other"


# ── Mathlib file list ─────────────────────────────────────────────────────────
MATHLIB_FILES = [
    # Algebra
    "Mathlib/Algebra/Group/Basic.lean",
    "Mathlib/Algebra/Ring/Basic.lean",
    "Mathlib/Algebra/Field/Basic.lean",
    "Mathlib/Algebra/Order/Ring/Lemmas.lean",
    "Mathlib/Algebra/BigOperators/Group/Finset/Basic.lean",
    "Mathlib/Algebra/Module/Basic.lean",
    "Mathlib/Algebra/GCDMonoid/Basic.lean",
    # Number theory
    "Mathlib/NumberTheory/Primes/Basic.lean",
    "Mathlib/NumberTheory/Bernoulli.lean",
    "Mathlib/Data/Nat/Basic.lean",
    "Mathlib/Data/Nat/Defs.lean",
    "Mathlib/Data/Int/Basic.lean",
    "Mathlib/Data/Int/Lemmas.lean",
    "Mathlib/Data/Rat/Basic.lean",
    # Data structures / combinatorics
    "Mathlib/Data/List/Basic.lean",
    "Mathlib/Data/List/Lemmas.lean",
    "Mathlib/Data/Finset/Basic.lean",
    "Mathlib/Data/Multiset/Basic.lean",
    "Mathlib/Data/Set/Basic.lean",
    "Mathlib/Data/Set/Function.lean",
    "Mathlib/Combinatorics/SimpleGraph/Basic.lean",
    "Mathlib/Combinatorics/Enumerative/Composition.lean",
    # Analysis / calculus
    "Mathlib/Analysis/SpecialFunctions/Pow/Real.lean",
    "Mathlib/Analysis/SpecialFunctions/Log/Basic.lean",
    "Mathlib/Analysis/SpecialFunctions/Trigonometric/Basic.lean",
    "Mathlib/Analysis/Normed/Group/Basic.lean",
    "Mathlib/Analysis/Calculus/Deriv/Basic.lean",
    "Mathlib/Analysis/Calculus/MeanValue.lean",
    # Topology
    "Mathlib/Topology/Basic.lean",
    "Mathlib/Topology/Algebra/Order/Basic.lean",
    "Mathlib/Topology/MetricSpace/Basic.lean",
    # Logic / order
    "Mathlib/Logic/Basic.lean",
    "Mathlib/Order/Basic.lean",
    "Mathlib/Order/Bounds/Basic.lean",
    # Linear algebra
    "Mathlib/LinearAlgebra/Basic.lean",
    "Mathlib/LinearAlgebra/Matrix/Basic.lean",
    # Geometry
    "Mathlib/Geometry/Euclidean/Basic.lean",
    # Measure theory
    "Mathlib/MeasureTheory/Measure/MeasureSpace.lean",
]


# ── Lean tactics (excluded from dependency extraction) ────────────────────────
LEAN_TACTICS = {
    "simp", "ring", "omega", "linarith", "norm_num", "exact", "apply",
    "intro", "intros", "constructor", "left", "right", "rfl", "trivial",
    "tauto", "contradiction", "assumption", "aesop", "decide", "positivity",
    "have", "obtain", "let", "use", "refine", "rw", "rewrite", "calc",
    "induction", "cases", "rcases", "ext", "funext", "congr", "conv",
    "push_neg", "pull_neg", "norm_cast", "push_cast", "field_simp",
    "ring_nf", "simp_rw", "gcongr", "nlinarith", "polyrith",
    "fin_cases", "interval_cases", "contrapose", "by_contra", "by_cases",
    "suffices", "show", "change", "unfold", "delta", "ac_rfl",
    "group", "abel", "module_cast", "norm_fin", "norm_cast", "positivity",
    "constructor", "trivial", "rfl", "rfl", "exact",
}

LEAN_KEYWORDS = {
    "theorem", "lemma", "def", "instance", "structure", "class", "where",
    "forall", "exists", "fun", "let", "have", "show", "from", "by",
    "match", "with", "if", "then", "else", "do", "return", "import",
    "open", "namespace", "end", "section", "variable", "attribute",
    "noncomputable", "private", "protected", "unsafe", "opaque",
    "true", "false", "Type", "Prop", "Sort", "and", "or", "not",
    "And", "Or", "Not", "True", "False", "Eq", "Ne", "HEq",
}

HARD_TACTICS = {"induction", "calc", "rcases", "obtain", "conv", "polyrith", "gcongr"}
EASY_TACTICS = {"rfl", "trivial", "decide", "assumption", "exact", "norm_num"}

THEOREM_RE = re.compile(
    r"^(theorem|lemma)\s+(\w+)\s*([^:]*?)\s*:\s*(.*?)\s*:=\s*by\b(.*?)(?=\n(?:theorem|lemma|def |instance |example |#|end |\Z))",
    re.DOTALL | re.MULTILINE,
)

# Matches dotted Mathlib-style identifiers (e.g. Nat.add_comm, List.length_append)
# Must have at least one dot (pure tactic names have no dot)
DEP_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+)\b")


def fetch(url: str, retries: int = 3) -> Optional[str]:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.read().decode("utf-8")
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
    return None


def extract_direct_deps(proof_body: str, known_names: set[str]) -> list[str]:
    """
    Extract named Mathlib lemma references from a proof body.

    We look for:
      1. Dotted identifiers (Nat.add_comm, List.length_append) — checked against
         both full name and last component in known_names.
      2. Underscore-style identifiers that exactly match a known theorem name
         (e.g. mul_comm, add_zero) — common in apply/exact/rw calls.

    Only keeps names present in known_names so the dep graph stays in-corpus.
    """
    deps = []
    seen = set()

    # Pass 1: dotted identifiers (Namespace.name style)
    for c in DEP_RE.findall(proof_body):
        if c in seen:
            continue
        seen.add(c)
        short = c.split(".")[-1]
        if c in known_names:
            deps.append(c)
        elif short in known_names:
            deps.append(short)

    # Pass 2: plain identifiers that exactly match a scraped theorem name.
    # Filter out tactics, keywords, and single-letter variables.
    for tok in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_\']{2,})\b", proof_body):
        if tok in seen:
            continue
        if tok in LEAN_TACTICS or tok in LEAN_KEYWORDS:
            continue
        if tok in known_names:
            seen.add(tok)
            deps.append(tok)

    return list(dict.fromkeys(deps))  # deduplicate preserving order


def extract_features(proof_body: str, statement: str) -> dict:
    lines = [l for l in proof_body.split("\n") if l.strip()]
    n_lines = len(lines)

    all_tokens = re.findall(r"\b\w+\b", proof_body)
    tactic_counts = {}
    for tok in all_tokens:
        if tok in LEAN_TACTICS:
            tactic_counts[tok] = tactic_counts.get(tok, 0) + 1

    n_tactics = sum(tactic_counts.values())
    n_unique_tactics = len(tactic_counts)
    n_hard = sum(tactic_counts.get(t, 0) for t in HARD_TACTICS)
    n_easy = sum(tactic_counts.get(t, 0) for t in EASY_TACTICS)

    has_induction = int("induction" in proof_body or "Nat.rec" in proof_body)
    has_calc = int("calc" in proof_body)
    has_have = int("have" in proof_body or "obtain" in proof_body)
    has_cases = int("cases" in proof_body or "rcases" in proof_body)
    nesting_depth = max((len(l) - len(l.lstrip())) // 2 for l in lines) if lines else 0

    stmt_tokens = re.findall(r"\b\w+\b", statement)
    stmt_len = len(stmt_tokens)
    has_forall = int("∀" in statement or "forall" in statement)
    has_exists = int("∃" in statement or "exists" in statement)
    has_iff = int("↔" in statement or "Iff" in statement)
    has_neg = int("¬" in statement or "Not" in statement)
    quantifier_depth = statement.count("∀") + statement.count("∃")

    return {
        "proof_length": n_lines,
        "n_tactics": n_tactics,
        "n_unique_tactics": n_unique_tactics,
        "n_hard_tactics": n_hard,
        "n_easy_tactics": n_easy,
        "has_induction": has_induction,
        "has_calc": has_calc,
        "has_have": has_have,
        "has_cases": has_cases,
        "nesting_depth": nesting_depth,
        "tactic_diversity": n_unique_tactics / max(n_tactics, 1),
        "hard_tactic_ratio": n_hard / max(n_tactics, 1),
        "stmt_token_count": stmt_len,
        "has_forall": has_forall,
        "has_exists": has_exists,
        "has_iff": has_iff,
        "has_neg": has_neg,
        "quantifier_depth": quantifier_depth,
    }


def parse_file(src: str, filename: str, max_per_file: int = 80) -> list[dict]:
    domain = infer_domain(filename)
    results = []
    for m in THEOREM_RE.finditer(src):
        kind = m.group(1)
        name = m.group(2)
        params = m.group(3).strip()
        typ = m.group(4).strip()
        body = m.group(5)

        if "sorry" in body or "admit" in body:
            continue
        if name.startswith("_") or len(name) < 3:
            continue

        proof_lines = [l for l in body.split("\n") if l.strip()]
        if len(proof_lines) < 1:
            continue

        features = extract_features(body, typ)
        if features["proof_length"] == 0:
            continue

        full_stmt = f"{kind} {name} {params} : {typ}" if params else f"{kind} {name} : {typ}"

        results.append({
            "id": f"{filename.replace('/', '_').replace('.lean', '')}_{name}",
            "file": filename,
            "domain": domain,
            "name": name,
            "kind": kind,
            "full_statement": full_stmt,
            "statement_type": typ,
            "params": params,
            "proof_body": body.strip(),
            # direct_deps filled in post-scrape once we have all names
            "direct_deps": [],
            **features,
            # dep_depth and dep_count filled after graph computation
            "dep_depth": 0,
            "dep_count": 0,
        })

        if len(results) >= max_per_file:
            break

    return results


# ── Dependency graph ──────────────────────────────────────────────────────────

def build_dependency_graph(records: list[dict]) -> list[dict]:
    """
    Two-pass algorithm:
      Pass 1: extract direct deps per theorem using the full scraped name set
      Pass 2: BFS/topo-sort to compute dep_depth (longest chain) and
              dep_count (transitive closure size) for each node

    dep_depth: length of the longest path TO this theorem through the dep DAG.
               A theorem with dep_depth=0 depends on nothing in our set (leaf).
               dep_depth=3 means there's a chain A->B->C->this theorem.

    dep_count: total number of distinct theorems reachable backwards from this
               node. Measures how deep the proof "stands on the shoulders of".
    """
    # Build name -> record index lookup (use short name for matching)
    name_to_idx = {}
    for i, r in enumerate(records):
        name_to_idx[r["name"]] = i
        name_to_idx[r["id"]] = i  # also index by full id

    known_names = set(name_to_idx.keys())

    print("  Building dependency graph...")

    # Pass 1: extract direct deps
    for r in records:
        deps = extract_direct_deps(r["proof_body"], known_names)
        # Don't allow self-loops
        deps = [d for d in deps if d != r["name"] and d != r["id"]]
        r["direct_deps"] = deps

    # Build adjacency: idx -> set of dependency indices
    adj: dict[int, set[int]] = defaultdict(set)
    for i, r in enumerate(records):
        for dep_name in r["direct_deps"]:
            j = name_to_idx.get(dep_name)
            if j is not None and j != i:
                adj[i].add(j)

    n = len(records)

    # Pass 2: compute dep_depth via iterative longest-path (DAG assumed, break cycles)
    # We use a simple iterative relaxation (Bellman-Ford style, max version)
    # This handles cycles gracefully by capping iterations.
    depth = [0] * n
    changed = True
    for _ in range(min(n, 20)):  # max chain length we care about
        if not changed:
            break
        changed = False
        for i in range(n):
            for j in adj[i]:
                if depth[i] < depth[j] + 1:
                    depth[i] = depth[j] + 1
                    changed = True

    # Pass 3: transitive dependency count via BFS per node
    # For efficiency, cache reachable sets only for nodes that matter
    # Use memoized BFS
    memo: dict[int, set] = {}

    def reachable(idx: int, visiting: set) -> set:
        if idx in memo:
            return memo[idx]
        if idx in visiting:
            return set()  # cycle break
        visiting = visiting | {idx}
        result = set()
        for j in adj[idx]:
            result.add(j)
            result |= reachable(j, visiting)
        memo[idx] = result
        return result

    for i in range(n):
        r_set = reachable(i, set())
        records[i]["dep_depth"] = depth[i]
        records[i]["dep_count"] = len(r_set)

    # Stats
    depths = [r["dep_depth"] for r in records]
    counts = [r["dep_count"] for r in records]
    print(f"  Dependency depth: min={min(depths)} mean={sum(depths)/n:.1f} max={max(depths)}")
    print(f"  Dep count:        min={min(counts)} mean={sum(counts)/n:.1f} max={max(counts)}")
    non_leaf = sum(1 for d in counts if d > 0)
    print(f"  Theorems with >=1 in-corpus dependency: {non_leaf}/{n} ({100*non_leaf/n:.0f}%)")

    return records


# ── Main scrape ───────────────────────────────────────────────────────────────

def scrape(output_path: str, max_per_file: int = 80, seed: int = 42) -> None:
    random.seed(seed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    print(f"Scraping {len(MATHLIB_FILES)} Mathlib4 files...")

    for i, path in enumerate(MATHLIB_FILES):
        url = f"{RAW_BASE}/{path}"
        print(f"  [{i+1:2d}/{len(MATHLIB_FILES)}] {path}...", end="", flush=True)
        src = fetch(url)
        if not src:
            print(" FAILED")
            continue
        records = parse_file(src, path, max_per_file)
        domain = infer_domain(path)
        print(f" {len(records)} theorems [{domain}]")
        all_records.extend(records)
        time.sleep(0.4)

    # Deduplicate by id
    seen = set()
    unique = []
    for r in all_records:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)

    print(f"\nScraped {len(unique)} unique theorems")

    # Domain distribution
    from collections import Counter
    domain_dist = Counter(r["domain"] for r in unique)
    print("Domain distribution:")
    for dom, cnt in sorted(domain_dist.items(), key=lambda x: -x[1]):
        print(f"  {dom:<20}: {cnt}")

    # Build dependency graph
    unique = build_dependency_graph(unique)

    random.shuffle(unique)

    with open(output_path, "w") as f:
        for r in unique:
            f.write(json.dumps(r) + "\n")

    lengths = [r["proof_length"] for r in unique]
    depths = [r["dep_depth"] for r in unique]
    print(f"\nProof length: min={min(lengths)} mean={sum(lengths)/len(lengths):.1f} max={max(lengths)}")
    print(f"Dep depth:    min={min(depths)} mean={sum(depths)/len(depths):.1f} max={max(depths)}")
    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/mathlib_raw.jsonl")
    parser.add_argument("--max_per_file", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    scrape(args.output, args.max_per_file, args.seed)
