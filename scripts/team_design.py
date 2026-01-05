#!/usr/bin/env python3
"""
Exhaustive (small-N) team design from embedded, labeled interview chunks.

This script mirrors the logic in the notebook:
- build per-person concept scores along dimension vectors
- enumerate allocations and score how well they "span" the simplex
- sample top allocations

It is intended for small cohorts (e.g. 10–20 people). For larger N, you will
need heuristics (ILP / greedy / genetic algorithms).
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)


def build_concept_vectors(df: pd.DataFrame, label_col: str = "label", emb_col: str = "embedding") -> Dict[str, np.ndarray]:
    """
    Concept vector for a label = mean embedding of chunks with that label.
    """
    concepts: Dict[str, np.ndarray] = {}
    for lab, g in df.groupby(label_col):
        embs = np.vstack(g[emb_col].to_numpy())
        concepts[str(lab)] = _normalize(embs.mean(axis=0))
    return concepts


def speaker_vectors(df: pd.DataFrame, speaker_col: str = "file", emb_col: str = "embedding") -> Dict[str, np.ndarray]:
    """
    One embedding per speaker/file = mean embedding across chunks.
    """
    out = {}
    for spk, g in df.groupby(speaker_col):
        embs = np.vstack(g[emb_col].to_numpy())
        out[str(spk)] = _normalize(embs.mean(axis=0))
    return out


def score_team(team: List[str], spk_vecs: Dict[str, np.ndarray], concepts: Dict[str, np.ndarray]) -> float:
    """
    Score = volume proxy: determinant of Gram matrix of concept-projected team vectors.
    We compute each member's projection onto each concept direction, forming a matrix (len(team) x K),
    then use the determinant of the (K x K) covariance (or Gram) as a spanning metric.

    This is a simple proxy; you can swap in other objectives.
    """
    concept_names = sorted(concepts.keys())
    K = len(concept_names)
    M = np.zeros((len(team), K))
    for i, person in enumerate(team):
        v = spk_vecs[person]
        for j, cname in enumerate(concept_names):
            M[i, j] = float(np.dot(v, concepts[cname]))
    # center columns
    M = M - M.mean(axis=0, keepdims=True)
    G = M.T @ M
    # numeric stability
    return float(np.linalg.det(G + 1e-9 * np.eye(K)))


def enumerate_allocations(people: List[str], group_sizes: List[int]):
    """
    Yield allocations as list of groups (each group is list of people).
    """
    people = list(people)
    if sum(group_sizes) != len(people):
        raise ValueError("group_sizes must sum to number of people")

    # recursive enumeration
    def rec(remaining: Tuple[str, ...], sizes: List[int]):
        if not sizes:
            yield []
            return
        sz = sizes[0]
        for group in itertools.combinations(remaining, sz):
            rem = tuple([p for p in remaining if p not in group])
            for rest in rec(rem, sizes[1:]):
                yield [list(group)] + rest

    yield from rec(tuple(people), group_sizes)


def allocation_score(allocation: List[List[str]], spk_vecs, concepts) -> float:
    return float(np.mean([score_team(g, spk_vecs, concepts) for g in allocation]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedded-parquet", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("./outputs"))
    ap.add_argument("--group-sizes", type=str, default="4,4,3,3", help="Comma-separated sizes, e.g. 4,4,3,3")
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    df = pd.read_parquet(args.embedded_parquet)
    # ensure numpy arrays
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x, dtype=float))

    concepts = build_concept_vectors(df)
    spk_vecs = speaker_vectors(df)

    people = sorted(spk_vecs.keys())
    group_sizes = [int(x) for x in args.group_sizes.split(",") if x.strip()]
    if sum(group_sizes) != len(people):
        raise SystemExit(f"group_sizes sum to {sum(group_sizes)} but there are {len(people)} people/files")

    scored = []
    for alloc in enumerate_allocations(people, group_sizes):
        s = allocation_score(alloc, spk_vecs, concepts)
        scored.append({"score": s, "allocation": alloc})

    scored = sorted(scored, key=lambda d: d["score"], reverse=True)
    top = scored[: args.top_k]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "top_allocations.json"
    out_path.write_text(json_dumps_pretty(top), encoding="utf-8")
    print(f"Wrote {out_path} (top {len(top)} allocations by score).")


def json_dumps_pretty(obj) -> str:
    import json
    return json.dumps(obj, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
