#!/usr/bin/env python3
"""
Evaluate MAGICC predictions against ground truth.

Conventions, fixed by protocol section 4.4d and WS5.4 and NOT negotiable here:

  * **R2 is the coefficient of determination**, 1 - SS_res/SS_tot. Squared
    Pearson r is reported separately and labelled `r_pearson_sq`; it is never
    called R2. Squared Pearson ignores bias and scale error and is always >= the
    coefficient of determination, which is why the two were confused in the
    submitted manuscript.
  * **Confidence intervals come from a cluster bootstrap** over reference
    genomes, because benchmark sets reuse each reference genome across multiple
    simulations. Clusters are (set, reference) pairs; resampling is with
    replacement over clusters, not over genomes. A naive i.i.d. bootstrap gives
    54 % coverage where the cluster bootstrap gives 95 % (WS5.4).
  * R2 is omitted where the true value has zero variance (R1-m19).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("completeness", "contamination")


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    need = {"genome_name", "pred_completeness", "pred_contamination"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing prediction columns {sorted(missing)}")
    return df.rename(columns={"genome_name": "genome_id"})[
        ["genome_id", "pred_completeness", "pred_contamination"]]


def point_stats(true: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    out = dict(
        n=int(true.size),
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        bias=float(np.mean(err)),
        true_variance=float(np.var(true)),
    )
    if ss_tot == 0.0:
        out["r2"] = None
        out["r2_omitted_reason"] = "true value has zero variance (R1-m19)"
        out["r_pearson_sq"] = None
    else:
        out["r2"] = 1.0 - ss_res / ss_tot
        out["r2_omitted_reason"] = None
        r = float(np.corrcoef(true, pred)[0, 1])
        out["r_pearson_sq"] = r ** 2
    return out


def cluster_bootstrap(true: np.ndarray, pred: np.ndarray, clusters: np.ndarray,
                      n_iter: int, seed: int) -> dict:
    """Percentile CIs for MAE, bias and R2, resampling clusters with replacement."""
    uniq, inverse = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(uniq.size)]
    rng = np.random.default_rng(seed)
    mae_s, bias_s, r2_s = [], [], []
    for _ in range(n_iter):
        pick = rng.integers(0, uniq.size, size=uniq.size)
        idx = np.concatenate([members[p] for p in pick])
        t, p = true[idx], pred[idx]
        e = p - t
        mae_s.append(np.mean(np.abs(e)))
        bias_s.append(np.mean(e))
        sst = np.sum((t - t.mean()) ** 2)
        r2_s.append(np.nan if sst == 0 else 1.0 - np.sum(e ** 2) / sst)

    def ci(v):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return [None, None]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    return dict(n_clusters=int(uniq.size), n_iterations=int(n_iter), seed=int(seed),
                mae_ci=ci(mae_s), bias_ci=ci(bias_s), r2_ci=ci(r2_s))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--metadata", nargs="+", required=True)
    ap.add_argument("--set-name", required=True)
    ap.add_argument("--set-labels", nargs="*", default=None,
                    help="labels aligned with --predictions when pooling")
    ap.add_argument("--cluster-column", default="dominant_accession")
    ap.add_argument("--bootstrap-iterations", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260726)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    if len(a.predictions) != len(a.metadata):
        raise SystemExit("--predictions and --metadata must have the same length")
    labels = a.set_labels or [Path(p).stem for p in a.predictions]
    if len(labels) != len(a.predictions):
        raise SystemExit("--set-labels must align with --predictions")

    frames = []
    for label, pred_path, meta_path in zip(labels, a.predictions, a.metadata):
        pred = read_predictions(Path(pred_path))
        meta = pd.read_csv(meta_path, sep="\t")
        for col in ("genome_id", "true_completeness", "true_contamination", a.cluster_column):
            if col not in meta.columns:
                raise SystemExit(f"{meta_path}: missing column {col}")
        merged = meta[["genome_id", "true_completeness", "true_contamination",
                       a.cluster_column]].merge(pred, on="genome_id", how="inner")
        if merged.empty:
            raise SystemExit(f"{label}: no genome_id overlap between predictions and metadata")
        n_unmatched = len(pred) - len(merged)
        if n_unmatched:
            print(f"  ! {label}: {n_unmatched} predicted genome(s) absent from metadata",
                  file=sys.stderr)
        # Cluster identity is scoped to the set, so a reference genome that
        # appears in two sets is two clusters, matching the WS5.4 convention.
        merged["cluster"] = label + "::" + merged[a.cluster_column].astype(str)
        merged["set"] = label
        frames.append(merged)

    df = pd.concat(frames, ignore_index=True)
    result = dict(
        schema="magicc-workflow-evaluation/1.0",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        set_name=a.set_name,
        sets_included=labels,
        n_genomes=int(len(df)),
        cluster_column=a.cluster_column,
        cluster_scope="(set, reference genome) pair",
        r2_convention="coefficient of determination, 1 - SS_res/SS_tot (protocol 4.4d)",
        metrics={},
    )
    for m in METRICS:
        true = df[f"true_{m}"].to_numpy(dtype=float)
        pred = df[f"pred_{m}"].to_numpy(dtype=float)
        stats = point_stats(true, pred)
        stats.update(cluster_bootstrap(true, pred, df["cluster"].to_numpy(),
                                       a.bootstrap_iterations, a.bootstrap_seed))
        result["metrics"][m] = stats
        r2 = stats["r2"]
        print(f"  {a.set_name:32s} {m:13s} n={stats['n']:5d} "
              f"MAE={stats['mae']:.4f} [{stats['mae_ci'][0]:.4f}, {stats['mae_ci'][1]:.4f}] "
              f"bias={stats['bias']:+.4f} R2={'n/a' if r2 is None else f'{r2:.4f}'}")

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
