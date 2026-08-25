#!/usr/bin/env python3
"""
Compare the freshly computed headline statistics with the recorded values and
write the reproduction report.

Exits non-zero if a reproduced value falls outside tolerance, so the workflow
FAILS loudly rather than producing a report that quietly disagrees with the
manuscript.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TOOL = "magicc_v5"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled", required=True)
    ap.add_argument("--per-set", nargs="+", required=True)
    ap.add_argument("--verified", required=True)
    ap.add_argument("--reference-table", required=True)
    ap.add_argument("--reference-row", required=True)
    ap.add_argument("--tolerance-pp", type=float, default=0.01)
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-json", required=True)
    a = ap.parse_args()

    pooled = json.loads(Path(a.pooled).read_text())
    verified = json.loads(Path(a.verified).read_text())
    per_set = {}
    for p in a.per_set:
        d = json.loads(Path(p).read_text())
        per_set[d["set_name"]] = d

    ref = pd.read_csv(a.reference_table, sep="\t")
    ref = ref[(ref["set"] == a.reference_row) & (ref["tool"] == TOOL)]

    subsampled = a.subsample and a.subsample > 0
    comparisons, failures = [], []
    for metric in ("completeness", "contamination"):
        got = pooled["metrics"][metric]
        row = ref[ref["metric"] == metric]
        if row.empty:
            comparisons.append(dict(metric=metric, status="NO_REFERENCE",
                                    reproduced_mae=got["mae"], recorded_mae=None, delta=None))
            continue
        rec_mae = float(row["mae"].iloc[0])
        delta = got["mae"] - rec_mae
        if subsampled:
            status = "SKIPPED_SUBSAMPLED"
        elif abs(delta) <= a.tolerance_pp:
            status = "MATCH"
        else:
            status = "MISMATCH"
            failures.append(f"{metric}: reproduced {got['mae']:.4f} vs recorded {rec_mae:.4f} "
                            f"(delta {delta:+.4f} pp, tolerance {a.tolerance_pp})")
        comparisons.append(dict(
            metric=metric, status=status,
            reproduced_mae=got["mae"], reproduced_mae_ci=got["mae_ci"],
            recorded_mae=rec_mae,
            recorded_mae_ci=[float(row["mae_ci_lo"].iloc[0]), float(row["mae_ci_hi"].iloc[0])],
            delta_pp=delta,
            reproduced_r2=got["r2"],
            recorded_r2=(None if pd.isna(row["r2"].iloc[0]) else float(row["r2"].iloc[0])),
            reproduced_bias=got["bias"],
            n=got["n"], n_clusters=got["n_clusters"]))

    overall = "FAIL" if failures else ("SUBSAMPLED_STRUCTURAL_ONLY" if subsampled else "PASS")

    summary = dict(
        schema="magicc-headline-reproduction/1.0",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        workstream_items=["WS7.4", "WS7.11"],
        reviewer_comments=["R1-M6", "R1-M8", "R1-m18"],
        overall=overall,
        subsampled=bool(subsampled),
        subsample_n_per_set=int(a.subsample),
        tolerance_pp=a.tolerance_pp,
        model=dict(sha256=verified["model_sha256"], version=verified["model_version"]),
        artefact_checks=verified["overall"],
        pooled=pooled,
        per_set={k: v["metrics"] for k, v in per_set.items()},
        comparisons=comparisons,
        failures=failures,
    )
    Path(a.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # ---------------- markdown report ----------------
    L = []
    L.append("# MAGICC headline-claim reproduction (WS7.4 / WS7.11)\n")
    L.append(f"Generated {datetime.now(timezone.utc).isoformat()} by "
             "`workflow/Snakefile` via `workflow/scripts/compare_and_report.py`.\n")
    L.append(f"**Result: {overall}**\n")
    if subsampled:
        L.append(f"> This run used `subsample={a.subsample}` genomes per set. It is a "
                 "**structural test of the workflow only** and is not a reproduction of "
                 "the headline numbers. Set `subsample: 0` for the real run.\n")
    L.append("## Frozen artefacts\n")
    L.append(f"* Model **{verified['model_version']}**, SHA256 `{verified['model_sha256']}`")
    L.append(f"* {verified['n_kmers']:,} selected 9-mers, SHA256 `{verified['kmer_list_sha256']}`")
    L.append(f"* Artefact checks: **{verified['overall']}**\n")

    L.append("## Pooled leakage-free five-set panel\n")
    L.append(f"Sets: {', '.join(pooled['sets_included'])} — "
             f"{pooled['n_genomes']:,} genomes, "
             f"{pooled['metrics']['completeness']['n_clusters']:,} clusters "
             f"({pooled['cluster_scope']}).\n")
    L.append("| Metric | Reproduced MAE (pp) | 95 % CI | Recorded MAE | Δ (pp) | Status |")
    L.append("|---|---|---|---|---|---|")
    for c in comparisons:
        ci = c.get("reproduced_mae_ci") or [None, None]
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "—"
        rec = f"{c['recorded_mae']:.4f}" if c.get("recorded_mae") is not None else "—"
        dl = f"{c['delta_pp']:+.5f}" if c.get("delta_pp") is not None else "—"
        L.append(f"| {c['metric']} | {c['reproduced_mae']:.4f} | {ci_s} | {rec} | {dl} | {c['status']} |")
    L.append("")
    L.append("R² (coefficient of determination, 1 − SS_res/SS_tot — protocol §4.4d):")
    for c in comparisons:
        if c.get("reproduced_r2") is not None:
            rec_r2 = c.get("recorded_r2")
            L.append(f"* {c['metric']}: reproduced **{c['reproduced_r2']:.4f}**"
                     + (f", recorded {rec_r2:.4f}" if rec_r2 is not None else ""))
    L.append("")

    L.append("## Per set\n")
    L.append("| Set | n | clusters | completeness MAE | contamination MAE | comp bias | cont bias |")
    L.append("|---|---|---|---|---|---|---|")
    for name in sorted(per_set):
        m = per_set[name]["metrics"]
        L.append(f"| {name} | {m['completeness']['n']:,} | {m['completeness']['n_clusters']:,} | "
                 f"{m['completeness']['mae']:.4f} | {m['contamination']['mae']:.4f} | "
                 f"{m['completeness']['bias']:+.4f} | {m['contamination']['bias']:+.4f} |")
    L.append("")

    if failures:
        L.append("## Failures\n")
        for f in failures:
            L.append(f"* {f}")
        L.append("")

    L.append("## What this does and does not establish\n")
    L.append("* It **does** establish that the released model artefact, the released "
             "9-mer list and normalisation parameters, and the released benchmark "
             "genomes together regenerate the pooled leakage-free headline numbers "
             "from raw FASTA, end to end, in one command.")
    L.append("* It **does not** establish that the model weights can be retrained "
             "bit-exactly: V5 training was never seeded (protocol §4.4b). The weights "
             "are pinned by SHA256 and inference is deterministic — see "
             "`results/revision/reproducibility/DETERMINISM.md`.")
    L.append("* Sets C and D from the submitted manuscript are **excluded**: their "
             "dominant genomes came from the training split (100 % of Set C, 90.3 % of "
             "Set D) and those results are withdrawn, not reproduced.")
    Path(a.output_md).write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"\n  {overall}: {a.output_md}")
    for f in failures:
        print(f"  FAILURE: {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
