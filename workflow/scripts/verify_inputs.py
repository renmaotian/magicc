#!/usr/bin/env python3
"""Verify the frozen MAGICC artefacts before the workflow spends any compute."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-card", required=True)
    ap.add_argument("--kmers", required=True)
    ap.add_argument("--normalization", required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    model = Path(a.model)
    card = json.loads(Path(a.model_card).read_text())

    model_sha = sha256(model)
    kmer_sha = sha256(Path(a.kmers))
    norm_sha = sha256(Path(a.normalization))
    n_kmers = sum(1 for ln in Path(a.kmers).read_text().splitlines() if ln.strip())

    checks = [
        ("model_sha256_matches_config", model_sha == a.expected_sha256,
         f"{model_sha} vs config {a.expected_sha256}"),
        ("model_sha256_matches_model_card", model_sha == card["onnx"]["sha256"],
         f"{model_sha} vs card {card['onnx']['sha256']}"),
        ("kmer_list_sha256_matches_model_card", kmer_sha == card["features"]["kmer_features"]["list_sha256"],
         f"{kmer_sha} vs card {card['features']['kmer_features']['list_sha256']}"),
        ("normalization_sha256_matches_model_card", norm_sha == card["features"]["normalization"]["sha256"],
         f"{norm_sha} vs card {card['features']['normalization']['sha256']}"),
        ("kmer_count_is_9249", n_kmers == card["features"]["kmer_features"]["n"],
         f"{n_kmers} vs card {card['features']['kmer_features']['n']}"),
    ]

    failed = [name for name, ok, _ in checks if not ok]
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        model=str(model), model_sha256=model_sha,
        model_version=card["frozen_model"]["model_version"],
        kmer_list_sha256=kmer_sha, n_kmers=n_kmers,
        normalization_sha256=norm_sha,
        checks=[dict(check=n, result="PASS" if ok else "FAIL", detail=d) for n, ok, d in checks],
        overall="PASS" if not failed else "FAIL",
    ), indent=2) + "\n", encoding="utf-8")

    if failed:
        print(f"FATAL: {len(failed)} artefact check(s) failed: {failed}", file=sys.stderr)
        return 1
    print(f"  all {len(checks)} artefact checks passed (model V{card['frozen_model']['model_version']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
