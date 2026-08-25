#!/usr/bin/env python3
"""
Build a deterministic, sorted genome list for `magicc predict --input-list`.

Sorting is natural-numeric on the trailing integer of the genome name where one
exists (genome_2 before genome_10), so the list -- and therefore the output row
order -- does not depend on filesystem enumeration order.

`--subsample N` takes the first N entries of that deterministic order. It exists
for fast structural tests of the workflow and is recorded in the output header
of the reproduction report; subsampled runs are never a reproduction of the
headline numbers.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FASTA_SUFFIXES = (".fasta", ".fa", ".fna", ".ffn", ".fas", ".fsa", ".seq")


def natural_key(p: Path):
    stem = p.name
    for suf in (".gz", ""):
        for ext in FASTA_SUFFIXES:
            if stem.endswith(ext + suf):
                stem = stem[: -len(ext + suf)]
                break
    m = re.search(r"(\d+)$", stem)
    return (stem[: m.start()] if m else stem, int(m.group(1)) if m else -1, stem)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta-dir", required=True)
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = Path(a.fasta_dir)
    files = [p for p in d.iterdir()
             if p.is_file() and any(p.name.endswith(e) or p.name.endswith(e + ".gz")
                                    for e in FASTA_SUFFIXES)]
    if not files:
        print(f"FATAL: no FASTA files under {d}", file=sys.stderr)
        return 1
    files.sort(key=natural_key)
    if a.subsample and a.subsample > 0:
        files = files[: a.subsample]

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(p) for p in files) + "\n", encoding="utf-8")
    print(f"  {len(files)} genomes -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
