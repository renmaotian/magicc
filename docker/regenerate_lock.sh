#!/usr/bin/env bash
# Regenerate docker/requirements-lock.txt (exact versions + sha256 hashes) by
# resolving docker/requirements.in inside the pinned base image.
#
#   bash docker/regenerate_lock.sh > docker/requirements-lock.body.txt
#
# The output is the body of the lock file; the comment header in
# requirements-lock.txt is maintained by hand.
set -euo pipefail

BASE="python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run --rm --cpus 4 -v "${HERE}:/w:ro" "${BASE}" bash -lc '
  pip install --quiet --upgrade "pip==24.2" >/dev/null 2>&1
  pip install --dry-run --report /tmp/r.json -r /w/requirements.in >/dev/null
  python - <<PY
import json
rep = json.load(open("/tmp/r.json"))
rows = []
for it in rep["install"]:
    md = it["metadata"]
    ai = it.get("download_info", {}).get("archive_info", {})
    h  = ai.get("hashes", {}).get("sha256") or ai.get("hash", "").split("=", 1)[-1]
    rows.append((md["name"].lower(), md["version"], h))
for n, v, h in sorted(rows):
    print(f"{n}=={v} --hash=sha256:{h}")
PY
'
