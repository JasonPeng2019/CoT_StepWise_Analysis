from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .schema import ModelFacts, RunManifest
from .store import stable_hash


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(repo_root: str | Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_manifest(
    *,
    repo_root: str | Path,
    config_payload: dict,
    lockfile_path: str | Path,
    model_facts: ModelFacts | None,
    seeds: dict[str, int],
    phase: int,
    tier: str,
) -> RunManifest:
    return RunManifest(
        git_sha=git_sha(repo_root),
        config_hash=stable_hash(config_payload),
        lockfile_hash=file_sha256(lockfile_path),
        model_facts=model_facts,
        seeds=seeds,
        phase=phase,
        tier=tier,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
