from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import Record


def stable_hash(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def build_key(
        self,
        *,
        phase: int,
        dataset: str,
        problem_id: str,
        artifact: str,
        inputs: dict[str, Any],
    ) -> Path:
        digest = stable_hash(inputs)
        return self.root / f"phase_{phase}" / dataset / problem_id / f"{artifact}-{digest}"

    def exists(self, key: str | Path, suffix: str) -> bool:
        return Path(f"{key}{suffix}").exists()

    def read_json(self, key: str | Path) -> Any:
        path = Path(f"{key}.json")
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def write_json(self, key: str | Path, payload: Any) -> Path:
        path = Path(f"{key}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def write_jsonl(self, key: str | Path, payloads: list[dict[str, Any]]) -> Path:
        path = Path(f"{key}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")
        os.replace(tmp, path)
        return path

    def read_jsonl(self, key: str | Path) -> list[dict[str, Any]]:
        path = Path(f"{key}.jsonl")
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def write_records_jsonl(self, key: str | Path, records: list[Record]) -> Path:
        return self.write_jsonl(key, [record.to_dict() for record in records])

    def write_parquet(self, key: str | Path, rows: list[dict[str, Any]]) -> Path:
        path = Path(f"{key}.parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, tmp)
        os.replace(tmp, path)
        return path

    def read_parquet(self, key: str | Path) -> list[dict[str, Any]]:
        path = Path(f"{key}.parquet")
        if not path.exists():
            raise FileNotFoundError(path)
        return pq.read_table(path).to_pylist()
