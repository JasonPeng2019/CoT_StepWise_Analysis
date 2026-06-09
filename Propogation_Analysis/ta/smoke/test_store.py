from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from infra.schema import Problem
from infra.store import ArtifactStore


class StoreSmokeTest(unittest.TestCase):
    def test_jsonl_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(tmpdir)
            key = store.build_key(
                phase=0,
                dataset="babi",
                problem_id="p1",
                artifact="problems",
                inputs={"version": 1},
            )
            records = [
                Problem(
                    problem_id="p1",
                    dataset="babi",
                    phase=0,
                    prompt="prompt",
                    gold_answer="yes",
                )
            ]
            path = store.write_records_jsonl(key, records)
            self.assertTrue(path.exists())
            restored = store.read_jsonl(key)
            self.assertEqual(restored[0]["problem_id"], "p1")

    def test_atomic_json_write_replaces_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            key = store.build_key(
                phase=0,
                dataset="babi",
                problem_id="p2",
                artifact="meta",
                inputs={"version": 1},
            )
            store.write_json(key, {"value": 1})
            store.write_json(key, {"value": 2})
            self.assertEqual(store.read_json(key)["value"], 2)

    def test_read_missing_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(tmpdir)
            with self.assertRaises(FileNotFoundError):
                store.read_json(Path(tmpdir) / "missing")
