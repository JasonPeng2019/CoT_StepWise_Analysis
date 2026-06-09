from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from infra.manifest import build_manifest
from infra.schema import ModelFacts


class ManifestSmokeTest(unittest.TestCase):
    def test_manifest_hashes_config_and_lockfile(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmpdir:
            lockfile = Path(tmpdir) / "requirements.lock.txt"
            lockfile.write_text("example==1.0\n", encoding="utf-8")
            manifest = build_manifest(
                repo_root=repo_root,
                config_payload={"phase": 0, "tier": "foundation"},
                lockfile_path=lockfile,
                model_facts=ModelFacts(
                    L=28,
                    H=12,
                    Hkv=2,
                    group_size=6,
                    d=1536,
                    d_head=128,
                    eps=1e-6,
                    norm_type="rmsnorm",
                    rope_theta=10000.0,
                    model_type="qwen2",
                    attn_impl="eager",
                ),
                seeds={"trace": 1},
                phase=0,
                tier="foundation",
            )

        payload = manifest.to_dict()
        self.assertEqual(payload["phase"], 0)
        self.assertEqual(payload["tier"], "foundation")
        self.assertTrue(payload["config_hash"])
        self.assertTrue(payload["lockfile_hash"])
        json.dumps(payload)
