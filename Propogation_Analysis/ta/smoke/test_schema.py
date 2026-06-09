from __future__ import annotations

import unittest

from infra.schema import Label, ModelFacts, RawRollout, RunManifest, Signals


class SchemaSmokeTest(unittest.TestCase):
    def test_label_roundtrip_preserves_raw_rollouts(self) -> None:
        label = Label(
            span_id="span-1",
            A=0.25,
            real_dist={"yes": 0.75, "no": 0.25},
            diff_dist={"yes": 0.25, "no": 0.75},
            raw_rollouts=[
                RawRollout(cond="real", answer="yes", parse_ok=True, filter_sim=0.1),
                RawRollout(cond="diff", answer="no", parse_ok=True, filter_sim=0.9),
            ],
            R_real=2,
            R_diff_net=1,
            R_gross=3,
            parse_fail_rate=0.0,
            metric="tv",
        )

        restored = Label.from_dict(label.to_dict())
        self.assertEqual(restored.to_dict(), label.to_dict())

    def test_forward_compatible_signals_shape(self) -> None:
        signals = Signals(span_id="span-1")
        payload = signals.to_dict()
        for field_name in [
            "E",
            "P",
            "Pchained",
            "G",
            "Pprime",
            "PprimeChained",
            "D",
            "Dchained",
            "D_ig",
            "c",
            "position",
            "length",
            "meta",
        ]:
            self.assertIn(field_name, payload)

    def test_manifest_roundtrip_keeps_model_facts(self) -> None:
        manifest = RunManifest(
            git_sha="abc123",
            config_hash="cfg",
            lockfile_hash="lock",
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
            seeds={"trace": 11},
            phase=0,
            tier="foundation",
            started_at="2026-06-09T00:00:00+00:00",
        )
        restored = RunManifest.from_dict(manifest.to_dict())
        self.assertEqual(restored.to_dict(), manifest.to_dict())
