from __future__ import annotations

import hashlib


def derive_seed(global_seed: int, purpose: str, problem_id: str) -> int:
    payload = f"{global_seed}:{purpose}:{problem_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)
