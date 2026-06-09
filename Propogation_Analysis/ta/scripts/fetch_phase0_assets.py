#!/usr/bin/env python3
"""Fetch the Phase 0 model and raw bAbI data into repo-local cache directories."""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path


BABI_URLS = [
    "http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz",
    "https://s3.amazonaws.com/text-datasets/tasks_1-20_v1-2.tar.gz",
]
BABI_HF_DATASET = "Muennighoff/babi"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def download_babi(raw_root: Path) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    archive_path = raw_root / "tasks_1-20_v1-2.tar.gz"
    extracted_tree = raw_root / "tasks_1-20_v1-2"
    stable_link = raw_root / "babi"
    mirror_root = raw_root / "babi_hf"

    if mirror_root.exists() or stable_link.exists() or extracted_tree.exists():
        if extracted_tree.exists() and not stable_link.exists():
            stable_link.symlink_to(extracted_tree, target_is_directory=True)
        return

    download_babi_hf_mirror(raw_root)
    if mirror_root.exists():
        return

    if not archive_path.exists():
        headers = {"User-Agent": "Mozilla/5.0"}
        downloaded = False
        for url in BABI_URLS:
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=15) as response, archive_path.open("wb") as sink:
                    sink.write(response.read())
                downloaded = True
                break
            except Exception:
                continue
        if not downloaded:
            download_babi_hf_mirror(raw_root)
            return

    if not extracted_tree.exists():
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(raw_root)
    if extracted_tree.exists() and not stable_link.exists():
        stable_link.symlink_to(extracted_tree, target_is_directory=True)


def download_babi_hf_mirror(raw_root: Path) -> None:
    from huggingface_hub import snapshot_download

    mirror_root = raw_root / "babi_hf"
    if mirror_root.exists():
        return
    snapshot_download(
        repo_id=BABI_HF_DATASET,
        repo_type="dataset",
        local_dir=str(mirror_root),
        ignore_patterns=[".gitattributes"],
    )


def download_model(cache_root: Path) -> None:
    from huggingface_hub import snapshot_download

    cache_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=str(cache_root),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.onnx", "*.msgpack"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--download-babi", action="store_true")
    args = parser.parse_args()

    if not args.download_model and not args.download_babi:
        parser.error("Select at least one of --download-model or --download-babi.")

    raw_root = args.root / "data" / "raw"
    cache_root = args.root / "data" / "cache" / "huggingface"

    if args.download_babi:
        download_babi(raw_root)
    if args.download_model:
        download_model(cache_root)


if __name__ == "__main__":
    main()
