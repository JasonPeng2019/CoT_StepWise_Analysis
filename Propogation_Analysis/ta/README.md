# Thought Anchors

This directory is the implementation workspace for the propagation-analysis "thought anchors"
experiment described in:

- `../Propogation_Analysis/thought_anchors_implementation.md`
- `../Propogation_Analysis/thought_anchors_experiments_v3 (4).md`

## Phase 0 setup status

This scaffold is prepared for Phase 0 implementation:

- Forward-looking repository layout from the implementation spec
- Local bootstrap script for dependencies
- Phase 0 config template for bAbI + DeepSeek-R1-Distill-Qwen-1.5B
- Asset fetch helper for the target model and the bAbI dataset

## Bootstrapping

```bash
cd ta
./scripts/bootstrap_phase0.sh
```

That script creates `.venv`, installs the required Python packages, and writes a lockfile with
exact installed versions.

## Fetching assets

```bash
cd ta
. .venv/bin/activate
python scripts/fetch_phase0_assets.py --download-model --download-babi
```

The model is cached under `data/cache/huggingface/` and the raw dataset under `data/raw/babi/`.
If the legacy bAbI tarball endpoint is unavailable, the helper falls back to a Hugging Face mirror
under `data/raw/babi_hf/`.

## Notes

- The implementation spec requests Python 3.11. This machine currently exposes Python 3.12, so the
  setup targets 3.12 unless/until a 3.11 interpreter is added.
- The full experiment writeup assumes `8xA100`. This machine currently has `2x TITAN RTX`, which is
  enough for foundation work and Phase 0 pilots, but likely not the final intended throughput.
