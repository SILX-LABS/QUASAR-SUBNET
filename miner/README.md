# Miner Tools for Quasar SN24

These scripts help miners validate and commit Quasar models to subnet 24.

## Base Checkpoint

Use the official SILX AI checkpoint as the starting point:

```text
silx-ai/Quasar-3B-A1B-Preview
```

Submitted models must keep the Quasar architecture, tokenizer behavior, and
shape-compatible safetensors weights from that checkpoint.

Key config values:

```json
{
  "model_type": "quasar",
  "vocab_size": 248320,
  "top_k": 4,
  "shared_expert_size": 3072,
  "routed_expert_size": 256
}
```

## Quick Start

```bash
pip install click huggingface_hub transformers safetensors bittensor

python miner/check_model.py --model-repo your-org/your-model
python miner/test_miner.py --model-repo your-org/your-model
```

Commit after checks pass:

```bash
python miner/miner.py \
  --wallet-name mywallet \
  --hotkey-name myhotkey \
  --model-repo your-org/your-model \
  --netuid 24 \
  --network finney \
  --dry-run
```

Remove `--dry-run` only when you are ready to publish the on-chain commitment.

## Rules

- One active model commitment per hotkey.
- Start from `silx-ai/Quasar-3B-A1B-Preview`.
- Preserve the official Quasar config and tokenizer behavior.
- Publish safetensors weights in a public Hugging Face repo.
- Do not upload GPTQ, AWQ, GGUF, FP8, or other quantized formats.
- Do not submit duplicate or copied weights.

## Files

| File | Purpose |
|------|---------|
| `miner.py` | Commit your model to Bittensor subnet 24 |
| `check_model.py` | Pre-submission validation |
| `test_miner.py` | Extended miner-side checks |
| `train.py` | Local distillation example |
