# Miner Tools for Quasar SN24

Everything a miner needs is in this directory.

## Quick Start

```bash
# 1. Install dependencies
pip install click huggingface_hub transformers safetensors bittensor

# 2. Train your model (or use your own)
# See train.py for a community KL distillation example

# 3. Pre-check before committing (saves you from wasting registration fees)
python miner/check_model.py --model-repo your-org/your-model --eval

# 4. Commit your model to the chain (PERMANENT — one per hotkey)
python miner/miner.py \
  --wallet-name mywallet \
  --hotkey-name myhotkey \
  --model-repo your-org/your-model \
  --netuid 24 \
  --network finney \
  --auto-publish \
  --hf-token hf_xxx
```

## Files

| File | Purpose |
|------|---------|
| `miner.py` | Commit your model to Bittensor Subnet 24 |
| `check_model.py` | Pre-submission validation — checks params, tokenizer, anti-cheat |
| `train.py` | Example KL distillation training script |

## Rules

- **One commitment per hotkey**, forever. If DQ'd, register a new hotkey.
- Max **5.25B parameters**
- Must use the **same tokenizer** as teacher (vocab 248,044)
- **No quantization** (GPTQ/AWQ/FP8 rejected)
- Model must be **public** on HuggingFace with safetensors weights
- Weights must be **unique** (SHA256 duplicate detection)

## Teacher

`Qwen/Qwen3.5-35B-A3B` — 35B total, 3B active MoE, 248,044 vocab

Your goal: distill this into ≤5.25B params with the lowest KL divergence.

## Support

For questions, check the dashboard docs or open an issue.
