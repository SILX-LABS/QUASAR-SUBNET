# Quasar Subnet

Quasar is SILX Labs' Bittensor subnet for distributed language-model
distillation on subnet 24.

Miners start from the official Quasar base checkpoint, improve it through
distillation, publish a public Hugging Face model repo, and commit that repo
on-chain. Validators evaluate committed models head-to-head against the current
king and assign weight to the best valid model.

## Network

- Chain: Bittensor Finney
- Netuid: 24
- Base checkpoint: [`silx-ai/Quasar-3B-A1B-Preview`](https://huggingface.co/silx-ai/Quasar-3B-A1B-Preview)
- Model family: Quasar 3B total / about 1B active Mixture-of-Experts
- Scoring: paired prompt-level KL improvement against the current king

## Model Requirements

Submitted models must preserve the official Quasar checkpoint structure:

```json
{
  "model_type": "quasar",
  "vocab_size": 248320,
  "d_model": 1536,
  "n_heads": 12,
  "n_layers": 24,
  "d_ff": 4096,
  "head_dim": 128,
  "num_shared_experts": 1,
  "num_routed_experts": 64,
  "top_k": 4,
  "shared_expert_size": 3072,
  "routed_expert_size": 256,
  "quasar_layers": 4,
  "gated_layers": 2,
  "dense_input_layers": 4,
  "max_seq_len": 16384,
  "num_loops": 1,
  "rope_theta": 1000000.0,
  "hidden_act": "silu"
}
```

Submissions also need to be public Hugging Face repos with safetensors weights,
a stable pinned revision, and the same tokenizer behavior as the official base
checkpoint. Quantized uploads and incompatible architecture changes are not
accepted.

## Miner Quick Start

Install the project:

```bash
pip install -e .
```

Check your model before committing:

```bash
python miner/check_model.py --model-repo your-username/your-model
python miner/test_miner.py --model-repo your-username/your-model
```

Commit your model to subnet 24:

```bash
python miner/miner.py \
  --network finney \
  --netuid 24 \
  --wallet-name my_wallet \
  --hotkey-name my_hotkey \
  --model-repo your-username/your-model \
  --dry-run
```

Remove `--dry-run` only after local checks pass. Treat each on-chain model
commitment as permanent for that hotkey.

## Validator Quick Start

Install and run:

```bash
pip install -e .
bash scripts/run_validator.sh
```

Common environment variables:

```bash
QUASAR_WALLET_NAME=validator
QUASAR_HOTKEY_NAME=validator
QUASAR_WALLET_PATH=/path/to/wallets
QUASAR_STATE_DIR=/path/to/state
QUASAR_EVAL_BACKEND=lium
LIUM_API_KEY=...
```

For a validator with its own GPU:

```bash
QUASAR_EVAL_BACKEND=local
QUASAR_LOCAL_EVAL_DIR=/path/to/local_eval_runs
```

## Dashboard

The public dashboard shows the current king, active evaluations, queue, and
duel history. During an active validation it displays progress, `MU_HAT`,
`LCB`, aggregate loss gap, king loss, challenger loss, and wall time.

## Development Checks

```bash
PYTHONPYCACHEPREFIX=/tmp/quasar_pycache python3 -m py_compile \
  api/server.py api/routes/dashboard.py eval/runtime.py eval/model_checker.py \
  scripts/remote_validator.py scripts/validator/results.py

cd frontend
npm run build
```

## License

See `LICENSE`.
