# Quasar Subnet

Quasar is a Bittensor subnet by SILX Labs for competitive language-model
distillation. Miners submit compact Qwen-compatible models, validators evaluate
them against the reigning king model, and emissions are assigned to the best
model through a king-of-the-hill mechanism.

- Network: Bittensor Finney
- Netuid: 24
- Teacher: `Qwen/Qwen3.5-35B-A3B`
- Maximum student size: 5.25B total parameters
- Scoring: paired prompt-level KL divergence
- Production backend target: Akash
- Public dashboard data cache: Hippius S3

## Repository Layout

```text
api/                 FastAPI public API and compact dashboard feed
config/              Shared subnet configuration
eval/                Dataset, scoring, runtime, and pod abstractions
frontend/public/     Public dashboard assets
miner/               Miner submission and validation tools
ops/akash/           Akash SDL deployment configuration
scripts/             Validator runtime, pod eval, publishing, and utilities
tests/               Regression tests
```

## Mechanism Overview

Miners upload model weights to Hugging Face and commit the model repo on-chain.
Validators read commitments from subnet 24, pre-check model validity, and run
head-to-head GPU evaluations against the current king.

Each round evaluates the king, top contenders, and new challengers on identical
block-hash-seeded prompts. The validator records per-prompt KL values for the
king and each challenger. A challenger can dethrone the king only when its
paired prompt-level improvement is statistically significant and passes the
configured improvement threshold.

The dashboard exposes the key round metrics:

- `MU_HAT`: mean paired delta, `king_prompt_kl - challenger_prompt_kl`
- `LCB`: conservative 95% lower confidence bound for the paired delta
- `DELTA`: aggregate loss gap, `king_loss - challenger_loss`
- `KING_LOSS`: king KL/loss in the round
- `CHALL_LOSS`: challenger KL/loss in the round
- wall time: measured elapsed evaluation time

## Miner Quick Start

Install the project:

```bash
pip install -e .
```

Run local pre-checks before submitting:

```bash
python miner/check_model.py --model-repo your-username/your-model
python miner/test_miner.py --model-repo your-username/your-model
```

Commit a model on-chain:

```bash
python miner/miner.py \
  --network finney \
  --netuid 24 \
  --wallet-name my_wallet \
  --hotkey-name my_hotkey \
  --model-repo your-username/your-model \
  --dry-run
```

Remove `--dry-run` only after the checks pass. Commitments are permanent for
that hotkey and commit block.

## Model Requirements

Submitted models must:

- use the Qwen tokenizer and compatible vocabulary
- stay at or below 5.25B total parameters
- publish weights in safetensors format
- remain public and revision-stable on Hugging Face
- avoid GPTQ, AWQ, GGUF, FP8, or other quantized formats
- use compatible architecture and tensor shapes
- avoid duplicate or functionally copied weights

Invalid, private, changed, quantized, or copied models are disqualified for that
commitment.

## Validator Quick Start

Install the project and run the validator wrapper:

```bash
pip install -e .
bash scripts/run_validator.sh
```

The wrapper defaults to:

- network: `finney`
- netuid: `24`
- eval backend: `lium`
- state directory: `state/`

Important environment variables:

```bash
LIUM_API_KEY=...
QUASAR_WALLET_NAME=validator
QUASAR_HOTKEY_NAME=validator
QUASAR_WALLET_PATH=/path/to/wallets
QUASAR_STATE_DIR=/path/to/state
QUASAR_EVAL_BACKEND=lium
```

For local GPU validation, set:

```bash
QUASAR_EVAL_BACKEND=local
QUASAR_LOCAL_EVAL_DIR=/path/to/local_eval_runs
```

## Public API

The backend is a FastAPI service. Important endpoints:

```text
GET /api/health
GET /api/metagraph
GET /api/price
GET /api/commitments
GET /api/scores
GET /api/eval-progress
GET /api/queue
GET /api/h2h-latest
GET /api/h2h-history
GET /api/king-history
GET /api/dashboard.json
GET /api/eval-stream
```

`/api/dashboard.json` is the compact feed consumed by the public dashboard.

## Dashboard

The dashboard browser assets live in `frontend/public/`. The page is public
and lightweight, but the data is live. It polls:

```text
/api/dashboard.json
https://s3.hippius.com/quasar/dashboard.json
```

Use demo mode for visual checks:

```text
/index.html?demo=1
```

## Akash Backend Deployment

Build and push the API image:

```bash
docker build -f Dockerfile.api -t ghcr.io/SILX-LABS/quasar-api:0.8.0 .
docker push ghcr.io/SILX-LABS/quasar-api:0.8.0
```

Update `ops/akash/deploy.yaml` with the image tag you published, then deploy
that SDL through Akash Console or the Akash CLI. The SDL exposes container port
`3710` as public HTTP port `80`.

The backend reads state from `QUASAR_STATE_DIR`. In production, validators can
either run with state available to the backend, or publish the compact
`dashboard.json` feed to Hippius from the validator host.

## Dashboard JSON Publishing

To publish the compact dashboard feed to Hippius:

```bash
export QUASAR_BUCKET_ENDPOINT_URL=https://s3.hippius.com
export QUASAR_BUCKET_NAME=quasar
export QUASAR_BUCKET_KEY=dashboard.json
export QUASAR_BUCKET_REGION=decentralized
export QUASAR_BUCKET_ADDRESSING_STYLE=path
export QUASAR_DASHBOARD_SOURCE_URL=http://127.0.0.1:3710/api/dashboard.json

python scripts/publish_dashboard_json.py
```

Set credentials through environment variables:

```bash
QUASAR_BUCKET_ACCESS_KEY_ID=...
QUASAR_BUCKET_SECRET_ACCESS_KEY=...
```

Do not commit wallet files, API keys, Hugging Face tokens, or bucket secrets.

## Development Checks

```bash
PYTHONPYCACHEPREFIX=/tmp/quasar_pycache python3 -m py_compile \
  api/server.py api/routes/dashboard.py eval/runtime.py scripts/validator/results.py

cd frontend
npm run build
```

## License

See `LICENSE`.
