# Quasar Validator Spec

Validators run the Quasar evaluation loop for SN24. They watch on-chain model commitments, pre-check submitted Hugging Face repos, run GPU evaluation, compute composite scores, and set weights to the current king.

---

## Hardware

### Recommended Production GPU
- 1x H100 80GB or better

### Practical Current Minimum
- 1x 48GB NVIDIA GPU, or equivalent setup with enough VRAM for the current evaluator
- 2x RTX 6000 Ada 48GB has been tested successfully

### Not Recommended for Production
- 24GB cards as the main validator GPU
- They may work for local testing or reduced settings, but should not be treated as the production validator target

### Host
- 16+ CPU cores
- 128GB+ RAM
- 500GB+ NVMe SSD
- Stable 1Gbps network
- Linux server, Ubuntu 22.04/24.04 preferred

---

## Software

- Python 3.10 or 3.11
- CUDA 12.x
- Recent NVIDIA driver
- Git
- Bittensor wallet registered as a validator on subnet 24
- Hugging Face token recommended
- Quasar attention dependency:

```bash
python -m pip install -r requirements-validator.txt
```

---

## Validator Runtime

### Coordination

Validators align evaluation by chain-coordinated rounds, so startup time should
not decide which submissions are scored. A validator may wait for the next round
before evaluating or setting weights.

Current coordination protocol: v2. Rounds are 720 blocks (two subnet epochs),
with activation at round start + 600 blocks. All validators must run the same
coordination protocol before the rollout activation window; mixing v1 and v2
will split round manifests.

Keep local state persistent. Only reset local validator state during an
announced rollout or explicit operator recovery.

For an announced catch-up reset, stop the validator, move `QUASAR_STATE_DIR`
aside, and restart clean. This clears stale scoring/eval history so the
reset-anchor can seat the frozen backlog in one clean round.

Duplicate policy is strict: if two commitments have identical weights,
identical tensor content, or a near-identical activation fingerprint, the
earlier on-chain commitment is canonical. Same-coldkey recommits are not exempt.

### Default Network Settings

```env
QUASAR_NETWORK=finney
QUASAR_NETUID=24
QUASAR_VALIDATOR_TEMPO=600
SINGLE_EVAL_MODE=1
QUASAR_EVAL_PROMPTS_H2H=300
QUASAR_VLLM_CONCURRENCY=8
```

### Single-Eval Crown Policy

For coordinated rollouts, pin these in operator env files so every validator
uses the same crown gates and no-winner behavior:

```env
SINGLE_EVAL_MIN_CROWN_QUALITY=0.20
SINGLE_EVAL_MIN_CROWN_QUALITY_AXES=4
QUASAR_RESCORING_REVALIDATE_KING=1
QUASAR_NO_WINNER_FALLBACK_TO_VALIDATOR_UID=0
```

The crown-quality score excludes relative KL axes (`kl` and `on_policy_rkl`) so
a model cannot become or remain king purely by minimizing token-distribution
loss while failing capability, judge, chat, or benchmark axes.

Only set this during explicit operator recovery:

```env
QUASAR_RESCORING_FALLBACK_UID=<uid>
```

`QUASAR_RESCORING_FALLBACK_UID` forces the fallback weight target if a scoring
policy migration uncrowns every model. Leave it unset unless a specific
fallback target is announced.

### Wallet Settings

```env
QUASAR_WALLET_NAME=validator
QUASAR_HOTKEY_NAME=validator
QUASAR_WALLET_PATH=/path/to/wallets
QUASAR_STATE_DIR=/path/to/state
```

### Remote GPU (Optional)

```env
QUASAR_EVAL_BACKEND=lium
LIUM_API_KEY=...
QUASAR_LIUM_POD_NAME=quasar-eval
```

### W&B Telemetry (Optional)

Validators can stream operational status to a shared Weights & Biases project.
Use one key per validator or a limited service-account key; do not share a
personal API key in public chat or commit it to the repo.

```env
WANDB_ENABLED=1
WANDB_PROJECT=sn24-validator
WANDB_ENTITY=<team-or-user>
VALIDATOR_NAME=<operator-or-validator-name>
WANDB_API_KEY=<secret-from-wandb>
```

Each validator should use a distinct `VALIDATOR_NAME` so runs are easy to
separate in the W&B dashboard.

---

## Running

```bash
git clone https://github.com/SILX-LABS/QUASAR-SUBNET.git
cd QUASAR-SUBNET
python -m pip install -r requirements-validator.txt
bash scripts/run_validator.sh
```

## Local State Reset (2026-05-18)

Use this when validators are asked to reset local validator state for
coordinated catch-up.

```bash
cd /path/to/QUASAR-SUBNET
git pull --ff-only

pm2 stop quasar-validator || true
pkill -TERM -f '[p]od_eval.py|[v]llm.entrypoints.openai.api_server|[V]LLM::EngineCore' || true

STATE_DIR="${QUASAR_STATE_DIR:-state}"
BACKUP_DIR="${STATE_DIR}.backup.$(date +%Y%m%d-%H%M%S)"
mv "$STATE_DIR" "$BACKUP_DIR" 2>/dev/null || true
mkdir -p "$STATE_DIR"

pm2 restart quasar-validator --update-env || pm2 start scripts/run_validator.sh --name quasar-validator
pm2 save
```

Verify:

```bash
pm2 status quasar-validator
pm2 logs quasar-validator --lines 120 --nostream
cat "${QUASAR_STATE_DIR:-state}/eval_progress.json" 2>/dev/null || echo "no active eval yet"
```

### Recommended Process Manager

```bash
pm2 start scripts/run_validator.sh --name quasar-validator
pm2 save
```

---

## Operational Rules

- Run one validator process per hotkey.
- Keep validator state persistent.
- Do not delete `state/` unless intentionally resetting.
- Keep wallet keys on the validator host.
- Do not put wallet keys on rented GPU machines.
- Monitor logs for eval failures, stale state, and weight-setting errors.
- Keep enough GPU headroom; future teacher upgrades may require larger GPUs.
