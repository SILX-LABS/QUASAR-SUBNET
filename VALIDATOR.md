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

Keep local state persistent. Only reset validator state during an announced
rollout or explicit operator recovery.

For an announced coordination reset, stop the validator, back up and clear the
local `QUASAR_STATE_DIR`, then pull/start the updated code with the other
validators. Do not wipe wallet keys or validator config.

### Default Network Settings

```env
QUASAR_NETWORK=finney
QUASAR_NETUID=24
QUASAR_VALIDATOR_TEMPO=600
SINGLE_EVAL_MODE=1
```

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
