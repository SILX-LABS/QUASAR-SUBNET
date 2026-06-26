# Quasar Training

Quasar Training is the production training system for continuing Quasar model
pretraining on Bittensor. It coordinates independent miners, validators, and a
subnet-operated orchestrator so real training work can be assigned, verified,
merged, and converted into subnet weights.

Our goal is to complete more than **10T additional pretraining tokens** on
Quasar Preview through the largest decentralized model-training run we can make
practical, verifiable, and repeatable.

The current base model is **Quasar Preview**:

- Model source: [`silx-ai/Quasar-Preview`](https://huggingface.co/silx-ai/Quasar-Preview)
- Architecture: Quasar MoE
- Revision: `main`
- Mainnet subnet: `netuid 24`
- Network: `finney`

The orchestrator starts from a published Quasar Preview checkpoint manifest,
splits the trainable model tensors into 24 fragments, assigns training leases to
miners, pulls live fragment states during training, merges validator-approved
live fragments, and periodically releases updated Quasar checkpoints for future
miners and restarts.

## System Roles

- **Miner:** trains assigned Quasar data and uploads signed training artifacts.
- **Validator:** verifies miner work, writes signed verdicts, and publishes weights.
- **Orchestrator:** subnet-operated service that assigns work, merges accepted updates, and releases checkpoints.

Most participants run either a miner or a validator. The orchestrator is run by the subnet operator.

## Live Training Flow

```text
orchestrator publishes the active run and current Quasar checkpoint
miners discover the run, download checkpoint/data through grants, and train
syncer asks live miners for fragment_id = global_step % 24
miners upload signed live fragment claims for their current fragment state
validators verify live claims, tensor contracts, hashes, and eval quality
orchestrator merges only validator-approved live fragments
orchestrator immediately publishes the updated absolute fragment back to miners
validators publish Bittensor weights from accepted live merge events
after fragments 0..23 are covered, orchestrator releases a full checkpoint
```

Rounds are assignment leases and audit windows. End-of-job receipts remain useful
for telemetry and audit, but live validator-approved fragment merges are the
authoritative training and scoring path. Miners do not evaluate themselves.
Validators do not assign miner work, merge updates, or release checkpoints.
Miners and validators use signed manifests plus scoped grants, not broad
operator credentials.

## Rewards And Weights

Rewards are based on accepted live merge events, not miner self-reported speed,
loss, or GPU names. The validator scores recent accepted live work with decay so
old work cannot dominate the subnet forever. Invalid, stale, replayed, unsigned,
or tensor-mismatched live claims receive zero merge weight.

Fast hardware still matters because it can keep serving valid live fragments
while training. The validator only counts work that was pulled by the syncer,
verified against the frozen previous fragment state, independently evaluated,
and accepted into the live merge ledger.

## Model And Checkpoints

Quasar Training works on Quasar Preview checkpoints, not placeholder smoke
models. A live checkpoint must include:

- the Quasar checkpoint archive or safetensors weights,
- tokenizer/config files required to load the model,
- `quasar_parameter_contract.json` or equivalent checkpoint metadata listing
  the trainable tensor names.

The parameter contract is required because the orchestrator/syncer owns global
fragment state and must split checkpoint tensors into fragments without importing
the training model code. Miners and validators load Quasar for training and
evaluation; the orchestrator coordinates tensor artifacts and checkpoint release.

Checkpoint release is assembled strictly from the latest absolute fragment
states. Delta artifacts are kept for merge/debug visibility, but released
checkpoints are built from absolute fragment state plus the base checkpoint tree.
The main release is not partial: it requires live accepted coverage for all 24
fragments since the previous release.

Checkpoint release is operational recovery infrastructure for late joiners and
restarts. The paper-aligned hot path is live fragment sync: accepted fragment,
merge now, broadcast updated fragment, repeat. Miners should not wait for every
full checkpoint release before continuing live training.

## Mainnet Defaults

```bash
export QUASAR_NETWORK=finney
export QUASAR_NETUID=24
export QUASAR_S3_BUCKET=quasar-incentive-sn24-529337356998-us-east-1
export QUASAR_S3_REGION=us-east-1
export QUASAR_S3_ANONYMOUS=true
```

Leave `QUASAR_RUN_ID` unset for normal mining and validation. Miners and
validators discover the active run from the public current-run metadata.

## Install

Clone the repo on the machine, then run the script for your role:

```bash
git clone https://github.com/SILX-LABS/QUASAR-SUBNET.git /workspace/quasar-incentive
cd /workspace/quasar-incentive
bash scripts/install_miner.sh /workspace/quasar-incentive
bash scripts/install_validator.sh /workspace/quasar-incentive
```

## Mainnet Miner Quick Start

Current mainnet settings:

```bash
export QUASAR_NETWORK=finney
export QUASAR_NETUID=24
export QUASAR_S3_BUCKET=quasar-incentive-sn24-529337356998-us-east-1
export QUASAR_S3_REGION=us-east-1
export QUASAR_S3_ANONYMOUS=true
export QUASAR_WALLET_PATH=~/.bittensor/wallets
export QUASAR_WALLET_NAME=<miner-wallet>
export QUASAR_HOTKEY_NAME=<miner-hotkey>
```

Run:

```bash
quasar-incentive miner run \
  --worker-id "$(hostname)-0" \
  --owner-identity 5GE25P2qGpGmjzGipqezZckMvyR2mpcsJS387bbcpitNSfm5
```

Leave `QUASAR_RUN_ID` unset. The miner discovers the live run automatically.
Keep the miner updated to the latest release before starting a new worker; the
current protocol expects signed live fragment claims and live-control grants.

## Common Environment

Set these for the role wallet and subnet:

```bash
export QUASAR_NETWORK=<network>
export QUASAR_NETUID=<netuid>
export QUASAR_WALLET_PATH=~/.bittensor/wallets
export QUASAR_WALLET_NAME=<wallet>
export QUASAR_HOTKEY_NAME=<hotkey>
```

Miners and external validators also use the subnet-provided public bucket metadata:

```bash
export QUASAR_S3_BUCKET=<bucket>
export QUASAR_S3_REGION=<region>
export QUASAR_S3_ANONYMOUS=true
```

Do not set `QUASAR_RUN_ID` for normal mining or validation. The active run is discovered automatically.

## Run A Miner

Create or use a registered ED25519 miner hotkey, then run:

```bash
quasar-incentive miner run \
  --worker-id <worker-id> \
  --owner-identity <orchestrator-hotkey>
```

For multi-GPU miners:

```bash
export QUASAR_MINER_DEVICES=0,1,2,3
quasar-incentive miner run \
  --worker-id <worker-id> \
  --owner-identity <orchestrator-hotkey>
```

More: [docs/mining.md](docs/mining.md)

## Run A Validator

Use a registered validator hotkey and set the orchestrator identity:

```bash
export QUASAR_OWNER_IDENTITY=<orchestrator-hotkey>
bash scripts/run_validator.sh
```

More: [docs/validator.md](docs/validator.md)

Validators publish weights from accepted work. If there is no accepted work to
assign yet, the validator publishes self-fallback to its own registered
validator hotkey so validator chain state stays live.

Validators do not merge model updates. Other validators only validate assigned
work, write verdicts, summarize accepted merge events, and set weights.

Validator scoring uses accepted live merge events. End-of-job receipts are audit
telemetry and should not reopen old merges or change already-accepted live
fragment state.

## Access Model

- Miners and external validators use public run metadata plus encrypted presigned grants.
- Miners verify orchestrator-signed jobs before training.
- Validators verify orchestrator-signed validation jobs and miner-signed receipts.
- Private training and validation artifacts are scoped to the assigned job.
- Operator bucket write credentials stay only on orchestrator-controlled machines.

## Development

```bash
python3 -m pip install -e ".[prod,dev]"
PYTHONPYCACHEPREFIX=/tmp/quasar-pycache pytest -q
```
