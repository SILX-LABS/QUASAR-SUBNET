# Quasar Subnet

**Bittensor subnet built to crush the long-context barrier | SN24 |**

Quasar is SILX Labs' competitive small-model subnet on Bittensor. Miners train
Quasar-compatible language models, publish them as public Hugging Face
repositories, and commit the pinned model revision on-chain. Validators verify
each valid commitment, score it with paired KL duels plus the production
composite evaluator, and set weights to the current king.

## Network

- Chain: Bittensor Finney
- Netuid: 24
- X: [`@QuasarModels`](https://x.com/QuasarModels)
- Base checkpoint: [`silx-ai/Quasar-3B-A1B-Preview`](https://huggingface.co/silx-ai/Quasar-3B-A1B-Preview)
- Launch teacher: [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B)
- Model family: Quasar 3B total / about 1B active Mixture-of-Experts
- Ranking: dethroning requires both a valid paired-KL win and a composite quality pass

## How It Works

Miners submit one public model repo per registered hotkey. Commitments are
permanent for that hotkey. If a commitment is disqualified, the miner can
register a new hotkey and submit a different model.

Validators score committed models with a composite evaluator that covers the
parts that matter for a useful long-context model:

- Distribution match: teacher support KL and on-policy distribution checks.
- Capability: math, code, reasoning, instruction following, tool use, long
  context, and robustness probes.
- Conversational quality: chat-turn and judge probes.
- Generation discipline: reasoning-density and collapse checks that penalize
  models that ramble, loop, or never answer.
- Robustness: block-seeded procedural prompts and prompt rewrites so miners
  cannot train against a static answer key.

The king is the strongest valid model under the composite rule. Validators set
one-hot weights on-chain: king `1.0`, everyone else `0.0`.

## King-of-the-Hill Evaluation

The validator uses a king-of-the-hill workflow for fast, high-confidence
ranking:

- Pre-checks run before GPU evaluation: architecture compliance, tokenizer
  compatibility, public revision integrity, safetensors availability, duplicate
  hashes, and quantization rejection.
- Each eligible commitment is evaluated on a deterministic prompt set seeded by
  the chain block context.
- The reference Quasar checkpoint is included so weak or broken axes can be
  handled consistently.
- Each model receives normalized per-axis scores plus aggregate `worst` and
  `weighted` composite scores.
- King selection uses both paired-KL evidence and composite quality. KL alone
  cannot crown a challenger, and composite alone cannot override a direct
  head-to-head regression.
- Weight setting happens after evaluation by assigning the current king the full
  validator weight. On-chain commit-reveal can make the visible chain weight lag
  the latest local eval winner until the reveal period completes.

## Validator Timing

Validators evaluate submissions in shared chain-coordinated rounds so the
network converges on the same winner. A valid public pinned Quasar repo may wait
until the next round before it is scored. The dashboard distinguishes the latest
local eval winner from the chain-revealed weight target when commit-reveal is
still pending.

Keep submitted repos public, pinned, valid, and unchanged after commit.

## Model Requirements

Submissions must match the official Quasar base interface. The canonical config
is the `config.json` in
[`silx-ai/Quasar-3B-A1B-Preview`](https://huggingface.co/silx-ai/Quasar-3B-A1B-Preview).

A valid model must:

- Use the Quasar architecture and tokenizer.
- Keep `vocab_size=248320`.
- Stay within the current subnet parameter cap.
- Provide public safetensors weights.
- Include the expected Quasar custom code files.
- Be loadable with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.
- Stay public and unchanged after the committed revision.
- Avoid quantized formats such as GPTQ, AWQ, GGUF, and FP8.
- Use unique weights that are not identical to an earlier committed model.
- Keep evaluation KL at or below the validator max KL threshold. The default is
  `4.0` for tail-bucket KL, configurable by validators with `QUASAR_MAX_KL_THRESHOLD`; models
  above the threshold are disqualified for quality.

## Mining Guide

Requirements:

- Bittensor wallet registered on subnet 24.
- Hugging Face account for model hosting.
- Training infrastructure of your choice.

Install the miner dependencies:

```bash
python -m pip install -r requirements-miner.txt
```

Check your model before committing:

```bash
python miner/check_model.py --model-repo your-username/your-model
python miner/test_miner.py --model-repo your-username/your-model
```

Submit a dry run first:

```bash
python miner/miner.py \
  --network finney \
  --netuid 24 \
  --wallet-name my_wallet \
  --hotkey-name my_hotkey \
  --model-repo your-username/your-model \
  --dry-run
```

Remove `--dry-run` only after the checks pass and you are ready to commit that
repo revision on-chain.

## Validator Guide

Requirements:

- Bittensor wallet registered as a validator on subnet 24.
- Python 3.10+.
- GPU capacity for the current evaluator.
- Local wallet keys kept on the validator host.

Quick start for local-GPU validators:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-validator.txt

python3 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install -U pip
.venv-vllm/bin/python -m pip install -r requirements-vllm.txt
```

Create an environment file outside the repo:

```bash
mkdir -p ~/.secrets
cat > ~/.secrets/quasar.env <<'EOF'
QUASAR_EVAL_BACKEND=local
QUASAR_NETWORK=finney
QUASAR_NETUID=24
QUASAR_WALLET_NAME=validator
QUASAR_HOTKEY_NAME=validator
QUASAR_WALLET_PATH=$HOME/.bittensor/wallets
QUASAR_STATE_DIR=$PWD/state
EOF

bash scripts/run_validator.sh
```

Do not install `vllm` manually into the validator environment. The local
backend uses `.venv-vllm/bin/python` automatically when it exists. Only set
`QUASAR_PYTHON` or `QUASAR_VLLM_PYTHON` if you intentionally use non-standard
virtualenv names.

If another service already uses the default vLLM port, set `QUASAR_VLLM_PORT`
to a free local port, for example `QUASAR_VLLM_PORT=9101`.

For remote Lium evaluation, set `QUASAR_EVAL_BACKEND=lium` and provide
`LIUM_API_KEY` in the same environment file.

Common validator settings:

```bash
QUASAR_NETWORK=finney
QUASAR_NETUID=24
QUASAR_WALLET_NAME=validator
QUASAR_HOTKEY_NAME=validator
QUASAR_WALLET_PATH=/path/to/wallets
QUASAR_STATE_DIR=/path/to/state
```

Keep wallet files, provider keys, Hugging Face tokens, and state credentials out
of git. Use a private environment file or your process manager's secret store.

For validator rollout updates, use the 2026-05-18 local-state reset block in
[`VALIDATOR.md`](VALIDATOR.md). The current production settings are
`QUASAR_EVAL_PROMPTS_H2H=300`, `QUASAR_VLLM_CONCURRENCY=8`, isolated
`.venv-vllm`, and a two-epoch chain coordination window.

For this rollout, pin the crown policy in each validator env file so all
operators run the same gates and no-winner behavior:

```bash
SINGLE_EVAL_MIN_CROWN_QUALITY=0.20
SINGLE_EVAL_MIN_CROWN_QUALITY_AXES=4
QUASAR_RESCORING_REVALIDATE_KING=1
QUASAR_NO_WINNER_FALLBACK_UID=155
QUASAR_NO_WINNER_FALLBACK_TO_VALIDATOR_UID=0
```

All validators should use the same no-winner fallback UID so vTrust stays
aligned while emissions are withheld from unsafe submissions. Do not set
`QUASAR_RESCORING_FALLBACK_UID` unless an explicit recovery target is announced.

## Disqualification

Models are disqualified for the current commitment when they fail production
checks:

- `COPY`: identical or near-identical weights to an earlier valid commitment.
- `REMOVED`: model deleted, made private, or changed after the committed
  revision.
- `INVALID`: incompatible architecture, tokenizer, custom code, parameter cap,
  format, or quantization.
- `EVAL_ERROR`: repeated non-transient failure during validator evaluation.

Disqualification is scoped to the commitment. The on-chain commitment remains
permanent, but the miner can register a new hotkey and submit a different model.

## Anti-Gaming

- Weight hashes and content hashes are tracked for duplicate detection.
- Earlier on-chain commitment owns an identical weight or content hash, even
  when the later commitment uses the same coldkey.
- Re-sharded copies are checked through content hashing.
- Public revision integrity is verified continuously.
- Quantized submissions are rejected.
- Prompt sets are block-seeded and not known before evaluation.
- Composite scoring prevents a model from winning on KL while failing core
  capability or generation-discipline checks.

## License

See `LICENSE`.
