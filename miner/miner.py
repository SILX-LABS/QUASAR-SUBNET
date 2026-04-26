#!/usr/bin/env python3
"""
Quasar SN24 Miner — One-time model commitment.

This is a one-shot script. It commits your distilled model to the chain
and exits. Once committed, your model is PERMANENT — you cannot update,
replace, or re-commit. One model per hotkey, forever.

Usage:
    # Dry run (validate without committing):
    python miner.py \\
        --model-repo user/my-quasar-model \\
        --wallet-name my_wallet \\
        --hotkey-name my_hotkey \\
        --dry-run

    # Full commitment (private model → auto-publish after commit):
    python miner.py \\
        --model-repo user/my-quasar-model \\
        --wallet-name my_wallet \\
        --hotkey-name my_hotkey \\
        --network finney \\
        --netuid 24 \\
        --hf-token hf_xxx \\
        --auto-publish

    # Test-only (alias for --dry-run):
    python miner.py \\
        --model-repo user/my-quasar-model \\
        --wallet-name my_wallet \\
        --hotkey-name my_hotkey \\
        --test-only
"""
import os
import sys
import json
import logging
import time

import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("quasar.miner")

try:
    from miner.constants import (
        TEACHER_MODEL,
        TEACHER_TOTAL_PARAMS_B,
        MAX_PARAM_RATIO,
        MIN_BITTENSOR_VERSION,
    )
except ModuleNotFoundError:
    from constants import (  # type: ignore
        TEACHER_MODEL,
        TEACHER_TOTAL_PARAMS_B,
        MAX_PARAM_RATIO,
        MIN_BITTENSOR_VERSION,
    )
DASHBOARD_URL = os.environ.get("QUASAR_DASHBOARD_URL", "http://localhost:3000")
API_URL = os.environ.get("QUASAR_PUBLIC_API_URL", "http://localhost:8000")

# ANSI colors
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def _check_bittensor():
    """Check bittensor is installed and meets version requirement. Exit with clear message if not."""
    try:
        import bittensor as bt
        version = getattr(bt, "__version__", "0.0.0")
        from packaging.version import Version
        if Version(version) < Version(MIN_BITTENSOR_VERSION):
            print(f"{RED}✗ bittensor {version} is too old. Required: >= {MIN_BITTENSOR_VERSION}{RESET}")
            print(f"  Older versions are missing set_reveal_commitment().")
            print(f"  {BOLD}Fix:{RESET} pip install --upgrade bittensor")
            sys.exit(1)
        return bt
    except ImportError:
        print(f"{RED}✗ bittensor is not installed.{RESET}")
        print(f"  {BOLD}Fix:{RESET} pip install bittensor>={MIN_BITTENSOR_VERSION}")
        sys.exit(1)


def _check_registration(subtensor, wallet, netuid):
    """
    Check if the hotkey is registered on the subnet.
    Returns (is_registered: bool, uid: int | None, metagraph).
    """
    try:
        metagraph = subtensor.metagraph(netuid)
        hotkey_str = wallet.hotkey.ss58_address
        if hotkey_str in metagraph.hotkeys:
            uid = metagraph.hotkeys.index(hotkey_str)
            return True, uid, metagraph
        return False, None, metagraph
    except Exception as e:
        logger.warning(f"Could not check registration: {e}")
        return False, None, None


def _verify_commitment_on_chain(subtensor, hotkey_str, netuid, expected_model, max_attempts=6, delay=10):
    """
    Poll the chain to verify the commitment is visible.
    Returns (verified: bool, block: int | None, data: str | None).
    """
    for attempt in range(1, max_attempts + 1):
        print(f"  Verifying on-chain... (attempt {attempt}/{max_attempts})", flush=True)
        try:
            revealed = subtensor.get_all_revealed_commitments(netuid)
            if hotkey_str in revealed and len(revealed[hotkey_str]) > 0:
                block, data = revealed[hotkey_str][-1]
                # Try to parse and check it matches
                try:
                    parsed = json.loads(data)
                    if parsed.get("model") == expected_model:
                        return True, block, data
                except (json.JSONDecodeError, TypeError):
                    pass
                # Even if can't parse, if it's there it's committed
                return True, block, data
        except Exception as e:
            if attempt < max_attempts:
                logger.debug(f"Verification attempt {attempt} failed: {e}")
        if attempt < max_attempts:
            time.sleep(delay)

    return False, None, None


def _make_repo_public(model_repo, hf_token):
    """Flip a HuggingFace repo from private to public."""
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.update_repo_visibility(repo_id=model_repo, private=False)


def _check_repo_visibility(model_repo, hf_token=None):
    """Check if a HuggingFace repo is public or private. Returns 'public', 'private', or 'error'."""
    try:
        from huggingface_hub import model_info as _model_info
        info = _model_info(model_repo, token=hf_token)
        return "private" if info.private else "public"
    except Exception as e:
        err = str(e).lower()
        if "404" in err:
            return "not_found"
        if "401" in err or "403" in err:
            return "private_no_access"
        return f"error: {e}"


@click.command()
@click.option("--network", default=lambda: os.getenv("NETWORK", "finney"),
              help="Bittensor network (default: finney)")
@click.option("--netuid", type=int, default=lambda: int(os.getenv("NETUID", "24")),
              help="Bittensor netuid (default: 24)")
@click.option("--wallet-name", required=True, help="Name of existing Bittensor wallet")
@click.option("--wallet-path", default="~/.bittensor/wallets", help="Path to wallet directory")
@click.option("--hotkey-name", required=True, help="Name of existing hotkey")
@click.option("--model-repo", required=True,
              help="HuggingFace repo e.g. 'user/my-quasar-model'")
@click.option("--revision", default=None, help="HF commit SHA (pinned at latest if omitted)")
@click.option("--force", is_flag=True, help="Skip the existing-commitment check (DANGEROUS)")
@click.option("--dry-run", is_flag=True,
              help="Run all validation checks without committing (safe to run)")
@click.option("--test-only", is_flag=True,
              help="Alias for --dry-run: run validation checks only")
@click.option("--hf-token", default=None,
              help="HuggingFace API token (for --auto-publish or private repo access)")
@click.option("--auto-publish", is_flag=True,
              help="Automatically make the HF repo public after on-chain commitment. "
                   "This lets you keep your model private until committed, preventing frontrunning.")
def main(network, netuid, wallet_name, wallet_path, hotkey_name, model_repo, revision,
         force, dry_run, test_only, hf_token, auto_publish):
    """
    Commit a distilled model to the distillation subnet.

    This is PERMANENT. Once committed, you cannot change your model.
    One commitment per hotkey, forever.

    \b
    Recommended workflow (prevents frontrunning):
      1. Upload model to a PRIVATE HuggingFace repo
      2. Run: python miner.py --model-repo user/model --auto-publish --hf-token hf_xxx ...
      3. The script commits on-chain first, then flips the repo to public
      4. No one can steal your model because it's private until after commitment

    \b
    Examples:
        # Validate only (no commitment):
        python miner.py --model-repo user/model --wallet-name w --hotkey-name h --dry-run

        # Commit with auto-publish (private → public):
        python miner.py --model-repo user/model --wallet-name w --hotkey-name h \\
            --auto-publish --hf-token hf_xxx --network finney --netuid 24

        # Commit (model already public):
        python miner.py --model-repo user/model --wallet-name w --hotkey-name h \\
            --network finney --netuid 24
    """
    is_dry_run = dry_run or test_only

    if auto_publish and not hf_token:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print(f"{RED}✗ --auto-publish requires --hf-token or HF_TOKEN env variable{RESET}")
            sys.exit(1)

    # ── Step 1: Check bittensor early ──────────────────────────────────
    bt = _check_bittensor()
    print(f"{GREEN}✓{RESET} bittensor {bt.__version__}")

    # ── Step 2: Load wallet ────────────────────────────────────────────
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_path)
    try:
        _ = wallet.hotkey
        hotkey_str = wallet.hotkey.ss58_address
        print(f"{GREEN}✓{RESET} Wallet loaded: {wallet_name}/{hotkey_name}")
        print(f"  Hotkey: {hotkey_str}")
    except Exception as e:
        print(f"{RED}✗ Could not load wallet '{wallet_name}' hotkey '{hotkey_name}'{RESET}")
        print(f"  Error: {e}")
        print(f"  {BOLD}Fix:{RESET} btcli wallet create --name {wallet_name} --hotkey {hotkey_name}")
        sys.exit(1)

    # ── Step 3: Check registration on subnet ───────────────────────────
    subtensor = bt.Subtensor(network=network)
    is_registered, uid, metagraph = _check_registration(subtensor, wallet, netuid)

    if not is_registered:
        print(f"{RED}✗ Hotkey {hotkey_str} is NOT registered on subnet {netuid}{RESET}")
        print(f"")
        print(f"  {BOLD}You must register before committing.{RESET}")
        print(f"  If you commit without registration, the commitment lands on-chain")
        print(f"  but the validator CANNOT see it (no UID in metagraph).")
        print(f"")
        print(f"  {BOLD}Fix:{RESET} Register first, then run this script again:")
        print(f"    btcli subnet register --netuid {netuid} --wallet-name {wallet_name} --hotkey {hotkey_name}")
        print(f"")
        print(f"  Or register with recycled TAO:")
        print(f"    btcli subnet pow-register --netuid {netuid} --wallet-name {wallet_name} --hotkey {hotkey_name}")
        sys.exit(1)
    else:
        print(f"{GREEN}✓{RESET} Registered on subnet {netuid} as {CYAN}UID {uid}{RESET}")

    # ── Step 4: Check for existing commitment ──────────────────────────
    existing_commitment = None
    try:
        revealed = subtensor.get_all_revealed_commitments(netuid)
        if hotkey_str in revealed and len(revealed[hotkey_str]) > 0:
            existing_block, existing_data = revealed[hotkey_str][-1]
            existing_commitment = (existing_block, existing_data)
    except Exception as e:
        logger.debug(f"Could not check existing commitments: {e}")

    if existing_commitment and not force:
        block, data = existing_commitment
        print(f"{RED}✗ COMMITMENT ALREADY EXISTS — CANNOT UPDATE{RESET}")
        print(f"  Hotkey: {hotkey_str}")
        print(f"  UID:    {uid}")
        print(f"  Block:  {block}")
        print(f"  Data:   {data}")
        print(f"")
        print(f"  This subnet enforces ONE commitment per hotkey, permanently.")
        print(f"  {BOLD}The validator WILL pick up this commitment{RESET} — no action needed.")
        print(f"")
        try:
            parsed = json.loads(data)
            model = parsed.get("model", "?")
            rev = parsed.get("revision", "?")[:12] if parsed.get("revision") else "?"
            print(f"  {CYAN}Your committed model:{RESET} {model} @ {rev}")
            # Check if model is accessible
            vis = _check_repo_visibility(model, hf_token)
            if vis == "public":
                print(f"  {GREEN}✓ Model is public and accessible{RESET}")
                print(f"  {GREEN}✓ The validator will evaluate it in the next epoch{RESET}")
            elif vis == "private" or vis == "private_no_access":
                print(f"  {YELLOW}⚠ Model is PRIVATE — validator cannot download it{RESET}")
                print(f"  {BOLD}Fix:{RESET} Make it public on HuggingFace, or use --auto-publish next time")
            else:
                print(f"  {YELLOW}⚠ Could not check model visibility: {vis}{RESET}")
        except (json.JSONDecodeError, TypeError):
            pass
        print(f"")
        print(f"  To change models, register a new hotkey (costs ~0.2τ).")
        sys.exit(1)

    # ── Step 5: Run model validation checks ────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Running model validation checks...")
    print(f"{'='*60}\n")

    from test_miner import run_all_checks, check_dependencies
    check_dependencies()

    results = run_all_checks(
        model_repo=model_repo,
        revision=revision,
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
        wallet_path=wallet_path,
        network=network,
        netuid=netuid,
    )

    if not results.all_passed:
        print(f"\n{RED}✗ Validation FAILED — {results.num_failed} check(s) did not pass.{RESET}")
        print(f"  Fix the issues above before committing.")
        sys.exit(1)

    print(f"\n{GREEN}✓ All validation checks passed{RESET}")

    if is_dry_run:
        print(f"\n{'='*60}")
        print(f"  {GREEN}DRY RUN COMPLETE — all checks passed ✓{RESET}")
        print(f"{'='*60}")
        print(f"  Your model is ready. Remove --dry-run to commit for real.")
        sys.exit(0)

    # ── Step 6: Resolve revision ───────────────────────────────────────
    from huggingface_hub import repo_info
    if not revision:
        info = repo_info(model_repo, repo_type="model", token=hf_token)
        revision = info.sha
    print(f"\n  Revision pinned: {CYAN}{revision}{RESET}")

    # ── Step 7: Check model visibility & auto-publish flow ─────────────
    visibility = _check_repo_visibility(model_repo, hf_token)

    if auto_publish and visibility == "public":
        print(f"{YELLOW}⚠ Model is already public — --auto-publish has no effect{RESET}")
        print(f"  (This is fine, proceeding with commitment)")
        auto_publish = False

    if not auto_publish and visibility in ("private", "private_no_access"):
        print(f"{YELLOW}⚠ Model is PRIVATE — the validator cannot download it after commitment{RESET}")
        print(f"  Options:")
        print(f"    1. Make it public on HuggingFace before committing")
        print(f"    2. Re-run with --auto-publish --hf-token YOUR_TOKEN")
        print(f"       (commits first, then auto-flips to public)")
        sys.exit(1)

    # ── Step 8: Interactive confirmation ───────────────────────────────
    commit_data = json.dumps({"model": model_repo, "revision": revision})

    print(f"\n{'='*60}")
    print(f"  ⚠️  {BOLD}IRREVERSIBLE COMMITMENT — PLEASE REVIEW{RESET}")
    print(f"{'='*60}")
    print(f"  Model:    {CYAN}{model_repo}{RESET}")
    print(f"  Revision: {CYAN}{revision}{RESET}")
    print(f"  Hotkey:   {hotkey_str}")
    print(f"  UID:      {uid}")
    print(f"  Network:  {network}")
    print(f"  Netuid:   {netuid}")
    if auto_publish:
        print(f"  Publish:  {YELLOW}Will auto-publish AFTER on-chain commit{RESET}")
    print(f"")
    print(f"  This commitment is {BOLD}PERMANENT{RESET}. You cannot update, replace,")
    print(f"  or re-commit. One model per hotkey, forever.")
    print(f"{'='*60}")
    print(f"")

    confirmation = input(f"  Type {BOLD}YES{RESET} to confirm commitment: ").strip()
    if confirmation != "YES":
        print(f"  Commitment cancelled.")
        sys.exit(0)

    # ── Step 9: Commit on-chain ────────────────────────────────────────
    print(f"\n  Submitting commitment to chain...", flush=True)

    try:
        subtensor.set_reveal_commitment(
            wallet=wallet,
            netuid=netuid,
            data=commit_data,
            blocks_until_reveal=1,
        )
    except Exception as e:
        print(f"\n{RED}✗ Commitment submission failed: {e}{RESET}")
        print(f"  The commitment was NOT recorded. Your hotkey is still available.")
        print(f"  Check your wallet balance and try again.")
        sys.exit(1)

    print(f"  {GREEN}✓ Commitment submitted!{RESET}")

    # ── Step 10: Verify commitment on-chain ────────────────────────────
    print(f"\n  Verifying commitment on-chain (this may take up to 60 seconds)...", flush=True)

    verified, block, data = _verify_commitment_on_chain(
        subtensor, hotkey_str, netuid, model_repo,
        max_attempts=6, delay=10,
    )

    if verified:
        print(f"\n{'='*60}")
        print(f"  {GREEN}✓ COMMITMENT VERIFIED ON-CHAIN{RESET}")
        print(f"{'='*60}")
        print(f"  Block:  {block}")
        print(f"  Data:   {data}")
        print(f"  UID:    {uid}")
        print(f"  Hotkey: {hotkey_str}")
    else:
        print(f"\n{YELLOW}⚠ Commitment not yet visible on-chain after 60 seconds.{RESET}")
        print(f"  This usually means the extrinsic was included but not yet finalized.")
        print(f"  Wait a few minutes and check:")
        print(f"    - {API_URL}/api/commitments")
        print(f"    - Or run this script again (it will show 'COMMITMENT ALREADY EXISTS')")
        print(f"  Do NOT re-commit — the commitment is likely already there.")

    # ── Step 11: Auto-publish if requested ─────────────────────────────
    if auto_publish:
        print(f"\n  Making model public on HuggingFace...", flush=True)
        try:
            _make_repo_public(model_repo, hf_token)
            print(f"  {GREEN}✓ Model is now PUBLIC{RESET}")
            print(f"  The validator can now download and evaluate it.")
        except Exception as e:
            print(f"  {RED}✗ Failed to make model public: {e}{RESET}")
            print(f"  {BOLD}You must manually make it public on HuggingFace{RESET}")
            print(f"  The commitment is on-chain, but the validator can't download a private model.")

    # ── Final status ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if verified:
        print(f"  {GREEN}✓ ALL DONE — The validator will evaluate your model{RESET}")
        print(f"    in the next epoch (~10 minutes).")
        print(f"")
        print(f"  Track progress at: {CYAN}{DASHBOARD_URL}{RESET}")
        print(f"  Your model: {CYAN}{model_repo}{RESET}")
        print(f"  Your UID:   {CYAN}{uid}{RESET}")
    else:
        print(f"  {YELLOW}⚠ Commitment submitted but not yet verified.{RESET}")
        print(f"  Check {API_URL}/api/commitments in a few minutes.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
