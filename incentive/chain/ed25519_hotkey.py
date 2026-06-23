"""Generate native ED25519 Bittensor hotkeys for encrypted assignment grants."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path


DEFAULT_WALLET_PATH = Path.home() / ".bittensor" / "wallets"


def _hotkey_file(wallet_path: Path, wallet_name: str, hotkey_name: str) -> Path:
    return wallet_path / wallet_name / "hotkeys" / hotkey_name


def _hotkey_pub_file(wallet_path: Path, wallet_name: str, hotkey_name: str) -> Path:
    return wallet_path / wallet_name / "hotkeys" / f"{hotkey_name}pub.txt"


def _require_native_ed25519_wallet() -> None:
    from bittensor_wallet import Wallet

    create_new_hotkey = inspect.signature(Wallet.create_new_hotkey)
    if "crypto_type" not in create_new_hotkey.parameters:
        raise RuntimeError(
            "Installed bittensor-wallet does not expose native ED25519 hotkey generation. "
            "Install the ED25519-capable wallet package from the miner/validator extra."
        )


def _self_test(seed: bytes, ss58_address: str) -> str:
    from nacl.public import SealedBox
    from nacl.signing import SigningKey, VerifyKey
    from scalecodec.utils.ss58 import ss58_decode

    public_key = bytes.fromhex(ss58_decode(ss58_address))
    signing_key = SigningKey(seed)
    if signing_key.verify_key.encode() != public_key:
        raise RuntimeError("generated private seed does not match SS58 public key")

    plaintext = b"https://example.invalid/quasar-presigned-link-self-test"
    ciphertext = SealedBox(VerifyKey(public_key).to_curve25519_public_key()).encrypt(plaintext)
    decrypted = SealedBox(signing_key.to_curve25519_private_key()).decrypt(ciphertext)
    if decrypted != plaintext:
        raise RuntimeError("ED25519-to-X25519 encryption self-test failed")
    return public_key.hex()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quasar-generate-ed25519-hotkey",
        description=(
            "Create a native ED25519 Bittensor hotkey. The SS58 hotkey can be "
            "registered normally and can decrypt Quasar encrypted assignment grants."
        ),
    )
    parser.add_argument("--wallet-name", required=True, help="Bittensor wallet/coldkey name")
    parser.add_argument("--hotkey", required=True, help="Hotkey name to create")
    parser.add_argument("--wallet-path", type=Path, default=DEFAULT_WALLET_PATH)
    parser.add_argument("--n-words", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_native_ed25519_wallet()

    from bittensor_wallet import Wallet
    from scalecodec.utils.ss58 import ss58_decode

    wallet = Wallet(name=args.wallet_name, hotkey=args.hotkey, path=str(args.wallet_path))
    wallet.create_new_hotkey(
        crypto_type=0,
        n_words=args.n_words,
        overwrite=args.overwrite,
        suppress=True,
        use_password=False,
    )

    private_path = _hotkey_file(args.wallet_path, args.wallet_name, args.hotkey)
    public_path = _hotkey_pub_file(args.wallet_path, args.wallet_name, args.hotkey)
    private_payload = json.loads(private_path.read_text(encoding="utf-8"))
    public_payload = json.loads(public_path.read_text(encoding="utf-8"))
    if private_payload.get("cryptoType") != 0 or public_payload.get("cryptoType") != 0:
        raise RuntimeError("wallet generated a non-ED25519 hotkey")

    ss58_address = str(private_payload["ss58Address"])
    public_key_hex = str(private_payload["publicKey"]).removeprefix("0x")
    seed = bytes.fromhex(str(private_payload["secretSeed"]).removeprefix("0x"))
    if bytes.fromhex(ss58_decode(ss58_address)).hex() != public_key_hex:
        raise RuntimeError("generated SS58 address does not decode to the hotkey public key")
    if public_payload.get("ss58Address") != ss58_address:
        raise RuntimeError("hotkey public file does not match the private hotkey file")
    self_test_public_key_hex = _self_test(seed, ss58_address)

    print("Created ED25519 Quasar miner hotkey")
    print(f"wallet: {args.wallet_name}")
    print(f"hotkey: {args.hotkey}")
    print(f"ss58_address: {ss58_address}")
    print(f"public_key_hex: {self_test_public_key_hex or public_key_hex}")
    print(f"native_hotkey_file: {private_path}")
    print(f"native_hotkey_pub_file: {public_path}")
    print("encryption_self_test: ok")
    print()
    print("Register with:")
    print(
        "btcli subnets register "
        f"--netuid <NETUID> --wallet-name {args.wallet_name} "
        f"--hotkey {args.hotkey} --wallet-path {args.wallet_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
