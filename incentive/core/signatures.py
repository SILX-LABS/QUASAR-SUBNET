"""Deterministic signing helpers for Quasar protocol records.

The local test path uses HMAC so the protocol can be exercised without a
Bittensor wallet. Production code should load a wallet-backed hotkey signer;
the bytes-to-sign surface stays the same either way.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_dict(value: dict[str, Any]) -> str:
    return sha256_hex(canonical_json(value))


def sign_payload(payload: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def verify_payload(payload: bytes, secret: str, signature: str) -> bool:
    return hmac.compare_digest(sign_payload(payload, secret), signature)


def sign_dict(value: dict[str, Any], secret: str) -> str:
    return sign_payload(canonical_json(value), secret)


def verify_dict(value: dict[str, Any], secret: str, signature: str) -> bool:
    return verify_payload(canonical_json(value), secret, signature)


class Signer(Protocol):
    identity: str

    def sign(self, payload: bytes) -> str: ...


class Verifier(Protocol):
    def verify(self, payload: bytes, signature: str, identity: str | None = None) -> bool: ...


@dataclass
class HmacSigner:
    secret: str
    identity: str = "dev-hmac"

    def sign(self, payload: bytes) -> str:
        return sign_payload(payload, self.secret)

    def verify(self, payload: bytes, signature: str, identity: str | None = None) -> bool:
        return verify_payload(payload, self.secret, signature)


def _is_hex(value: str) -> bool:
    try:
        bytes.fromhex(value)
        return True
    except ValueError:
        return False


def _signature_bytes(signature: str) -> bytes:
    value = signature.removeprefix("0x")
    return bytes.fromhex(value) if _is_hex(value) else signature.encode("utf-8")


def _looks_like_ss58(identity: str) -> bool:
    return 32 <= len(identity) <= 64 and identity.isalnum()


def public_key_from_ss58(identity: str) -> bytes:
    try:
        from substrateinterface.utils.ss58 import ss58_decode
    except Exception:
        try:
            from scalecodec.utils.ss58 import ss58_decode
        except Exception as exc:
            raise RuntimeError("substrateinterface or scalecodec is required to decode SS58 hotkeys") from exc
    return bytes.fromhex(ss58_decode(identity))


def verify_hotkey_payload(payload: bytes, identity: str, signature: str) -> bool:
    """Verify a payload against an SS58 hotkey when wallet libraries exist."""

    sig = _signature_bytes(signature)

    try:
        from bittensor_wallet import Keypair

        for crypto_type in (0, 1):
            try:
                if bool(Keypair(ss58_address=identity, crypto_type=crypto_type).verify(payload, sig)):
                    return True
            except Exception:
                continue
    except Exception:
        pass

    try:
        from substrateinterface import Keypair

        if bool(Keypair(ss58_address=identity).verify(payload, sig)):
            return True
    except Exception:
        pass

    try:
        from nacl.signing import VerifyKey

        VerifyKey(public_key_from_ss58(identity)).verify(payload, sig)
        return True
    except Exception:
        return False


def verify_identity_payload(
    payload: bytes,
    identity: str,
    signature: str,
    *,
    allow_dev_hmac: bool = True,
) -> bool:
    if _looks_like_ss58(identity):
        return verify_hotkey_payload(payload, identity, signature)
    return allow_dev_hmac and verify_payload(payload, identity, signature)


def verify_identity_dict(
    value: dict[str, Any],
    identity: str,
    signature: str,
    *,
    allow_dev_hmac: bool = True,
) -> bool:
    return verify_identity_payload(canonical_json(value), identity, signature, allow_dev_hmac=allow_dev_hmac)


@dataclass
class NativeEd25519HotkeySigner:
    """Signer for native ED25519 Bittensor hotkey files."""

    seed: bytes
    identity: str

    @staticmethod
    def from_keyfile(path: str | Path) -> "NativeEd25519HotkeySigner":
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        crypto_type = data.get("cryptoType")
        if crypto_type != 0:
            raise ValueError(f"expected native ED25519 cryptoType 0 hotkey, got {crypto_type!r}")
        seed_hex = str(data.get("secretSeed") or "").removeprefix("0x")
        if not seed_hex:
            raise ValueError("native ED25519 hotkey file is missing secretSeed")
        seed = bytes.fromhex(seed_hex)
        if len(seed) != 32:
            raise ValueError(f"expected 32-byte ED25519 seed, got {len(seed)} bytes")
        identity = str(data.get("ss58Address") or "")
        if not identity:
            raise ValueError("native ED25519 hotkey file is missing ss58Address")
        return NativeEd25519HotkeySigner(seed=seed, identity=identity)

    @staticmethod
    def from_wallet(*, wallet_path: str | Path, wallet_name: str, hotkey_name: str) -> "NativeEd25519HotkeySigner":
        keyfile = Path(wallet_path).expanduser() / wallet_name / "hotkeys" / hotkey_name
        return NativeEd25519HotkeySigner.from_keyfile(keyfile)

    def sign(self, payload: bytes) -> str:
        from nacl.signing import SigningKey

        return SigningKey(self.seed).sign(payload).signature.hex()

    def verify(self, payload: bytes, signature: str, identity: str | None = None) -> bool:
        return verify_hotkey_payload(payload, identity or self.identity, signature)


class BittensorHotkeySigner:
    """Small adapter around the Bittensor wallet hotkey signing surface."""

    def __init__(self, *, wallet: Any, identity: str | None = None) -> None:
        self.wallet = wallet
        self.identity = identity or getattr(getattr(wallet, "hotkey", None), "ss58_address", "bittensor-hotkey")

    def sign(self, payload: bytes) -> str:
        signer = getattr(getattr(self.wallet, "hotkey", None), "sign", None)
        if signer is None:
            raise RuntimeError("wallet hotkey does not expose sign(payload)")
        signature = signer(payload)
        return signature.hex() if isinstance(signature, bytes) else str(signature)

    def verify(self, payload: bytes, signature: str, identity: str | None = None) -> bool:
        verifier = getattr(getattr(self.wallet, "hotkey", None), "verify", None)
        if verifier is None:
            raise RuntimeError("wallet hotkey does not expose verify(payload, signature)")
        return bool(verifier(payload, _signature_bytes(signature)))


def load_wallet_hotkey_signer(
    *,
    wallet_path: str | Path,
    wallet_name: str,
    hotkey_name: str,
) -> Signer:
    try:
        return NativeEd25519HotkeySigner.from_wallet(
            wallet_path=wallet_path,
            wallet_name=wallet_name,
            hotkey_name=hotkey_name,
        )
    except ValueError:
        pass

    import bittensor as bt

    try:
        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name, path=str(wallet_path))
    except TypeError:
        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    return BittensorHotkeySigner(wallet=wallet)
