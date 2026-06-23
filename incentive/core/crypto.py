"""Assignment-grant encryption helpers.

Production grants are sealed to native ED25519 hotkey public keys. The local
test double below is intentionally not exposed by runtime CLIs.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .protocol import AssignmentGrant, EncryptedAssignmentGrant
from .signatures import canonical_json, public_key_from_ss58, sha256_hex


class AssignmentEncryptor(Protocol):
    scheme: str

    def encrypt_for_hotkey(
        self,
        grant: AssignmentGrant,
        *,
        recipient_hotkey: str,
        recipient_public_key: bytes | str | None = None,
    ) -> EncryptedAssignmentGrant: ...


class AssignmentDecryptor(Protocol):
    scheme: str

    def decrypt(self, encrypted: EncryptedAssignmentGrant, *, expected_hotkey: str) -> AssignmentGrant: ...


def _xor_crypt(data: bytes, *, key: str) -> bytes:
    key_bytes = key.encode("utf-8")
    if not key_bytes:
        raise ValueError("xor key must be non-empty")
    return bytes(byte ^ key_bytes[index % len(key_bytes)] for index, byte in enumerate(data))


@dataclass
class DevAssignmentCrypto:
    """Deterministic local crypto used for tests and dry runs."""

    secret: str = "quasar-dev-assignment"
    scheme: str = "dev-xor-v1"

    def encrypt_for_hotkey(
        self,
        grant: AssignmentGrant,
        *,
        recipient_hotkey: str,
        recipient_public_key: bytes | str | None = None,
    ) -> EncryptedAssignmentGrant:
        plaintext = canonical_json(grant.to_dict())
        ciphertext = _xor_crypt(plaintext, key=f"{self.secret}:{recipient_hotkey}")
        return EncryptedAssignmentGrant(
            job_id=grant.job_id,
            run_id=grant.run_id,
            recipient_hotkey=recipient_hotkey,
            ciphertext_b64=base64.b64encode(ciphertext).decode("ascii"),
            crypto_scheme=self.scheme,
            recipient_public_key_hex=None,
            sender_public_key_hex=sha256_hex(self.secret.encode("utf-8")),
        )

    def decrypt(self, encrypted: EncryptedAssignmentGrant, *, expected_hotkey: str) -> AssignmentGrant:
        if encrypted.crypto_scheme != self.scheme:
            raise ValueError(f"unsupported assignment scheme {encrypted.crypto_scheme!r}")
        if encrypted.recipient_hotkey != expected_hotkey:
            raise ValueError("assignment grant recipient hotkey mismatch")
        ciphertext = base64.b64decode(encrypted.ciphertext_b64.encode("ascii"))
        plaintext = _xor_crypt(ciphertext, key=f"{self.secret}:{expected_hotkey}")
        return AssignmentGrant.from_dict(json.loads(plaintext.decode("utf-8")))


@dataclass
class Ed25519SealedBoxAssignmentCrypto:
    """Anonymous sealed-box assignment grants for native ED25519 hotkeys."""

    seed: bytes | None = None
    scheme: str = "ed25519-sealed-box-v1"

    @staticmethod
    def from_keyfile(path: str | Path) -> "Ed25519SealedBoxAssignmentCrypto":
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
        return Ed25519SealedBoxAssignmentCrypto(seed=seed)

    def encrypt_for_hotkey(
        self,
        grant: AssignmentGrant,
        *,
        recipient_hotkey: str,
        recipient_public_key: bytes | str | None = None,
    ) -> EncryptedAssignmentGrant:
        try:
            from nacl.public import SealedBox
            from nacl.signing import VerifyKey
        except ImportError as exc:
            raise ImportError("PyNaCl is required for ED25519 sealed-box assignment grants") from exc

        recipient_public_key_bytes = (
            _key_material_to_bytes(recipient_public_key, "recipient_public_key")
            if recipient_public_key is not None
            else public_key_from_ss58(recipient_hotkey)
        )
        plaintext = canonical_json(grant.to_dict())
        ciphertext = SealedBox(VerifyKey(recipient_public_key_bytes).to_curve25519_public_key()).encrypt(plaintext)
        return EncryptedAssignmentGrant(
            job_id=grant.job_id,
            run_id=grant.run_id,
            recipient_hotkey=recipient_hotkey,
            ciphertext_b64=base64.b64encode(ciphertext).decode("ascii"),
            crypto_scheme=self.scheme,
            recipient_public_key_hex=recipient_public_key_bytes.hex(),
        )

    def decrypt(self, encrypted: EncryptedAssignmentGrant, *, expected_hotkey: str) -> AssignmentGrant:
        if encrypted.crypto_scheme != self.scheme:
            raise ValueError(f"unsupported assignment scheme {encrypted.crypto_scheme!r}")
        if encrypted.recipient_hotkey != expected_hotkey:
            raise ValueError("assignment grant recipient hotkey mismatch")
        if self.seed is None:
            raise ValueError("ED25519 sealed-box decrypt requires a native hotkey seed")
        try:
            from nacl.public import SealedBox
            from nacl.signing import SigningKey
        except ImportError as exc:
            raise ImportError("PyNaCl is required for ED25519 sealed-box assignment grants") from exc

        signing_key = SigningKey(self.seed)
        if encrypted.recipient_public_key_hex:
            expected_public_key = bytes.fromhex(encrypted.recipient_public_key_hex)
            if signing_key.verify_key.encode() != expected_public_key:
                raise ValueError("assignment grant recipient public key does not match local hotkey")
        ciphertext = base64.b64decode(encrypted.ciphertext_b64.encode("ascii"))
        plaintext = SealedBox(signing_key.to_curve25519_private_key()).decrypt(ciphertext)
        return AssignmentGrant.from_dict(json.loads(plaintext.decode("utf-8")))


def _key_material_to_bytes(value: bytes | bytearray | str | None, field_name: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        text = value[2:] if value.startswith("0x") else value
        try:
            return bytes.fromhex(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be bytes or a hex string") from exc
    raise TypeError(f"{field_name} must be bytes or a hex string")
