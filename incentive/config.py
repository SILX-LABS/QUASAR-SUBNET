"""Environment-driven production configuration for the incentive system."""

from __future__ import annotations

import os
from dataclasses import dataclass

from incentive.core.runtime import env_bool, env_int


DEFAULT_MODEL_ID = "silx-ai/Quasar-Preview"


@dataclass(frozen=True)
class ChainConfig:
    network: str = "test"
    netuid: int = 508
    wallet_name: str = "validator"
    hotkey_name: str = "validator"
    wallet_path: str = "~/.bittensor/wallets"
    dry_run: bool = False

    @staticmethod
    def from_env(prefix: str = "QUASAR") -> "ChainConfig":
        return ChainConfig(
            network=os.environ.get(f"{prefix}_NETWORK", "test"),
            netuid=env_int(f"{prefix}_NETUID", 508),
            wallet_name=os.environ.get(f"{prefix}_WALLET_NAME", "validator"),
            hotkey_name=os.environ.get(f"{prefix}_HOTKEY_NAME", "validator"),
            wallet_path=os.environ.get(f"{prefix}_WALLET_PATH", "~/.bittensor/wallets"),
            dry_run=env_bool(f"{prefix}_DRY_RUN", False),
        )


@dataclass(frozen=True)
class S3Config:
    bucket: str
    region: str = "us-east-1"
    endpoint_url: str | None = None
    access_key: str | None = None
    secret_key: str | None = None
    anonymous: bool = False

    @staticmethod
    def from_env(prefix: str = "QUASAR_S3") -> "S3Config":
        bucket = os.environ.get(f"{prefix}_BUCKET", "").strip()
        if not bucket:
            raise ValueError(f"{prefix}_BUCKET is required")
        return S3Config(
            bucket=bucket,
            region=os.environ.get(f"{prefix}_REGION", "us-east-1"),
            endpoint_url=os.environ.get(f"{prefix}_ENDPOINT_URL") or None,
            access_key=os.environ.get(f"{prefix}_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID") or None,
            secret_key=os.environ.get(f"{prefix}_SECRET_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
            anonymous=env_bool(f"{prefix}_ANONYMOUS", False),
        )


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = DEFAULT_MODEL_ID
    revision: str = "main"

    @staticmethod
    def from_env(prefix: str = "QUASAR") -> "ModelConfig":
        return ModelConfig(
            model_id=os.environ.get(f"{prefix}_MODEL_ID", DEFAULT_MODEL_ID),
            revision=os.environ.get(f"{prefix}_MODEL_REVISION", "main"),
        )
