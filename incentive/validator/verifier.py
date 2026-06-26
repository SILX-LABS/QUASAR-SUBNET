"""Validator-side receipt and artifact verification."""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, replace
from math import ceil
from pathlib import Path

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore
from incentive.bucket.transport import GrantTransport, PresignedArtifactTransport
from incentive.coordination.mesh import DilocoMessage
from incentive.core.protocol import (
    ArtifactDigest,
    LiveFragmentClaim,
    LiveFragmentVerdict,
    MinerReceipt,
    TrainingJobManifest,
    ValidatorVerdict,
    WorkerIdentity,
    PresignedUrlGrant,
)
from incentive.core.signatures import Signer
from incentive.fragments.artifacts import (
    FRAGMENT_SYNC_FORMAT,
    FragmentTensorSpec,
    FragmentUpdateArtifact,
    FragmentUpdateManifest,
    load_fragment_update_artifact,
    load_safetensors_file,
    validate_fragment_state_tensors,
)
from .generalization import GeneralizationPolicy, evaluate_delta_generalization
from .resource_sanity import trusted_tokens_per_step, validate_receipt_resource_claims
from .rewards import expected_training_units_from_manifest, quality_multiplier_from_summary
from typing import Any


def evaluate_quasar_update_receipt(*args: Any, **kwargs: Any):
    from .quasar_update_eval import evaluate_quasar_update_receipt as _evaluate

    return _evaluate(*args, **kwargs)


@dataclass
class VerificationResult:
    receipt: MinerReceipt
    verdict: ValidatorVerdict
    verdict_uri: str


@dataclass
class LiveFragmentVerificationResult:
    claim: LiveFragmentClaim
    verdict: LiveFragmentVerdict
    verdict_uri: str


@dataclass
class ValidatorVerifierConfig:
    netuid: int
    run_id: str
    validator_hotkey: str
    independent_quasar_eval: Any | None = None
    owner_identity: str = ""
    allow_dev_signatures: bool = False


class ValidatorVerifier:
    def __init__(self, *, bucket: ObjectStore, config: ValidatorVerifierConfig, signer: Signer | str) -> None:
        self.bucket = bucket
        self.config = config
        self.signer = signer

    def _materialize_object(
        self,
        uri: str,
        target: Path,
        *,
        expected_sha256: str | None = None,
    ) -> tuple[str, int]:
        return self.bucket.get_to_path(uri, target, expected_sha256=expected_sha256)

    def verify_receipt_uri(self, receipt_uri: str) -> VerificationResult:
        receipt = MinerReceipt.from_dict(self.bucket.get_json(receipt_uri))
        manifest_uri = self.bucket.uri_for_key(paths.job_manifest_key(self.config.netuid, receipt.run_id, receipt.job_id))
        manifest = TrainingJobManifest.from_dict(self.bucket.get_json(manifest_uri))

        status = "pass"
        reasons: list[str] = []
        summary: dict[str, object] = {}

        if receipt.manifest_hash != manifest.manifest_hash:
            status = "fail"
            reasons.append("manifest hash mismatch")
        if self.config.owner_identity and not manifest.verify_signature(
            self.config.owner_identity,
            allow_dev_hmac=self.config.allow_dev_signatures,
        ):
            status = "fail"
            reasons.append("manifest owner signature failed")
        if receipt.worker.hotkey_ss58 != manifest.assigned_hotkey:
            status = "fail"
            reasons.append("receipt hotkey mismatch")
        if not receipt.verify_signature(
            receipt.worker.hotkey_ss58,
            allow_dev_hmac=self.config.allow_dev_signatures,
        ):
            status = "fail"
            reasons.append("bad miner signature")

        if manifest.deadline_unix >= 1_000_000_000:
            late_by_sec = max(0.0, float(receipt.finished_unix) - float(manifest.deadline_unix))
            summary["timing"] = {
                "request_expires_unix": int(manifest.deadline_unix),
                "finished_unix": float(receipt.finished_unix),
                "finished_after_request_expiry": late_by_sec > 0.0,
                "late_by_sec": late_by_sec,
            }

        output_results = self._check_output_digests(manifest, receipt.output_digests)
        summary["outputs"] = output_results
        if any(not item["ok"] for item in output_results):
            status = "fail"
            reasons.append("output digest mismatch")

        gpu_proof = self._check_gpu_proof(manifest, receipt)
        summary["gpu_proof"] = gpu_proof
        if not gpu_proof["ok"]:
            status = "fail"
            reasons.append(str(gpu_proof["reason"]))

        fragment_check, fragment_artifact = self._check_fragment_update(manifest, receipt)
        summary["fragment_update"] = fragment_check
        if not fragment_check["ok"]:
            status = "fail"
            reasons.append(str(fragment_check["reason"]))

        independent_eval_config = self._independent_eval_config(manifest)
        if independent_eval_config is not None:
            if status == "pass":
                try:
                    independent_report = evaluate_quasar_update_receipt(
                        bucket=self.bucket,
                        manifest=manifest,
                        receipt=receipt,
                        config=independent_eval_config,
                        artifact=fragment_artifact,
                    )
                    summary["independent_quasar_eval"] = independent_report.to_dict()
                    if independent_report.status == "fail":
                        status = "fail"
                        reasons.extend(independent_report.reasons)
                except Exception as exc:
                    status = "fail"
                    reasons.append(f"independent Quasar eval failed: {exc}")
                    summary["independent_quasar_eval"] = {"status": "error", "error": str(exc)}
            else:
                summary["independent_quasar_eval"] = {"status": "skipped", "reason": "basic receipt checks failed"}
        else:
            generalization_policy = GeneralizationPolicy.from_dict(
                dict(manifest.validation_policy.get("generalization") or {})
            )
            if manifest.validation_policy.get("generalization") or fragment_artifact is not None:
                report = evaluate_delta_generalization(
                    metrics=receipt.metrics,
                    artifact=fragment_artifact,
                    policy=generalization_policy,
                )
                summary["generalization"] = report.to_dict()
                if report.status == "fail":
                    status = "fail"
                    reasons.extend(report.reasons)

        if status == "pass" and receipt.claimed_local_steps <= 0:
            status = "fail"
            reasons.append("no claimed local steps")
        token_claim = self._check_claimed_tokens(manifest, receipt)
        summary["claimed_tokens"] = token_claim
        if not token_claim["ok"]:
            status = "fail"
            reasons.append(str(token_claim["reason"]))
        resource_claim = validate_receipt_resource_claims(manifest, receipt)
        summary["resource_sanity"] = resource_claim.to_dict()
        if not resource_claim.ok:
            status = "fail"
            reasons.append(resource_claim.reason)

        accepted_update_weight = quality_multiplier_from_summary(summary) if status == "pass" else 0.0
        if status == "pass":
            summary["quality_multiplier"] = accepted_update_weight

        verdict = ValidatorVerdict(
            verdict_id=f"{receipt.receipt_id}:validator={self.config.validator_hotkey}",
            receipt_id=receipt.receipt_id,
            manifest_hash=receipt.manifest_hash,
            job_id=receipt.job_id,
            run_id=receipt.run_id,
            miner_hotkey=receipt.worker.hotkey_ss58,
            validator_hotkey=self.config.validator_hotkey,
            status=status,
            reason="; ".join(reasons) if reasons else "ok",
            estimated_training_units=self._expected_training_units(manifest) if status == "pass" else 0.0,
            accepted_update_weight=accepted_update_weight,
            validation_summary=summary,
            checked_unix=time.time(),
        ).sign(self.signer)

        verdict_uri = self.bucket.uri_for_key(
            paths.verdict_key(self.config.netuid, receipt.run_id, self.config.validator_hotkey, receipt.receipt_id)
        )
        self.bucket.put_json(verdict_uri, verdict.to_dict())
        return VerificationResult(receipt=receipt, verdict=verdict, verdict_uri=verdict_uri)

    def verify_live_fragment_claim_uri(
        self,
        claim_uri: str,
        *,
        verdict_put_grant: PresignedUrlGrant | None = None,
        transport: GrantTransport | None = None,
    ) -> LiveFragmentVerificationResult:
        claim = LiveFragmentClaim.from_dict(self.bucket.get_json(claim_uri))
        status = "pass"
        reasons: list[str] = []
        summary: dict[str, object] = {"claim_uri": claim_uri}
        manifest: TrainingJobManifest | None = None
        artifact: FragmentUpdateArtifact | None = None

        def fail(reason: str) -> None:
            nonlocal status
            status = "fail"
            reasons.append(reason)

        if claim.run_id != self.config.run_id:
            fail("claim run_id mismatch")
        if not claim.miner_hotkey:
            fail("claim missing miner hotkey")
        elif not claim.verify_signature(claim.miner_hotkey, allow_dev_hmac=self.config.allow_dev_signatures):
            fail("bad live claim miner signature")
        if claim.learner_id and ":" in claim.learner_id:
            learner_hotkey, learner_worker = claim.learner_id.split(":", 1)
            if learner_hotkey != claim.miner_hotkey:
                fail("claim learner hotkey mismatch")
            if claim.worker_id and learner_worker != claim.worker_id:
                fail("claim learner worker mismatch")
        if claim.fragment_count <= 0 or not (0 <= claim.fragment_id < claim.fragment_count):
            fail("claim fragment id/count mismatch")
        if claim.local_step < claim.target_local_step:
            fail("claim local step below frozen target")
        if not claim.fragment_state_uri or not claim.fragment_state_sha256:
            fail("claim missing learner fragment state")
        if not claim.previous_fragment_state_uri or not claim.previous_fragment_state_sha256:
            fail("claim missing frozen previous fragment state")
        if not claim.job_id:
            fail("claim missing job_id")

        if claim.job_id:
            try:
                manifest = TrainingJobManifest.from_dict(
                    self.bucket.get_json(self.bucket.uri_for_key(paths.job_manifest_key(self.config.netuid, claim.run_id, claim.job_id)))
                )
                summary["assignment"] = {
                    "job_id": manifest.job_id,
                    "round_id": int(manifest.round_id),
                    "assigned_hotkey": manifest.assigned_hotkey,
                }
                if manifest.assigned_hotkey != claim.miner_hotkey:
                    fail("claim miner hotkey does not match assignment")
                if manifest.run_id != claim.run_id:
                    fail("claim assignment run mismatch")
            except Exception as exc:
                fail(f"claim assignment unavailable: {type(exc).__name__}: {exc}")

        request_payload = self._live_pull_request_payload(claim)
        if request_payload is None:
            fail("claim pull_fragment request unavailable")
        else:
            request_fragment_id = self._as_int(request_payload.get("fragment_id"), default=-1)
            request_fragment_count = self._as_int(request_payload.get("fragment_count"), default=0)
            request_global_step = self._as_int(request_payload.get("global_step"), default=0)
            request_round_id = self._as_int(request_payload.get("round_id"), default=0)
            request_target_local_step = self._as_int(request_payload.get("target_local_step"), default=0)
            summary["pull_fragment_request"] = {
                "request_id": str(request_payload.get("request_id") or ""),
                "fragment_id": request_fragment_id,
                "fragment_count": request_fragment_count,
                "global_step": request_global_step,
                "target_local_step": request_target_local_step,
            }
            if str(request_payload.get("request_id") or "") != claim.request_id:
                fail("claim request_id not bound to pull_fragment")
            if request_fragment_id != int(claim.fragment_id):
                fail("claim fragment_id not bound to pull_fragment")
            if request_fragment_count != int(claim.fragment_count):
                fail("claim fragment_count not bound to pull_fragment")
            if request_global_step != int(claim.global_step):
                fail("claim global_step not bound to pull_fragment")
            if request_round_id != int(claim.round_id):
                fail("claim round_id not bound to pull_fragment")
            if request_target_local_step != int(claim.target_local_step):
                fail("claim target_local_step not bound to pull_fragment")
            if str(request_payload.get("previous_fragment_state_uri") or "") != claim.previous_fragment_state_uri:
                fail("claim previous fragment URI not bound to pull_fragment")
            if str(request_payload.get("previous_fragment_state_sha256") or "") != claim.previous_fragment_state_sha256:
                fail("claim previous fragment sha not bound to pull_fragment")

        trusted_tokens, trusted_steps, work_summary = self._trusted_live_claim_work(claim, manifest)
        summary["work"] = work_summary
        if trusted_steps <= 0 or trusted_tokens <= 0:
            fail("claim has no positive trusted fragment work")

        if status == "pass":
            try:
                artifact, tensor_summary = self._live_claim_artifact(
                    claim=claim,
                    manifest=manifest,
                    trusted_tokens=trusted_tokens,
                    trusted_steps=trusted_steps,
                )
                summary["fragment_update"] = tensor_summary
            except Exception as exc:
                fail(f"live fragment tensor validation failed: {type(exc).__name__}: {exc}")
                summary["fragment_update"] = {"status": "error", "error": str(exc)}

        independent_eval_config = self._independent_eval_config(manifest) if manifest is not None else None
        if status == "pass" and independent_eval_config is not None and artifact is not None and manifest is not None:
            try:
                synthetic_receipt = MinerReceipt(
                    receipt_id=f"live:{claim.request_id}:{claim.learner_id}",
                    manifest_hash=manifest.manifest_hash or manifest.compute_manifest_hash(),
                    job_id=claim.job_id,
                    run_id=claim.run_id,
                    round_id=int(claim.round_id),
                    global_step=int(claim.global_step),
                    worker=WorkerIdentity(
                        hotkey_ss58=claim.miner_hotkey,
                        worker_id=claim.worker_id,
                    ),
                    input_digests=[],
                    output_digests=[],
                    started_unix=0.0,
                    finished_unix=time.time(),
                    compute_sec=0.0,
                    claimed_tokens=trusted_tokens,
                    claimed_local_steps=trusted_steps,
                    claimed_bytes_read=0,
                    claimed_bytes_written=0,
                    metrics={
                        "fragment_trained_tokens": trusted_tokens,
                        "fragment_local_steps": trusted_steps,
                    },
                )
                independent_report = evaluate_quasar_update_receipt(
                    bucket=self.bucket,
                    manifest=manifest,
                    receipt=synthetic_receipt,
                    config=independent_eval_config,
                    artifact=artifact,
                )
                summary["independent_quasar_eval"] = independent_report.to_dict()
                if independent_report.status == "fail":
                    fail("; ".join(independent_report.reasons) or "independent Quasar eval failed")
            except Exception as exc:
                fail(f"independent Quasar eval failed: {type(exc).__name__}: {exc}")
                summary["independent_quasar_eval"] = {"status": "error", "error": str(exc)}
        elif status == "pass" and independent_eval_config is None:
            fail("live fragment validation requires independent Quasar eval")
            summary["independent_quasar_eval"] = {"status": "fail", "reason": "not configured"}

        quality = quality_multiplier_from_summary(summary) if status == "pass" else 0.0
        paper_weight = float(trusted_tokens) * (float(trusted_tokens) / float(max(1, trusted_steps))) if trusted_tokens > 0 else 0.0
        accepted_weight = paper_weight * max(0.0, float(quality)) if status == "pass" else 0.0
        if status == "pass":
            summary["quality_multiplier"] = quality
            summary["paper_weight"] = paper_weight

        verdict = LiveFragmentVerdict(
            verdict_id=f"{claim.request_id}:{claim.learner_id}:validator={self.config.validator_hotkey}",
            run_id=claim.run_id,
            request_id=claim.request_id,
            learner_id=claim.learner_id,
            miner_hotkey=claim.miner_hotkey,
            validator_hotkey=self.config.validator_hotkey,
            status=status,
            reason="; ".join(reasons) if reasons else "ok",
            fragment_id=claim.fragment_id,
            fragment_count=claim.fragment_count,
            global_step=claim.global_step,
            claim_uri=claim_uri,
            claim_digest=claim.digest(),
            fragment_state_uri=claim.fragment_state_uri,
            fragment_state_sha256=claim.fragment_state_sha256,
            previous_fragment_state_uri=claim.previous_fragment_state_uri,
            previous_fragment_state_sha256=claim.previous_fragment_state_sha256,
            accepted_weight=accepted_weight,
            trained_tokens=trusted_tokens if status == "pass" else 0,
            local_steps=trusted_steps if status == "pass" else 0,
            quality_multiplier=quality,
            validation_summary=summary,
            checked_unix=time.time(),
        ).sign(self.signer)
        verdict_uri = self.bucket.uri_for_key(
            paths.live_fragment_verdict_key(
                self.config.netuid,
                claim.run_id,
                self.config.validator_hotkey,
                claim.request_id,
                claim.learner_id,
            )
        )
        verdict_payload = json.dumps(verdict.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        if verdict_put_grant is not None:
            (transport or PresignedArtifactTransport(self.bucket)).put(
                verdict_put_grant,
                verdict_payload,
                expected_uri=verdict_uri,
            )
        else:
            self.bucket.put(verdict_uri, verdict_payload)
        return LiveFragmentVerificationResult(claim=claim, verdict=verdict, verdict_uri=verdict_uri)

    @staticmethod
    def _claim_counter_value(claim: LiveFragmentClaim, name: str) -> int:
        values = claim.counters.get(name)
        if isinstance(values, list) and 0 <= int(claim.fragment_id) < len(values):
            try:
                return max(0, int(values[int(claim.fragment_id)]))
            except (TypeError, ValueError):
                return 0
        return 0

    def _live_pull_request_payload(self, claim: LiveFragmentClaim) -> dict[str, object] | None:
        if not claim.learner_id or not claim.request_id:
            return None
        mailbox_uri = self.bucket.uri_for_key(
            paths.learner_mailbox_key(self.config.netuid, claim.run_id, claim.learner_id)
        )
        try:
            mailbox = dict(self.bucket.get_json(mailbox_uri))
        except Exception:
            mailbox = {}
        for item in reversed([dict(message) for message in mailbox.get("messages") or []]):
            if str(item.get("sender") or "") != "syncer" or str(item.get("kind") or "") != "pull_fragment":
                continue
            payload = dict(item.get("payload") or {})
            if str(payload.get("request_id") or "") == claim.request_id:
                return payload

        prefix = self.bucket.uri_for_key(
            paths.mesh_channel_prefix(self.config.netuid, claim.run_id, "syncer", claim.learner_id)
        )
        candidates: list[tuple[int, str]] = []
        for uri in self.bucket.list(prefix):
            if "/seq=" not in uri or not uri.endswith(".json"):
                continue
            raw = uri.rsplit("/seq=", 1)[-1].removesuffix(".json")
            try:
                candidates.append((int(raw), uri))
            except ValueError:
                continue
        for _sequence, uri in sorted(candidates, reverse=True)[:64]:
            try:
                message = DilocoMessage.from_dict(self.bucket.get_json(uri))
            except Exception:
                continue
            if message.kind != "pull_fragment":
                continue
            payload = dict(message.payload or {})
            if str(payload.get("request_id") or "") == claim.request_id:
                return payload
        return None

    def _trusted_live_claim_work(
        self,
        claim: LiveFragmentClaim,
        manifest: TrainingJobManifest | None,
    ) -> tuple[int, int, dict[str, object]]:
        raw_tokens = max(0, int(claim.trained_tokens or self._claim_counter_value(claim, "tokens")))
        raw_steps = max(0, int(claim.local_steps or self._claim_counter_value(claim, "steps")))
        if manifest is None:
            return raw_tokens, raw_steps, {
                "tokens": raw_tokens,
                "steps": raw_steps,
                "capped": False,
                "cap_available": False,
            }
        tokens_per_step = trusted_tokens_per_step(manifest)
        step_cap = self._assigned_live_step_cap(manifest, tokens_per_step=tokens_per_step)
        trusted_steps = min(raw_steps, step_cap) if step_cap > 0 else raw_steps
        allowance = tokens_per_step
        max_tokens = max(0, trusted_steps) * tokens_per_step + allowance
        planned_tokens = self._assigned_live_token_cap(manifest)
        if planned_tokens > 0:
            max_tokens = min(max_tokens, planned_tokens + allowance)
        trusted_tokens = min(raw_tokens, max_tokens)
        return trusted_tokens, trusted_steps, {
            "tokens": trusted_tokens,
            "steps": trusted_steps,
            "raw_tokens": raw_tokens,
            "raw_steps": raw_steps,
            "trusted_tokens_per_step": tokens_per_step,
            "step_cap": step_cap,
            "allowance": allowance,
            "max_tokens": max_tokens,
            "planned_token_cap": planned_tokens,
            "capped": raw_tokens > max_tokens or trusted_steps < raw_steps,
            "steps_capped": trusted_steps < raw_steps,
            "tokens_capped": raw_tokens > max_tokens,
        }

    @staticmethod
    def _assigned_live_token_cap(manifest: TrainingJobManifest) -> int:
        params = dict(manifest.task_params or {})
        env = dict(params.get("env") or {})
        for value in (
            params.get("planned_tokens"),
            params.get("expected_training_units"),
            env.get("QUASAR_ASSIGNED_TOKENS"),
        ):
            try:
                parsed = int(float(value))
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
        return 0

    @classmethod
    def _assigned_live_step_cap(cls, manifest: TrainingJobManifest, *, tokens_per_step: int) -> int:
        params = dict(manifest.task_params or {})
        env = dict(params.get("env") or {})
        caps: list[int] = []
        for value in (env.get("QUASAR_LOCAL_STEPS"), params.get("planned_local_steps")):
            try:
                parsed = int(float(value))
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                caps.append(parsed)
        token_cap = cls._assigned_live_token_cap(manifest)
        if token_cap > 0:
            caps.append(int(ceil(float(token_cap) / float(max(1, tokens_per_step)))))
        if not caps:
            return 0
        return max(1, min(caps) + 1)

    def _live_claim_artifact(
        self,
        *,
        claim: LiveFragmentClaim,
        manifest: TrainingJobManifest | None,
        trusted_tokens: int,
        trusted_steps: int,
    ) -> tuple[FragmentUpdateArtifact, dict[str, object]]:
        with tempfile.TemporaryDirectory(prefix="quasar-live-fragment-") as tmp:
            tmpdir = Path(tmp)
            previous_path = tmpdir / "previous.safetensors"
            learner_path = tmpdir / "learner.safetensors"
            previous_sha, _previous_size = self._materialize_object(
                claim.previous_fragment_state_uri,
                previous_path,
                expected_sha256=claim.previous_fragment_state_sha256,
            )
            learner_sha, _learner_size = self._materialize_object(
                claim.fragment_state_uri,
                learner_path,
                expected_sha256=claim.fragment_state_sha256,
            )
            previous = load_safetensors_file(previous_path)
            learner = load_safetensors_file(learner_path)
        expected_names = sorted(previous)
        expected_shapes = {name: tuple(previous[name].shape) for name in expected_names}
        validate_fragment_state_tensors(
            expected_names=expected_names,
            expected_shapes=expected_shapes,
            tensors=previous,
            artifact_name=f"{FRAGMENT_SYNC_FORMAT}:previous",
        )
        validate_fragment_state_tensors(
            expected_names=expected_names,
            expected_shapes=expected_shapes,
            tensors=learner,
            artifact_name=f"{FRAGMENT_SYNC_FORMAT}:learner",
        )
        delta = {
            name: previous[name].detach().float().cpu() - learner[name].detach().float().cpu()
            for name in expected_names
        }
        fragment_manifest = FragmentUpdateManifest(
            run_id=claim.run_id,
            job_id=claim.job_id,
            round_id=int(claim.round_id),
            global_step=int(claim.global_step),
            fragment_id=int(claim.fragment_id),
            fragment_count=int(claim.fragment_count),
            miner_hotkey=claim.miner_hotkey,
            base_checkpoint_uri=manifest.checkpoint_ref.uri if manifest is not None else claim.previous_fragment_state_uri,
            base_checkpoint_sha256=manifest.checkpoint_ref.sha256 if manifest is not None else claim.previous_fragment_state_sha256,
            trained_tokens=int(trusted_tokens),
            local_steps=int(trusted_steps),
            fragment_trained_tokens=int(trusted_tokens),
            fragment_local_steps=int(trusted_steps),
            tensors=tuple(FragmentTensorSpec.from_tensor(name, tensor) for name, tensor in delta.items()),
            metadata={
                "source": "live_fragment_claim",
                "request_id": claim.request_id,
                "previous_fragment_state_uri": claim.previous_fragment_state_uri,
                "previous_fragment_state_sha256": previous_sha,
                "learner_fragment_state_uri": claim.fragment_state_uri,
                "learner_fragment_state_sha256": learner_sha,
            },
        )
        artifact = FragmentUpdateArtifact(manifest=fragment_manifest, tensors=delta)
        return artifact, {
            "status": "ok",
            "format": artifact.format,
            "algorithm": artifact.algorithm,
            "fragment_id": int(claim.fragment_id),
            "fragment_count": int(claim.fragment_count),
            "tensor_count": artifact.tensor_count(),
            "numel": artifact.numel(),
            "l2_norm": artifact.l2_norm(),
            "component_l2": artifact.component_l2(),
            "previous_fragment_state_sha256": previous_sha,
            "learner_fragment_state_sha256": learner_sha,
        }

    def _independent_eval_config(self, manifest: TrainingJobManifest) -> IndependentQuasarEvalConfig | None:
        data = manifest.validation_policy.get("independent_quasar_eval")
        override = self.config.independent_quasar_eval
        if override is None and not (isinstance(data, dict) and data.get("enabled")):
            return None

        from .quasar_update_eval import IndependentQuasarEvalConfig

        manifest_config = (
            IndependentQuasarEvalConfig.from_policy(data)
            if isinstance(data, dict) and data.get("enabled")
            else None
        )
        if override is None:
            return manifest_config
        if manifest_config is None:
            return override
        if override.random_token_uris:
            return override
        return replace(
            override,
            random_token_uris=manifest_config.random_token_uris,
            require_random_eval=manifest_config.require_random_eval,
            sequence_length=override.sequence_length or manifest_config.sequence_length,
        )

    def _check_fragment_update(self, manifest: TrainingJobManifest, receipt: MinerReceipt) -> tuple[dict[str, object], FragmentUpdateArtifact | None]:
        try:
            update_digest = self._digest_by_name(receipt, {"fragment_update"})
            fragment_manifest_digest = self._digest_by_name(receipt, {"fragment_manifest"})
            with tempfile.TemporaryDirectory(prefix="quasar-validate-fragment-") as tmp:
                tmpdir = Path(tmp)
                update_path = tmpdir / "fragment_update.safetensors"
                manifest_path = tmpdir / "fragment_manifest.json"
                self._materialize_object(update_digest.uri, update_path, expected_sha256=update_digest.sha256)
                self._materialize_object(
                    fragment_manifest_digest.uri,
                    manifest_path,
                    expected_sha256=fragment_manifest_digest.sha256,
                )
                artifact = load_fragment_update_artifact(update_path, manifest_path)
                fragment_manifest = artifact.manifest
                fragment_base_check = self._check_fragment_base(manifest, receipt, artifact)
                if not fragment_base_check["ok"]:
                    return fragment_base_check, None
                expected_format = str(
                    manifest.task_params.get("fragment_artifact")
                    or manifest.task_params.get("artifact_format")
                    or ""
                )
                if expected_format and fragment_manifest.format != expected_format:
                    return {"ok": False, "reason": f"fragment format mismatch: {fragment_manifest.format} != {expected_format}"}, None
                if fragment_manifest.run_id != manifest.run_id or fragment_manifest.job_id != manifest.job_id:
                    return {"ok": False, "reason": "fragment manifest job/run mismatch"}, None
                if fragment_manifest.miner_hotkey and fragment_manifest.miner_hotkey != receipt.worker.hotkey_ss58:
                    return {"ok": False, "reason": "fragment manifest miner hotkey mismatch"}, None
                expected_fragment_id = manifest.task_params.get("fragment_id")
                if expected_fragment_id is not None and int(expected_fragment_id) != int(fragment_manifest.fragment_id):
                    return {"ok": False, "reason": "fragment id mismatch"}, None
                expected_fragment_count = manifest.task_params.get("fragment_count")
                if expected_fragment_count is not None and int(expected_fragment_count) != int(fragment_manifest.fragment_count):
                    return {"ok": False, "reason": "fragment count mismatch"}, None
                checkpoint_sha = manifest.checkpoint_ref.sha256
                if checkpoint_sha and fragment_manifest.base_checkpoint_sha256 and checkpoint_sha != fragment_manifest.base_checkpoint_sha256:
                    return {"ok": False, "reason": "base checkpoint sha mismatch"}, None
                if fragment_manifest.local_steps <= 0 or fragment_manifest.trained_tokens <= 0:
                    return {"ok": False, "reason": "fragment manifest has no positive training work"}, None
                return {
                    "ok": True,
                    "reason": "ok",
                    "format": fragment_manifest.format,
                    "algorithm": fragment_manifest.algorithm,
                    "fragment_id": fragment_manifest.fragment_id,
                    "fragment_count": fragment_manifest.fragment_count,
                    "tensor_count": artifact.tensor_count(),
                    "numel": artifact.numel(),
                    "l2_norm": artifact.l2_norm(),
                    "fragment_nonzero_fraction": artifact.actual_density(),
                    "trained_tokens": fragment_manifest.trained_tokens,
                    "local_steps": fragment_manifest.local_steps,
                    "merge_weight": fragment_manifest.merge_weight(),
                    "component_l2": artifact.component_l2(),
                    "fragment_base": fragment_base_check,
                }, artifact
        except Exception as exc:
            return {"ok": False, "reason": f"fragment update validation failed: {type(exc).__name__}: {exc}"}, None

    def _check_fragment_base(
        self,
        manifest: TrainingJobManifest,
        receipt: MinerReceipt,
        artifact: FragmentUpdateArtifact,
    ) -> dict[str, object]:
        expected = next((ref for ref in manifest.expected_outputs if ref.name == "fragment_base"), None)
        if expected is None:
            return {"required": False, "ok": True}
        digest = next((item for item in receipt.output_digests if item.uri == expected.uri and item.name == "fragment_base"), None)
        if digest is None:
            return {"required": True, "ok": False, "reason": "receipt missing fragment_base digest"}
        with tempfile.TemporaryDirectory(prefix="quasar-validate-base-") as tmp:
            base_path = Path(tmp) / "fragment_base.safetensors"
            actual_sha, _size = self._materialize_object(digest.uri, base_path, expected_sha256=digest.sha256)
            tensors = load_safetensors_file(base_path)
        expected_shapes = {spec.name: tuple(spec.shape) for spec in artifact.manifest.tensors}
        validate_fragment_state_tensors(
            expected_names=expected_shapes,
            expected_shapes=expected_shapes,
            tensors=tensors,
            artifact_name="fragment_base",
        )
        return {
            "required": True,
            "ok": True,
            "reason": "ok",
            "sha256": actual_sha,
            "tensor_count": len(tensors),
        }

    @staticmethod
    def _check_claimed_tokens(manifest: TrainingJobManifest, receipt: MinerReceipt) -> dict[str, object]:
        planned = manifest.task_params.get("planned_tokens")
        if planned is None:
            return {"ok": True, "required": False}
        try:
            planned_tokens = int(planned)
        except (TypeError, ValueError):
            return {"ok": True, "required": False}
        allowance = trusted_tokens_per_step(manifest)
        claimed = int(receipt.claimed_tokens or 0)
        limit = max(0, planned_tokens) + allowance
        return {
            "ok": claimed <= limit,
            "required": True,
            "reason": "ok" if claimed <= limit else f"claimed tokens above planned allowance: {claimed} > {limit}",
            "claimed_tokens": claimed,
            "planned_tokens": planned_tokens,
            "allowance": allowance,
        }

    @staticmethod
    def _digest_by_name(receipt: MinerReceipt, names: set[str]) -> ArtifactDigest:
        for digest in receipt.output_digests:
            if digest.name in names:
                return digest
        raise ValueError(f"receipt {receipt.receipt_id} missing output digest in {sorted(names)}")

    def _check_output_digests(
        self,
        manifest: TrainingJobManifest,
        digests: list[ArtifactDigest],
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        digest_by_uri = {digest.uri: digest for digest in digests}
        expected_uris = {ref.uri for ref in manifest.expected_outputs}
        deferred_fragment_outputs = {"fragment_update", "fragment_manifest", "fragment_base"}
        for ref in manifest.expected_outputs:
            digest = digest_by_uri.get(ref.uri)
            if digest is None:
                results.append({"name": ref.name, "uri": ref.uri, "ok": False, "error": "missing_from_receipt"})
                continue
            if ref.name in deferred_fragment_outputs:
                results.append({"name": digest.name, "uri": digest.uri, "ok": True, "deferred": "fragment_validation"})
                continue
            try:
                data = self.bucket.get(digest.uri)
                actual = ArtifactDigest.from_bytes(name=digest.name, uri=digest.uri, data=data)
                ok = (
                    digest.name == ref.name
                    and actual.sha256 == digest.sha256
                    and actual.size_bytes == digest.size_bytes
                )
                results.append({"name": digest.name, "uri": digest.uri, "ok": ok, "sha256": actual.sha256})
            except FileNotFoundError:
                results.append({"name": digest.name, "uri": digest.uri, "ok": False, "error": "missing"})
        for digest in digests:
            if digest.uri not in expected_uris:
                results.append({"name": digest.name, "uri": digest.uri, "ok": False, "error": "unexpected_output"})
        return results

    def _check_gpu_proof(self, manifest: TrainingJobManifest, receipt: MinerReceipt) -> dict[str, object]:
        required_gpus = int(manifest.resource_requirements.required_gpus)
        if required_gpus <= 1:
            return {"required": False, "ok": True}
        expected = next((ref for ref in manifest.expected_outputs if ref.name == "gpu_proof"), None)
        if expected is None:
            return {"required": True, "ok": False, "reason": "manifest missing gpu_proof output"}
        digest = next((item for item in receipt.output_digests if item.uri == expected.uri and item.name == "gpu_proof"), None)
        if digest is None:
            return {"required": True, "ok": False, "reason": "receipt missing gpu_proof digest"}
        try:
            proof = json.loads(self.bucket.get(digest.uri).decode("utf-8"))
        except Exception as exc:
            return {"required": True, "ok": False, "reason": f"gpu_proof unreadable: {type(exc).__name__}: {exc}"}
        ranks = proof.get("ranks") or []
        if not isinstance(ranks, list):
            return {"required": True, "ok": False, "reason": "gpu_proof ranks is not a list"}
        local_ranks = {
            parsed
            for parsed in (
                self._as_int(rank.get("local_rank"), default=-1)
                for rank in ranks
                if isinstance(rank, dict)
            )
            if parsed >= 0
        }
        cuda_indices = {
            parsed
            for parsed in (
                self._as_int(rank.get("cuda_device_index"), default=-1)
                for rank in ranks
                if isinstance(rank, dict)
            )
            if parsed >= 0
        }
        gpu_uuids = {
            str(rank.get("gpu_uuid"))
            for rank in ranks
            if isinstance(rank, dict) and str(rank.get("gpu_uuid") or "")
        }
        matmul_checksums = [
            float(rank.get("matmul_checksum") or 0.0)
            for rank in ranks
            if isinstance(rank, dict)
        ]
        allreduce_checksums = [
            float(rank.get("allreduce_checksum") or 0.0)
            for rank in ranks
            if isinstance(rank, dict) and rank.get("allreduce_checksum") is not None
        ]
        allreduce_expected = sum(matmul_checksums)
        allreduce_checksum_ok = bool(allreduce_checksums) and all(
            abs(value - allreduce_expected) <= max(1e-6, abs(allreduce_expected) * 1e-6)
            for value in allreduce_checksums
        )
        rank_count = self._as_int(proof.get("rank_count"), default=len(ranks))
        world_size = self._as_int(proof.get("world_size"), default=0)
        visible_gpu_count = self._as_int(proof.get("visible_gpu_count"), default=0)
        requested_gpus = self._as_int(proof.get("requested_gpus"), default=0)
        allreduce_ok = all(bool(rank.get("allreduce_ok")) for rank in ranks if isinstance(rank, dict))
        capabilities = dict(receipt.worker.capabilities or {})
        capability_gpu_count = self._capability_gpu_count(capabilities)
        proof_visible_ids = self._split_csv(str(proof.get("cuda_visible_devices") or ""))
        capability_gpu_ids = self._capability_gpu_ids(capabilities)
        visible_matches_capabilities = True
        if capability_gpu_ids:
            visible_matches_capabilities = (
                len(proof_visible_ids) >= required_gpus
                and set(proof_visible_ids).issubset(set(capability_gpu_ids))
            )
        checks = {
            "manifest_hash_ok": proof.get("manifest_hash") == (manifest.manifest_hash or manifest.compute_manifest_hash()),
            "job_id_ok": proof.get("job_id") == manifest.job_id,
            "run_id_ok": proof.get("run_id") == manifest.run_id,
            "requested_gpus_ok": requested_gpus >= required_gpus,
            "visible_gpu_count_ok": visible_gpu_count >= required_gpus,
            "world_size_ok": world_size >= required_gpus,
            "rank_count_ok": rank_count >= required_gpus and len(local_ranks) >= required_gpus,
            "unique_cuda_device_indices_ok": not cuda_indices or len(cuda_indices) >= required_gpus,
            "unique_gpu_uuids_ok": not gpu_uuids or len(gpu_uuids) >= required_gpus,
            "distributed_ok": bool(proof.get("distributed")),
            "allreduce_ok": allreduce_ok,
            "allreduce_checksum_ok": allreduce_checksum_ok,
            "worker_capability_gpu_count_ok": capability_gpu_count >= required_gpus,
            "worker_group_id_ok": bool(capabilities.get("worker_group_id")),
            "worker_cuda_ok": capabilities.get("cuda_ok") is not False,
            "visible_matches_capabilities_ok": visible_matches_capabilities,
        }
        failed = [name for name, ok in checks.items() if not ok]
        return {
            "required": True,
            "ok": not failed,
            "reason": "ok" if not failed else f"gpu_proof failed checks: {', '.join(failed)}",
            "required_gpus": required_gpus,
            "requested_gpus": requested_gpus,
            "visible_gpu_count": visible_gpu_count,
            "world_size": world_size,
            "rank_count": rank_count,
            "worker_gpu_count": capability_gpu_count,
            "checks": checks,
        }

    @staticmethod
    def _as_int(value: object, *, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _split_csv(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    @staticmethod
    def _capability_gpu_ids(capabilities: dict[str, object]) -> list[str]:
        visible = str(capabilities.get("cuda_visible_devices") or "").strip()
        if visible:
            return ValidatorVerifier._split_csv(visible)
        for key in ("gpu_ids", "gpu_indices", "device_group"):
            raw = capabilities.get(key)
            if isinstance(raw, str):
                return ValidatorVerifier._split_csv(raw)
            if isinstance(raw, list):
                return [str(item) for item in raw]
        return []

    @staticmethod
    def _capability_gpu_count(capabilities: dict[str, object]) -> int:
        for key in ("gpu_count", "world_size", "n_gpus"):
            raw = capabilities.get(key)
            if raw is None or raw == "":
                continue
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                continue
            if parsed >= 0:
                return parsed
        return len(ValidatorVerifier._capability_gpu_ids(capabilities))

    @staticmethod
    def _expected_training_units(manifest: TrainingJobManifest) -> float:
        return expected_training_units_from_manifest(manifest)
