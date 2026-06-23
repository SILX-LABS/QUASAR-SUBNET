"""Prepare a local Quasar Preview directory with the real Raven source."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Prepare Quasar Preview with repo-local Raven source")
    parser.add_argument("--model-id", default="silx-ai/Quasar-Preview")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", default=str(root / ".quasar-model" / "Quasar-Preview"))
    parser.add_argument("--raven-repo", default="https://github.com/goombalab/raven.git")
    parser.add_argument("--raven-revision", default="main")
    parser.add_argument("--train-deps-dir", default=str(root / ".train-deps"), help="Dependency target dir where a raven symlink should be created")
    parser.add_argument("--include-weights", action="store_true", help="Download model weights too. Default is code/tokenizer/config only.")
    return parser


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def _patch_raven_training_path(raven_file: Path) -> dict[str, Any]:
    text = raven_file.read_text()
    patched = False

    import_line = "from fla.ops.gsa.naive import naive_recurrent_gsa as naive_recurrent_raven"
    if import_line not in text:
        text = text.replace(
            "from fla.ops.gsa import chunk_gsa as chunk_raven, fused_recurrent_gsa as fused_recurrent_raven\n",
            "from fla.ops.gsa import chunk_gsa as chunk_raven, fused_recurrent_gsa as fused_recurrent_raven\n"
            f"{import_line}\n",
        )
        patched = True

    os_import_line = "import os"
    if os_import_line not in text:
        text = text.replace("from __future__ import annotations\n\n", "from __future__ import annotations\n\nimport os\n")
        patched = True

    training_mode_line = (
        "training_mode = os.environ.get('QUASAR_RAVEN_TRAINING_MODE', 'fused_recurrent')\n"
        "        mode = training_mode if self.training else ('fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode)"
    )
    if training_mode_line not in text:
        text = text.replace(
            "        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode\n",
            "        # Default to the autograd-safe PyTorch path for training, but allow\n"
            "        # explicit kernel probes from the incentive runner.\n"
            f"        {training_mode_line}\n",
        )
        text = text.replace(
            "        mode = 'fused_recurrent' if (hidden_states.shape[1] <= 64 and not self.training) else self.mode\n",
            "        # Default to the autograd-safe PyTorch path for training, but allow\n"
            "        # explicit kernel probes from the incentive runner.\n"
            f"        {training_mode_line}\n",
        )
        patched = True

    recurrent_block = (
        "        recurrent_state = last_state['recurrent_state'] if last_state is not None else None\n"
        "        if self.num_kv_groups > 1:\n"
        "            k, v, f, s = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_kv_groups), (k, v, f, s))\n"
    )
    patched_recurrent_block = (
        "        recurrent_state = last_state['recurrent_state'] if last_state is not None else None\n"
        "        k_naive, v_naive, f_naive, s_naive = k, v, f, s\n"
        "        if self.num_kv_groups > 1:\n"
        "            # K/V are grouped and must be expanded to Q heads for the\n"
        "            # recurrent kernels. Gate/slot tensors are already per-Q-head.\n"
        "            k, v = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_kv_groups), (k, v))\n"
    )
    if "f_naive, s_naive" not in text and recurrent_block in text:
        text = text.replace(recurrent_block, patched_recurrent_block)
        patched = True

    old_grouped_repeat = (
        "        k_naive, v_naive, f_naive, s_naive = k, v, f, s\n"
        "        if self.num_kv_groups > 1:\n"
        "            k, v, f, s = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_kv_groups), (k, v, f, s))\n"
    )
    if old_grouped_repeat in text:
        text = text.replace(old_grouped_repeat, patched_recurrent_block)
        patched = True

    naive_branch = (
        "        elif mode == 'naive_recurrent':\n"
        "            if cu_seqlens is not None:\n"
        "                raise NotImplementedError('naive_recurrent Raven training path does not support packed cu_seqlens inputs')\n"
        "            # The PyTorch reference expects K/V already aligned to Q heads\n"
        "            # here, while Raven's gate/slot tensors are already per-Q-head.\n"
        "            o, recurrent_state = naive_recurrent_raven(\n"
        "                q=q,\n"
        "                k=k,\n"
        "                v=v,\n"
        "                s=s_naive,\n"
        "                g=f_naive,\n"
        "                initial_state=recurrent_state,\n"
        "                output_final_state=use_cache,\n"
        "                scale=self.scale,\n"
        "            )\n"
    )
    if "elif mode == 'naive_recurrent':" not in text:
        text = text.replace(
            "        else:\n"
            "            raise NotImplementedError(f\"Not supported mode `{mode}`.\")\n",
            naive_branch
            + "        else:\n"
            + "            raise NotImplementedError(f\"Not supported mode `{mode}`.\")\n",
        )
        patched = True

    if patched:
        raven_file.write_text(text)

    return {
        "raven_patch_applied": patched,
        "has_naive_import": import_line in text,
        "has_training_naive_mode": training_mode_line in text,
        "has_naive_branch": "elif mode == 'naive_recurrent':" in text,
    }


def _patch_quasar_modeling(modeling_file: Path) -> dict[str, Any]:
    text = modeling_file.read_text()
    patched = False

    replacements = [
        (
            "        if w12_key not in state_dict:\n"
            "            state_dict[w12_key] = self.experts_w12.data\n"
            "            state_dict[w3_key] = self.experts_w3.data\n",
            "        if w12_key not in state_dict:\n"
            "            state_dict[w12_key] = self.experts_w12.data\n"
            "        if w3_key not in state_dict:\n"
            "            state_dict[w3_key] = self.experts_w3.data\n",
        ),
        (
            '        if os.environ.get("LOCAL_RANK", "0") == "0" and getattr(self, "_model_forward_debug", 0) < 1:\n',
            '        if os.environ.get("QUASAR_MODEL_FORWARD_DEBUG", "0").lower() in {"1", "true", "yes"} and os.environ.get("LOCAL_RANK", "0") == "0" and getattr(self, "_model_forward_debug", 0) < 1:\n',
        ),
        (
            '            if os.environ.get("LOCAL_RANK", "0") == "0" and getattr(self, "_loss_debug_count", 0) < 5:\n',
            '            if os.environ.get("QUASAR_MODEL_LOSS_DEBUG", "0").lower() in {"1", "true", "yes"} and os.environ.get("LOCAL_RANK", "0") == "0" and getattr(self, "_loss_debug_count", 0) < 5:\n',
        ),
        (
            '            if os.environ.get("LOCAL_RANK", "0") == "0" and getattr(self, "_loss_debug_count", 0) <= 5:\n',
            '            if os.environ.get("QUASAR_MODEL_LOSS_DEBUG", "0").lower() in {"1", "true", "yes"} and os.environ.get("LOCAL_RANK", "0") == "0" and getattr(self, "_loss_debug_count", 0) <= 5:\n',
        ),
    ]
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            patched = True

    raven_top_old = (
        '    _RAVEN_PATH = _os.path.join(_HERE, "raven")\n'
        '    if _RAVEN_PATH not in _sys.path:\n'
    )
    raven_top_new = (
        '    _RAVEN_PATH = _os.environ.get("QUASAR_RAVEN_DIR") or _os.path.join(_HERE, "raven")\n'
        '    if _RAVEN_PATH not in _sys.path:\n'
    )
    if raven_top_old in text and "QUASAR_RAVEN_DIR" not in text:
        text = text.replace(raven_top_old, raven_top_new, 1)
        patched = True

    raven_guard_old = (
        '        if not os.path.isdir(os.path.join(_HERE, "raven")):\n'
        '            raise ModuleNotFoundError("Quasar requires the bundled repo-local raven/ folder for Raven hybrid layers")\n'
        '        from raven.layers.raven import RavenAttention\n'
    )
    raven_guard_new = (
        '        _raven_dir = os.environ.get("QUASAR_RAVEN_DIR") or os.path.join(_HERE, "raven")\n'
        '        if _raven_dir not in _sys.path:\n'
        '            _sys.path.insert(0, _raven_dir)\n'
        '        if not os.path.isdir(_raven_dir):\n'
        '            raise ModuleNotFoundError("Quasar requires the bundled repo-local raven/ folder for Raven hybrid layers")\n'
        '        from raven.layers.raven import RavenAttention\n'
    )
    if raven_guard_old in text:
        text = text.replace(raven_guard_old, raven_guard_new, 1)
        patched = True

    static_forward_old = (
        "        elif not self.training and infer_all_experts and (not decode_only_all_experts or seq_len == 1):\n"
        "            y = self.moe_all_experts(hidden_states, topk_idx, topk_weight)\n"
        "        else:\n"
        "            y = self.moe_vectorized(hidden_states, topk_idx, topk_weight)\n"
    )
    static_forward_new = (
        "        elif not self.training and infer_all_experts and (not decode_only_all_experts or seq_len == 1):\n"
        "            y = self.moe_all_experts(hidden_states, topk_idx, topk_weight)\n"
        "        elif self.training and os.environ.get(\"QUASAR_MOE_STATIC_ALL_EXPERTS\", \"0\") == \"1\":\n"
        "            y = self.moe_static_all_experts(hidden_states, topk_idx, topk_weight)\n"
        "        else:\n"
        "            y = self.moe_vectorized(hidden_states, topk_idx, topk_weight)\n"
    )
    if "def moe_static_all_experts(" not in text and static_forward_old in text:
        text = text.replace(static_forward_old, static_forward_new)
        patched = True

    static_method = '''    def moe_static_all_experts(self, x, topk_ids, topk_weight):
        bsz, seq_len, h_dim = x.shape
        k = topk_ids.shape[-1]
        num_tokens = bsz * seq_len
        num_assignments = num_tokens * k
        num_experts = self.config.num_experts
        flat_x = x.view(num_tokens, h_dim)
        flat_topk_idx = topk_ids.reshape(-1)

        tokens_per_expert = torch.bincount(flat_topk_idx, minlength=num_experts)
        avg_tokens = num_assignments // num_experts
        capacity = max(128, int(2.0 * avg_tokens))

        sorted_indices = torch.argsort(flat_topk_idx)
        expert_idx = flat_topk_idx[sorted_indices]
        token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(k)[sorted_indices]
        expert_starts = torch.cat(
            [tokens_per_expert.new_zeros(1), tokens_per_expert[:-1].cumsum(0)],
            dim=0,
        )
        intra_offsets = torch.arange(num_assignments, device=x.device) - expert_starts.repeat_interleave(tokens_per_expert)

        mask = intra_offsets < capacity
        sorted_indices = sorted_indices[mask]
        expert_idx = expert_idx[mask]
        token_indices = token_indices[mask]
        intra_offsets = intra_offsets[mask]

        grouped_x = flat_x[token_indices]
        flat_dest_indices = expert_idx * capacity + intra_offsets

        padded_x = x.new_zeros(num_experts, capacity, h_dim)
        padded_x.view(-1, h_dim).index_copy_(0, flat_dest_indices, grouped_x)

        h12 = torch.bmm(padded_x, self.experts_w12)
        h1, h2 = h12.chunk(2, dim=-1)
        h = F.silu(h1) * h2
        expert_out_padded = torch.bmm(h, self.experts_w3)

        routed_out = torch.zeros_like(flat_x)
        gating = topk_weight.reshape(-1)[sorted_indices].unsqueeze(1).to(dtype=expert_out_padded.dtype)
        routed_out.index_add_(
            0,
            token_indices,
            expert_out_padded.view(-1, h_dim)[flat_dest_indices] * gating,
        )
        return routed_out.view(bsz, seq_len, h_dim)

'''
    if "def moe_static_all_experts(" not in text:
        marker = "    def moe_vectorized(self, x, topk_ids, topk_weight):\n"
        if marker in text:
            text = text.replace(marker, static_method + marker)
            patched = True

    if patched:
        modeling_file.write_text(text)

    return {
        "quasar_modeling_patch_applied": patched,
        "loss_debug_env_gated": "QUASAR_MODEL_LOSS_DEBUG" in text,
        "forward_debug_env_gated": "QUASAR_MODEL_FORWARD_DEBUG" in text,
        "moe_w3_state_dict_patch": "if w3_key not in state_dict:" in text,
        "moe_static_all_experts_patch": "def moe_static_all_experts(" in text,
        "raven_env_dir_patch": "QUASAR_RAVEN_DIR" in text,
    }


def _patch_tokenizer_config(model_dir: Path) -> dict[str, Any]:
    tokenizer_config = model_dir / "tokenizer_config.json"
    if not tokenizer_config.exists():
        return {"tokenizer_config_patch_applied": False, "tokenizer_class_removed": None}

    data = json.loads(tokenizer_config.read_text())
    tokenizer_class = data.get("tokenizer_class")
    if tokenizer_class != "TokenizersBackend":
        return {"tokenizer_config_patch_applied": False, "tokenizer_class_removed": tokenizer_class}

    data.pop("tokenizer_class", None)
    tokenizer_config.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return {"tokenizer_config_patch_applied": True, "tokenizer_class_removed": tokenizer_class}


def _link_raven_package(raven_dir: Path, train_deps_dir: str | None) -> str | None:
    if not train_deps_dir:
        return None
    package_dir = (raven_dir / "raven").expanduser().resolve()
    deps_dir = Path(train_deps_dir).expanduser().resolve()
    deps_dir.mkdir(parents=True, exist_ok=True)
    link = deps_dir / "raven"
    if link.is_symlink() or not link.exists():
        if link.is_symlink():
            link.unlink()
        link.symlink_to(package_dir, target_is_directory=True)
    elif link.resolve() != package_dir:
        raise RuntimeError(f"{link} exists and does not point to {package_dir}")
    return f"{link} -> {package_dir}"


def prepare(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    snapshot_kwargs: dict[str, Any] = {
        "repo_id": args.model_id,
        "revision": args.revision or None,
        "local_dir": str(output_dir),
    }
    if not args.include_weights:
        snapshot_kwargs["allow_patterns"] = [
            "*.py",
            "*.json",
            "*.txt",
            "*.model",
            "*.tiktoken",
            "tokenizer*",
            "generation_config.json",
            "fla/**",
        ]
        snapshot_kwargs["ignore_patterns"] = [
            "*.safetensors",
            "*.bin",
            "*.pt",
            "*.pth",
            "*.ckpt",
            "*.gguf",
        ]
    snapshot_path = snapshot_download(**snapshot_kwargs)
    model_dir = Path(snapshot_path).resolve()

    raven_dir = model_dir / "raven"
    if (raven_dir / ".git").exists():
        _run(["git", "fetch", "--all", "--prune"], cwd=raven_dir)
        _run(["git", "checkout", args.raven_revision], cwd=raven_dir)
        _run(["git", "pull", "--ff-only"], cwd=raven_dir)
    elif raven_dir.exists():
        raise RuntimeError(f"{raven_dir} already exists but is not a git checkout")
    else:
        _run(["git", "clone", "--depth", "1", "--branch", args.raven_revision, args.raven_repo, str(raven_dir)])

    raven_file = raven_dir / "raven" / "layers" / "raven.py"
    patch_result = _patch_raven_training_path(raven_file)
    modeling_patch_result = _patch_quasar_modeling(model_dir / "modeling_quasar_long.py")
    tokenizer_patch_result = _patch_tokenizer_config(model_dir)
    raven_link = _link_raven_package(raven_dir, args.train_deps_dir)

    return {
        "ok": True,
        "model_dir": str(model_dir),
        "raven_dir": str(raven_dir),
        "raven_package_link": raven_link,
        "runtime_pythonpath": f"{args.train_deps_dir}:{model_dir}",
        "weights_included": bool(args.include_weights),
        "has_modeling": (model_dir / "modeling_quasar_long.py").exists(),
        "has_fla": (model_dir / "fla").is_dir(),
        "has_raven": raven_dir.is_dir(),
        "has_raven_attention": raven_file.exists(),
        **patch_result,
        **modeling_patch_result,
        **tokenizer_patch_result,
    }


def main(argv: list[str] | None = None) -> int:
    result = prepare(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
