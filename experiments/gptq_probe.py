from __future__ import annotations

import argparse
import copy
import fnmatch
import io
import math
import sys
import zlib
from pathlib import Path

import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_gpt as tg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Same-checkpoint GPTQ-style exporter probe.")
    p.add_argument("--checkpoint", default="final_model.pt")
    p.add_argument("--targets", nargs="+", required=True, help="State-dict glob patterns ending in .weight")
    p.add_argument("--calib-batches", type=int, default=2)
    p.add_argument("--calib-batch-tokens", type=int, default=16384)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--val-max-tokens", type=int, default=65536)
    p.add_argument("--val-batch-size", type=int, default=16384)
    p.add_argument("--eval-stride", type=int, default=64)
    p.add_argument("--damp", type=float, default=1e-2)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_model(args: tg.Hyperparameters, device: torch.device) -> tg.GPT:
    model = tg.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_shared_blocks=args.num_shared_blocks,
        num_shared_mlps=args.num_shared_mlps,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        mlp_kind=args.mlp_kind,
        mlp_hidden=args.mlp_hidden,
        adapter_rank=args.adapter_rank,
        tie_embeddings=args.tie_embeddings,
        tied_emb_fp32_master=args.tied_emb_fp32_master,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device)
    model.bfloat16()
    tg.restore_low_dim_params_to_fp32(model)
    if args.tied_emb_fp32_master and args.tie_embeddings:
        tg.restore_tied_embedding_to_fp32(model)
    return model


def resolve_targets(state_dict: dict[str, torch.Tensor], patterns: list[str]) -> list[str]:
    names = []
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            continue
        if any(fnmatch.fnmatch(name, pat) for pat in patterns):
            names.append(name)
    if not names:
        raise ValueError(f"No tensors matched target patterns: {patterns}")
    return sorted(names)


def collect_hessians(
    model: tg.GPT,
    target_names: list[str],
    args: tg.Hyperparameters,
    device: torch.device,
    calib_batches: int,
    calib_batch_tokens: int,
    seq_len: int,
) -> dict[str, torch.Tensor]:
    module_map = dict(model.named_modules())
    target_modules = {name: module_map[name.removesuffix(".weight")] for name in target_names}
    hessians = {name: None for name in target_names}
    counts = {name: 0 for name in target_names}

    def make_hook(name: str):
        def hook(_module, inputs):
            x = inputs[0].detach().to(dtype=torch.float32)
            flat = x.reshape(-1, x.shape[-1])
            h = flat.T @ flat
            h_cpu = h.cpu().clone()
            hessians[name] = h_cpu if hessians[name] is None else hessians[name].add_(h_cpu)
            counts[name] += flat.shape[0]
        return hook

    hooks = [target_modules[name].register_forward_pre_hook(make_hook(name)) for name in target_names]
    loader = tg.DistributedTokenLoader(args.train_files, rank=0, world_size=1, device=device)
    model.eval()
    with torch.inference_mode():
        for _ in range(calib_batches):
            x, _ = loader.next_batch(calib_batch_tokens, seq_len, grad_accum_steps=1)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                model(x)
    for hook in hooks:
        hook.remove()

    for name in target_names:
        if hessians[name] is None or counts[name] == 0:
            raise RuntimeError(f"Failed to collect activations for {name}")
        hessians[name] = (hessians[name].clone() / counts[name]).contiguous()
    return hessians


def gptq_quantize_matrix(weight: torch.Tensor, hessian: torch.Tensor, damp: float) -> tuple[torch.Tensor, torch.Tensor]:
    w = weight.detach().to(dtype=torch.float32, device="cpu").contiguous()
    rows, cols = w.shape
    qmax = 127.0
    q_codes = torch.empty((rows, cols), dtype=torch.int8)
    scales = torch.empty((rows,), dtype=torch.float32)

    h = hessian.to(dtype=torch.float32, device="cpu").contiguous()
    eye = torch.eye(h.shape[0], dtype=torch.float32)
    avg_diag = float(torch.diag(h).mean().item()) if h.numel() else 1.0
    damping = max(damp * avg_diag, 1e-6)
    while True:
        try:
            chol = torch.linalg.cholesky(h + eye * damping)
            h_inv = torch.cholesky_inverse(chol)
            break
        except RuntimeError:
            damping *= 10.0
            if damping > 1e6:
                raise

    diag = torch.diag(h_inv).clamp_min(1e-8)
    for r in range(rows):
        row = w[r].clone()
        clip_abs = float(torch.quantile(row.abs(), tg.INT8_CLIP_Q).item()) if row.numel() else 0.0
        scale = clip_abs / qmax if clip_abs > 0 else 1.0
        scales[r] = scale
        if clip_abs <= 0:
            q_codes[r].zero_()
            continue
        for c in range(cols):
            q = float(torch.clamp(torch.round(torch.clamp(row[c], -clip_abs, clip_abs) / scale), -qmax, qmax).item())
            err = (row[c] - q * scale) / diag[c]
            row[c:] -= err * h_inv[c, c:]
            q_codes[r, c] = int(q)
    return q_codes, scales


def compressed_size_bytes(obj: dict[str, object]) -> int:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return len(zlib.compress(buf.getvalue(), level=9))


def evaluate_quant_obj(
    quant_obj: dict[str, object],
    args: tg.Hyperparameters,
    device: torch.device,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
) -> tuple[float, float]:
    model = build_model(args, device)
    model.load_state_dict(tg.dequantize_state_dict_int8(quant_obj), strict=True)
    return tg.eval_val(
        args,
        model,
        rank=0,
        world_size=1,
        device=device,
        grad_accum_steps=1,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )


def main() -> None:
    cli = parse_args()
    device = torch.device(cli.device)
    state_dict = torch.load(cli.checkpoint, map_location="cpu")
    target_names = resolve_targets(state_dict, cli.targets)

    args = tg.Hyperparameters()
    args.train_seq_len = cli.seq_len
    args.eval_stride_tokens = cli.eval_stride
    args.val_max_tokens = cli.val_max_tokens
    args.val_batch_size = cli.val_batch_size
    args.model_dim = int(state_dict["tok_emb.weight"].shape[1])
    args.vocab_size = int(state_dict["tok_emb.weight"].shape[0])
    args.num_layers = int(state_dict["attn_scales"].shape[0])
    args.num_shared_blocks = len({name.split(".")[1] for name in state_dict if name.startswith("attn_blocks.") and name.endswith(".weight")})
    args.num_shared_mlps = len({name.split(".")[1] for name in state_dict if name.startswith("mlp_blocks.") and name.endswith(".weight")})
    args.mlp_hidden = int(state_dict[[n for n in state_dict if n.endswith("mlp.fc.weight")][0]].shape[0])

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = tg.load_validation_tokens(args.val_files, args.train_seq_len, args.val_max_tokens)
    if device.type == "cuda":
        sp_luts = tg.build_sentencepiece_luts(sp, args.vocab_size, device)
    else:
        sp_luts = tg.build_sentencepiece_luts(sp, args.vocab_size, torch.device("cpu"))

    calib_model = build_model(args, device)
    calib_model.load_state_dict(state_dict, strict=True)
    hessians = collect_hessians(
        calib_model,
        target_names,
        args,
        device=device,
        calib_batches=cli.calib_batches,
        calib_batch_tokens=cli.calib_batch_tokens,
        seq_len=cli.seq_len,
    )

    baseline_obj, _ = tg.quantize_state_dict_int8(state_dict)
    baseline_loss, baseline_bpb = evaluate_quant_obj(
        baseline_obj, args, device, val_tokens, *sp_luts
    )
    baseline_bytes = compressed_size_bytes(baseline_obj)
    print(f"baseline bytes={baseline_bytes} val_loss={baseline_loss:.8f} val_bpb={baseline_bpb:.8f}")

    gptq_obj = copy.deepcopy(baseline_obj)
    for name in target_names:
        q_codes, scales = gptq_quantize_matrix(state_dict[name], hessians[name], cli.damp)
        gptq_obj["quantized"][name] = q_codes.contiguous()
        gptq_obj["scales"][name] = scales.to(dtype=tg.INT8_PER_ROW_SCALE_DTYPE).contiguous()

    gptq_loss, gptq_bpb = evaluate_quant_obj(
        gptq_obj, args, device, val_tokens, *sp_luts
    )
    gptq_bytes = compressed_size_bytes(gptq_obj)
    print("targets:")
    for name in target_names:
        print(f"  - {name}")
    print(
        f"gptq bytes={gptq_bytes} val_loss={gptq_loss:.8f} val_bpb={gptq_bpb:.8f} "
        f"delta_bpb={gptq_bpb - baseline_bpb:+.8f} delta_bytes={gptq_bytes - baseline_bytes:+d}"
    )


if __name__ == "__main__":
    main()
