from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_gpt as tg
from experiments.gptq_probe import build_model, compressed_size_bytes, evaluate_quant_obj, normalize_checkpoint_state_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Same-checkpoint exact exporter probe.")
    p.add_argument("--checkpoint", default="final_model.pt")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--val-max-tokens", type=int, default=65536)
    p.add_argument("--val-batch-size", type=int, default=16384)
    p.add_argument("--eval-stride", type=int, default=64)
    p.add_argument("--calib-batch-tokens", type=int, default=16384)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def infer_shared_count(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
    ids = {
        name.split(".")[1]
        for name in state_dict
        if name.startswith(prefix) and name.endswith(".weight")
    }
    return len(ids)


def main() -> None:
    cli = parse_args()
    device = torch.device(cli.device)
    state_dict = normalize_checkpoint_state_dict(torch.load(cli.checkpoint, map_location="cpu"))

    args = tg.Hyperparameters()
    args.train_seq_len = cli.seq_len
    args.eval_stride_tokens = cli.eval_stride
    args.val_max_tokens = cli.val_max_tokens
    args.val_batch_size = cli.val_batch_size
    args.train_batch_tokens = cli.calib_batch_tokens
    args.model_dim = int(state_dict["tok_emb.weight"].shape[1])
    args.vocab_size = int(state_dict["tok_emb.weight"].shape[0])
    args.num_layers = int(state_dict["attn_scales"].shape[0])
    args.num_shared_blocks = infer_shared_count(state_dict, "attn_blocks.")
    args.num_shared_mlps = infer_shared_count(state_dict, "mlp_blocks.")
    if args.num_shared_blocks == 0:
        args.num_shared_blocks = infer_shared_count(state_dict, "blocks.")
    if args.num_shared_mlps == 0:
        args.num_shared_mlps = infer_shared_count(state_dict, "blocks.")
    args.mlp_hidden = int(state_dict[[n for n in state_dict if n.endswith("mlp.fc.weight")][0]].shape[0])

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    bos_id = int(sp.bos_id())
    val_tokens = tg.load_validation_tokens(
        args.val_files,
        args.train_seq_len,
        args.val_max_tokens,
        preserve_docs=args.eval_doc_isolated,
        bos_id=bos_id,
    )
    sp_luts = tg.build_sentencepiece_luts(sp, args.vocab_size, device if device.type == "cuda" else torch.device("cpu"))

    calib_model = build_model(args, device)
    calib_model.load_state_dict(state_dict, strict=True)
    activation_stats = tg.collect_activation_reparam_stats(
        args,
        calib_model,
        rank=0,
        world_size=1,
        device=device,
        grad_accum_steps=1,
    )
    gptq_hessians = tg.collect_gptq_hessians(
        args,
        calib_model,
        rank=0,
        world_size=1,
        device=device,
        grad_accum_steps=1,
    )

    quant_obj, _ = tg.quantize_state_dict_int8(
        state_dict,
        activation_stats=activation_stats,
        gptq_hessians=gptq_hessians,
    )
    q_loss, q_bpb = evaluate_quant_obj(
        quant_obj,
        args,
        device,
        val_tokens,
        *sp_luts,
        bos_id,
    )
    q_bytes = compressed_size_bytes(quant_obj)

    print(f"checkpoint={cli.checkpoint}")
    print(
        "exporter_flags "
        f"center_rows={int(tg.INT8_CENTER_ROWS)} "
        f"group_size={tg.INT8_GROUP_SIZE} "
        f"keep_tok_emb_fp16={int(tg.INT8_KEEP_TOK_EMB_FP16)} "
        f"rotation_kind={tg.INT8_ROTATION_KIND} "
        f"scale_reparam_kind={tg.INT8_SCALE_REPARAM_KIND} "
        f"activation_reparam_kind={tg.INT8_ACTIVATION_REPARAM_KIND} "
        f"activation_reparam_calib_batches={tg.INT8_ACTIVATION_REPARAM_CALIB_BATCHES} "
        f"gptq_target={tg.INT8_GPTQ_TARGET} "
        f"gptq_calib_batches={tg.INT8_GPTQ_CALIB_BATCHES}"
    )
    if activation_stats:
        print(f"activation_stats_tensors={len(activation_stats)}")
    if gptq_hessians:
        print(f"gptq_hessian_tensors={len(gptq_hessians)}")
    print(
        f"exact bytes={q_bytes} val_loss={q_loss:.8f} val_bpb={q_bpb:.8f} "
        f"seq_len={args.train_seq_len} eval_stride={args.eval_stride_tokens} "
        f"val_max_tokens={args.val_max_tokens} eval_doc_isolated={int(args.eval_doc_isolated)} "
        f"calib_batch_tokens={args.train_batch_tokens}"
    )


if __name__ == "__main__":
    main()
