#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import torch
torch.set_float32_matmul_precision('high')

from common import (
    FaissLayerIndex,
    ProbeRecord,
    PromptExample,
    build_faiss_index,
    decode_token,
    faiss_metric_name,
    is_special_token_id,
    jsonl_write,
    knn_top1_and_true_rank,
    load_bank_tensor,
    load_bank_vectors_for_faiss,
    load_model_and_tokenizer,
    maybe_normalize,
    prepare_prompt_examples,
    print_summary,
    resolve_bank_path,
    saved_faiss_metric,
    validate_layers,
)

import numpy as np
import random
import os

def set_seed(seed: int = 42):
    # Python built-in random module
    random.seed(seed)
    
    # Numpy library
    np.random.seed(seed)
    
    # PyTorch seed for CPU and all GPU devices
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN determinism (crucial for GPU reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"Random seed set as {seed}")

set_seed(42)


class ActivationBank:
    def __init__(self, bank_dir: str, layers: Sequence[int]):
        self.bank_dir = Path(bank_dir)
        self.layers = list(layers)
        self.bank_tensors: Dict[int, torch.Tensor] = {}
        self.faiss_indices: Dict[int, FaissLayerIndex] = {}

    def bank_path(self, layer: int) -> Path:
        return resolve_bank_path(self.bank_dir, layer)

    def faiss_path(self, layer: int) -> Path:
        return self.bank_dir / f"layer_{layer}.faiss"

    def exists(self) -> bool:
        return all(self.bank_path(l).exists() or self.faiss_path(l).exists() for l in self.layers)

    def load(self, device: str, use_faiss: bool, use_cosine: bool) -> None:
        self.bank_tensors = {}
        self.faiss_indices = {}
        if use_faiss:
            import faiss
            requested_metric = faiss_metric_name(use_cosine)
            recorded_metric = saved_faiss_metric(self.bank_dir)
            for layer in self.layers:
                fpath = self.faiss_path(layer)
                if fpath.exists() and recorded_metric == requested_metric:
                    self.faiss_indices[layer] = FaissLayerIndex(faiss.read_index(str(fpath)), use_cosine=use_cosine)
                else:
                    if fpath.exists():
                        if recorded_metric is None:
                            print(
                                f"Saved FAISS index for layer {layer} has no recorded metric; "
                                "rebuilding from raw bank tensors."
                            )
                        else:
                            print(
                                f"Saved FAISS index for layer {layer} uses metric {recorded_metric!r}, "
                                f"not requested {requested_metric!r}; rebuilding from raw bank tensors."
                            )
                    try:
                        vectors = load_bank_vectors_for_faiss(self.bank_dir, layer)
                    except FileNotFoundError as exc:
                        raise FileNotFoundError(
                            f"Could not rebuild FAISS index for layer {layer}: raw bank tensors are missing "
                            f"under {self.bank_dir}, and the saved index is not reusable for metric "
                            f"{requested_metric!r}."
                        ) from exc
                    self.faiss_indices[layer] = FaissLayerIndex(build_faiss_index(vectors, use_cosine=use_cosine), use_cosine=use_cosine)
        else:
            self.bank_tensors = {
                layer: load_bank_tensor(self.bank_dir, layer, device, normalize=use_cosine)
                for layer in self.layers
            }

    def search(self, layer: int, query: torch.Tensor, true_token_id: int, topk_true_rank: int, use_cosine: bool):
        if layer in self.faiss_indices:
            return self.faiss_indices[layer].search(query=query, true_token_id=true_token_id, topk_true_rank=topk_true_rank)
        return knn_top1_and_true_rank(query, self.bank_tensors[layer], true_token_id, use_cosine, topk_true_rank)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe prefill/decode states against activation bank.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--prompts_file", type=str, required=True)
    p.add_argument("--bank_dir", type=str, required=True)
    p.add_argument("--out_jsonl", type=str, required=True)
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="HF hidden_states indices to use. If omitted, use all layers including embeddings."
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_prompts", type=int, default=None)
    p.add_argument("--max_prompt_tokens", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--normalize_queries", action="store_true")
    p.add_argument(
        "--normalize_bank",
        action="store_true",
        help="Normalize bank vectors at probe time for cosine retrieval. Raw bank tensors stay unnormalized on disk.",
    )
    p.add_argument("--collect_attention_stats", action="store_true")
    p.add_argument("--topk_true_rank", type=int, default=10)
    p.add_argument(
        "--use_faiss",
        action="store_true",
        help="Use saved FAISS indices when their recorded metric matches, otherwise rebuild an in-memory index from raw bank tensors.",
    )
    p.add_argument("--prompt_format", type=str, default="raw", choices=["raw", "chat"])
    p.add_argument("--add_generation_prompt", action="store_true", help="For chat prompts, append assistant-generation marker in the template.")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=[None, "eager", "sdpa", "flash_attention_2", "flash_attention_3"],
    )
    return p.parse_args()


def summarize_prefill_attention(attn_tensor: torch.Tensor, pos: int, bos_positions: Sequence[int]) -> Dict[str, float]:
    probs = attn_tensor[:, pos, : pos + 1].to(torch.float32)
    k_len = probs.size(-1)
    positions = torch.arange(k_len, device=probs.device, dtype=torch.float32)
    eps = 1e-12
    entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)
    top1 = probs.topk(k=min(1, k_len), dim=-1).values.sum(dim=-1)
    top5 = probs.topk(k=min(5, k_len), dim=-1).values.sum(dim=-1)
    recent8 = probs[:, max(0, k_len - 8):].sum(dim=-1)
    recent32 = probs[:, max(0, k_len - 32):].sum(dim=-1)
    bos_idx = [i for i in bos_positions if i <= pos]
    bos = probs[:, bos_idx].sum(dim=-1) if bos_idx else torch.zeros(probs.size(0), device=probs.device)
    prev = probs[:, pos - 1] if pos - 1 >= 0 else torch.zeros(probs.size(0), device=probs.device)
    mean_dist = (probs * (pos - positions).unsqueeze(0)).sum(dim=-1)
    return {
        "attn_entropy": float(entropy.mean().item()),
        "attn_top1_mass": float(top1.mean().item()),
        "attn_top5_mass": float(top5.mean().item()),
        "attn_recent_8_mass": float(recent8.mean().item()),
        "attn_recent_32_mass": float(recent32.mean().item()),
        "attn_bos_mass": float(bos.mean().item()),
        "attn_prev_mass": float(prev.mean().item()),
        "attn_mean_distance": float(mean_dist.mean().item()),
    }


def summarize_decode_attention(attn_tensor: torch.Tensor, bos_positions: Sequence[int]) -> Dict[str, float]:
    probs = attn_tensor[:, -1, :].to(torch.float32)
    k_len = probs.size(-1)
    positions = torch.arange(k_len, device=probs.device, dtype=torch.float32)
    eps = 1e-12
    entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)
    top1 = probs.topk(k=min(1, k_len), dim=-1).values.sum(dim=-1)
    top5 = probs.topk(k=min(5, k_len), dim=-1).values.sum(dim=-1)
    recent8 = probs[:, max(0, k_len - 8):].sum(dim=-1)
    recent32 = probs[:, max(0, k_len - 32):].sum(dim=-1)
    bos = probs[:, bos_positions].sum(dim=-1) if bos_positions else torch.zeros(probs.size(0), device=probs.device)
    prev = probs[:, -2] if k_len >= 2 else torch.zeros(probs.size(0), device=probs.device)
    mean_dist = (probs * ((k_len - 1) - positions).unsqueeze(0)).sum(dim=-1)
    return {
        "attn_entropy": float(entropy.mean().item()),
        "attn_top1_mass": float(top1.mean().item()),
        "attn_top5_mass": float(top5.mean().item()),
        "attn_recent_8_mass": float(recent8.mean().item()),
        "attn_recent_32_mass": float(recent32.mean().item()),
        "attn_bos_mass": float(bos.mean().item()),
        "attn_prev_mass": float(prev.mean().item()),
        "attn_mean_distance": float(mean_dist.mean().item()),
    }


def _prefill_template_flag(example: PromptExample, pos: int) -> bool:
    mask = example.template_control_token_mask or []
    return bool(mask[pos]) if pos < len(mask) else False


@torch.no_grad()
def probe_prefill(
    model,
    tokenizer,
    prompt_index: int,
    example: PromptExample,
    layers: Sequence[int],
    bank: ActivationBank,
    normalize_queries: bool,
    normalize_bank: bool,
    collect_attention_stats: bool,
    topk_true_rank: int,
    device: str,
) -> List[ProbeRecord]:
    input_ids = torch.tensor([example.input_ids], dtype=torch.long, device=device)
    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        output_attentions=collect_attention_stats,
        use_cache=False,
        return_dict=True,
    )
    token_ids = input_ids[0].tolist()
    bos_positions = [i for i, tid in enumerate(token_ids) if tid == tokenizer.bos_token_id]
    use_cosine = normalize_queries and normalize_bank
    records: List[ProbeRecord] = []

    for pos, true_id in enumerate(token_ids):
        is_template_control = _prefill_template_flag(example, pos)
        for layer in layers:
            query = outputs.hidden_states[layer][0, pos, :].to(torch.float32)
            query = maybe_normalize(query, normalize_queries)
            nn_id, score, rank_true = bank.search(layer, query, true_id, topk_true_rank, use_cosine)
            attn_stats = {}
            if collect_attention_stats and layer >= 1:
                attn_stats = summarize_prefill_attention(outputs.attentions[layer - 1][0], pos, bos_positions)
            records.append(ProbeRecord(
                prompt_index=prompt_index,
                phase="prefill",
                position_in_phase=pos,
                global_position=pos,
                token_id=int(true_id),
                token_text=decode_token(tokenizer, true_id),
                is_special_token=is_special_token_id(tokenizer, true_id),
                is_template_control_token=is_template_control,
                layer=int(layer),
                nn_token_id=nn_id,
                nn_token_text=decode_token(tokenizer, nn_id),
                disentangled=(nn_id == int(true_id)),
                score=score,
                rank_of_true=rank_true,
                **attn_stats,
            ))
    return records


@torch.no_grad()
def sample_next_token(logits: torch.Tensor, temperature: float) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def probe_decode(
    model,
    tokenizer,
    prompt_index: int,
    example: PromptExample,
    layers: Sequence[int],
    bank: ActivationBank,
    max_new_tokens: int,
    temperature: float,
    normalize_queries: bool,
    normalize_bank: bool,
    collect_attention_stats: bool,
    topk_true_rank: int,
    device: str,
) -> List[ProbeRecord]:
    input_ids = torch.tensor([example.input_ids], dtype=torch.long, device=device)
    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        output_attentions=collect_attention_stats,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    full_ids = input_ids[0].tolist()
    use_cosine = normalize_queries and normalize_bank
    records: List[ProbeRecord] = []

    for step in range(max_new_tokens):
        next_id = sample_next_token(outputs.logits[0, -1, :], temperature)
        next_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
        step_outputs = model(
            input_ids=next_input,
            past_key_values=past_key_values,
            output_hidden_states=True,
            output_attentions=collect_attention_stats,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = step_outputs.past_key_values
        outputs = step_outputs
        full_ids.append(next_id)
        bos_positions = [i for i, tid in enumerate(full_ids) if tid == tokenizer.bos_token_id]

        for layer in layers:
            query = step_outputs.hidden_states[layer][0, 0, :].to(torch.float32)
            query = maybe_normalize(query, normalize_queries)
            nn_id, score, rank_true = bank.search(layer, query, next_id, topk_true_rank, use_cosine)
            attn_stats = {}
            if collect_attention_stats and layer >= 1:
                attn = step_outputs.attentions[layer - 1][0]
                attn_stats = summarize_decode_attention(attn, bos_positions)
            records.append(ProbeRecord(
                prompt_index=prompt_index,
                phase="decode",
                position_in_phase=step,
                global_position=len(full_ids) - 1,
                token_id=int(next_id),
                token_text=decode_token(tokenizer, next_id),
                is_special_token=is_special_token_id(tokenizer, next_id),
                is_template_control_token=False,
                layer=int(layer),
                nn_token_id=nn_id,
                nn_token_text=decode_token(tokenizer, nn_id),
                disentangled=(nn_id == int(next_id)),
                score=score,
                rank_of_true=rank_true,
                **attn_stats,
            ))

        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break

    return records


def main() -> None:
    args = parse_args()
    if args.collect_attention_stats:
        print("Attention stats enabled: loading model with attn_implementation='eager'.")
    model, tokenizer = load_model_and_tokenizer(args)
    if args.layers is None:
        args.layers = list(range(num_hidden_state_slots(model)))
    print(f"Using layers: {args.layers}")
    validate_layers(args.layers, model)
    bank = ActivationBank(args.bank_dir, args.layers)
    if not bank.exists():
        raise FileNotFoundError(f"Activation bank not found in {args.bank_dir}")
    bank.load(args.device, use_faiss=args.use_faiss, use_cosine=(args.normalize_queries and args.normalize_bank))

    examples = prepare_prompt_examples(
        prompts_file=args.prompts_file,
        tokenizer=tokenizer,
        prompt_format=args.prompt_format,
        max_prompts=args.max_prompts,
        max_prompt_tokens=args.max_prompt_tokens,
        add_generation_prompt=args.add_generation_prompt,
    )
    all_records: List[ProbeRecord] = []
    for i, example in enumerate(examples):
        print(f"Probing prompt {i + 1}/{len(examples)}")
        all_records.extend(probe_prefill(
            model, tokenizer, i, example, args.layers, bank,
            args.normalize_queries, args.normalize_bank,
            args.collect_attention_stats, args.topk_true_rank, args.device,
        ))
        all_records.extend(probe_decode(
            model, tokenizer, i, example, args.layers, bank,
            args.max_new_tokens, args.temperature,
            args.normalize_queries, args.normalize_bank,
            args.collect_attention_stats, args.topk_true_rank, args.device,
        ))

    jsonl_write(args.out_jsonl, all_records)
    print_summary(all_records)
    print(f"Wrote {len(all_records)} rows to {args.out_jsonl}")


if __name__ == "__main__":
    main()
