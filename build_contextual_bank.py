#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

torch.set_float32_matmul_precision("high")

from common import (
    PromptExample,
    SparseTokenCodebookAccumulator,
    SparseTokenStatsAccumulator,
    build_distribution_token_mask,
    dtype_name,
    ensure_dir,
    get_dtype,
    hidden_size_from_model,
    load_distribution_specs,
    load_model_and_tokenizer,
    maybe_normalize,
    num_hidden_state_slots,
    prepare_prompt_examples,
    prepare_prompt_examples_from_chat_objects,
    validate_layers,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


set_seed(42)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build contextualized token bank and codebook from chat datasets.")
    p.add_argument("--model", type=str, required=True)
    source_group = p.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--dataset_file", type=str, default=None)
    source_group.add_argument("--hf_dataset", type=str, default=None)
    p.add_argument("--distribution_config", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="HF hidden_states indices to use. If omitted, use all layers including embeddings.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--max_prompt_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--hf_split", type=str, default="train")
    p.add_argument("--hf_revision", type=str, default=None)
    p.add_argument("--hf_streaming", action="store_true")
    p.add_argument("--normalize_states", action="store_true")
    p.add_argument("--codebook_k", type=int, default=8)
    p.add_argument("--codebook_min_count", type=int, default=8)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
    )
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.codebook_k <= 0:
        raise ValueError("--codebook_k must be positive.")
    if args.codebook_min_count <= 0:
        raise ValueError("--codebook_min_count must be positive.")
    has_dataset_file = bool(getattr(args, "dataset_file", None))
    has_hf_dataset = bool(getattr(args, "hf_dataset", None))
    if has_dataset_file == has_hf_dataset:
        raise ValueError("Provide exactly one dataset source: --dataset_file or --hf_dataset.")


def load_hf_chat_objects(
    dataset_name: str,
    split: str,
    revision: str | None,
    streaming: bool,
    max_examples: int | None,
) -> List[Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Direct Hugging Face dataset loading requires the 'datasets' package.") from exc

    dataset = load_dataset(dataset_name, split=split, revision=revision, streaming=streaming)
    objects: List[Any] = []
    iterator = dataset if streaming else dataset
    for idx, row in enumerate(iterator):
        objects.append(row)
        if max_examples is not None and idx + 1 >= max_examples:
            break
    return objects


def load_input_examples(args: argparse.Namespace, tokenizer) -> tuple[List[PromptExample], Dict[str, Any]]:
    if getattr(args, "dataset_file", None):
        examples = prepare_prompt_examples(
            prompts_file=args.dataset_file,
            tokenizer=tokenizer,
            prompt_format="chat",
            max_prompts=args.max_examples,
            max_prompt_tokens=args.max_prompt_tokens,
            add_generation_prompt=False,
        )
        return examples, {
            "dataset_source": "local_jsonl",
            "dataset_name": None,
            "dataset_split": None,
            "dataset_revision": None,
        }

    objects = load_hf_chat_objects(
        dataset_name=args.hf_dataset,
        split=args.hf_split,
        revision=args.hf_revision,
        streaming=bool(getattr(args, "hf_streaming", False)),
        max_examples=args.max_examples,
    )
    examples = prepare_prompt_examples_from_chat_objects(
        objects=objects,
        tokenizer=tokenizer,
        max_prompts=None,
        max_prompt_tokens=args.max_prompt_tokens,
        add_generation_prompt=False,
        source_metadata={
            "dataset_source": "huggingface",
            "dataset_name": args.hf_dataset,
            "dataset_split": args.hf_split,
            "dataset_revision": args.hf_revision,
        },
    )
    return examples, {
        "dataset_source": "huggingface",
        "dataset_name": args.hf_dataset,
        "dataset_split": args.hf_split,
        "dataset_revision": args.hf_revision,
    }


def collate_prompt_examples(
    examples: Sequence[PromptExample],
    pad_token_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, List[int]]:
    if not examples:
        raise ValueError("Expected at least one example to collate.")

    max_len = max(len(example.input_ids or []) for example in examples)
    input_ids = torch.full((len(examples), max_len), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long, device=device)
    lengths: List[int] = []

    for row, example in enumerate(examples):
        ids = list(example.input_ids or [])
        mask = list(example.attention_mask or [1] * len(ids))
        seq_len = min(len(ids), len(mask))
        if seq_len == 0:
            lengths.append(0)
            continue
        row_ids = torch.tensor(ids[:seq_len], dtype=torch.long, device=device)
        row_mask = torch.tensor(mask[:seq_len], dtype=torch.long, device=device)
        input_ids[row, :seq_len] = row_ids
        attention_mask[row, :seq_len] = row_mask
        lengths.append(int(row_mask.sum().item()))

    return input_ids, attention_mask, lengths


@torch.no_grad()
def run_contextual_bank(args: argparse.Namespace) -> None:
    _validate_args(args)
    model, tokenizer = load_model_and_tokenizer(args)

    if args.layers is None:
        args.layers = list(range(num_hidden_state_slots(model)))
    print(f"Using layers: {args.layers}")
    validate_layers(args.layers, model)

    distributions = load_distribution_specs(args.distribution_config)
    print(f"Loaded distributions: {[spec.name for spec in distributions]}")

    examples, dataset_meta = load_input_examples(args, tokenizer)
    if not examples:
        if dataset_meta["dataset_source"] == "huggingface":
            raise ValueError(
                f"No examples found in Hugging Face dataset "
                f"{dataset_meta['dataset_name']} split {dataset_meta['dataset_split']}"
            )
        raise ValueError(f"No examples found in dataset file {args.dataset_file}")
    if dataset_meta["dataset_source"] == "huggingface":
        print(f"Loaded {len(examples)} chat examples from {args.hf_dataset} [{args.hf_split}]")
    else:
        print(f"Loaded {len(examples)} chat examples from {args.dataset_file}")

    d_model = hidden_size_from_model(model)
    storage_dtype = get_dtype(args.dtype)

    distribution_masks = [
        {spec.name: build_distribution_token_mask(example, tokenizer, spec) for spec in distributions}
        for example in examples
    ]

    stats_accumulators: Dict[str, Dict[int, SparseTokenStatsAccumulator]] = {
        spec.name: {layer: SparseTokenStatsAccumulator(d_model) for layer in args.layers}
        for spec in distributions
    }
    codebook_accumulators: Dict[str, Dict[int, SparseTokenCodebookAccumulator]] = {
        spec.name: {layer: SparseTokenCodebookAccumulator(d_model, args.codebook_k) for layer in args.layers}
        for spec in distributions
    }

    for start in range(0, len(examples), args.batch_size):
        batch_examples = examples[start : start + args.batch_size]
        batch_masks = distribution_masks[start : start + args.batch_size]
        input_ids, attention_mask, lengths = collate_prompt_examples(batch_examples, tokenizer.pad_token_id, args.device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        normalized_hidden_states = {
            layer: maybe_normalize(outputs.hidden_states[layer].detach().to(torch.float32), args.normalize_states)
            for layer in args.layers
        }

        for row, (example, mask_by_distribution, seq_len) in enumerate(zip(batch_examples, batch_masks, lengths)):
            if seq_len <= 0:
                continue
            token_ids = list(example.input_ids or [])[:seq_len]
            for pos, token_id in enumerate(token_ids):
                matched_distributions = [
                    spec.name
                    for spec in distributions
                    if pos < len(mask_by_distribution[spec.name]) and mask_by_distribution[spec.name][pos]
                ]
                if not matched_distributions:
                    continue
                for layer in args.layers:
                    vector = normalized_hidden_states[layer][row, pos, :]
                    for distribution_name in matched_distributions:
                        stats_accumulators[distribution_name][layer].update(token_id, vector)
                        codebook_accumulators[distribution_name][layer].update(token_id, vector)

        end = min(start + args.batch_size, len(examples))
        print(f"Processed examples {start + 1}-{end} / {len(examples)}")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    meta = {
        "model": args.model,
        "hidden_size": int(d_model),
        "layers": [int(layer) for layer in args.layers],
        "dtype": dtype_name(storage_dtype),
        "dataset_format": "chat",
        "dataset_source": dataset_meta["dataset_source"],
        "dataset_name": dataset_meta["dataset_name"],
        "dataset_split": dataset_meta["dataset_split"],
        "dataset_revision": dataset_meta["dataset_revision"],
        "normalize_states": bool(args.normalize_states),
        "codebook_k": int(args.codebook_k),
        "codebook_min_count": int(args.codebook_min_count),
        "max_examples": args.max_examples,
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "batch_size": int(args.batch_size),
        "hf_streaming": bool(getattr(args, "hf_streaming", False)),
        "num_examples_processed": len(examples),
        "distributions": [asdict(spec) for spec in distributions],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    for spec in distributions:
        distribution_dir = out_dir / f"distribution_{spec.name}"
        ensure_dir(distribution_dir)
        for layer in args.layers:
            stats_payload = stats_accumulators[spec.name][layer].finalize(storage_dtype)
            codebook_payload = codebook_accumulators[spec.name][layer].finalize(
                storage_dtype=storage_dtype,
                stats_accumulator=stats_accumulators[spec.name][layer],
                min_count=args.codebook_min_count,
            )
            torch.save(stats_payload, distribution_dir / f"layer_{layer}_stats.pt")
            torch.save(codebook_payload, distribution_dir / f"layer_{layer}_codebook.pt")

    print(f"Saved contextualized bank artifacts to {args.out_dir}")


def main() -> None:
    args = parse_args()
    run_contextual_bank(args)


if __name__ == "__main__":
    main()
