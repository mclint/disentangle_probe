#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence

import torch
torch.set_float32_matmul_precision('high')
from tensordict import MemoryMappedTensor

from common import (
    bank_memmap_path,
    bank_metadata_path,
    batch_iter,
    build_faiss_index,
    ensure_dir,
    get_dtype,
    hidden_size_from_model,
    load_model_and_tokenizer,
    maybe_normalize,
    save_bank_metadata,
    validate_layers,
    num_hidden_state_slots,
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
    def __init__(self, bank_dir: str, layers: Sequence[int], dtype: torch.dtype):
        self.bank_dir = Path(bank_dir)
        self.layers = list(layers)
        self.dtype = dtype
        self.bank_arrays: Dict[int, torch.Tensor] = {}

    def bank_path(self, layer: int) -> Path:
        return bank_memmap_path(self.bank_dir, layer)

    def faiss_path(self, layer: int) -> Path:
        return self.bank_dir / f"layer_{layer}.faiss"

    def initialize(self, vocab_size: int, d_model: int) -> None:
        ensure_dir(self.bank_dir)
        save_bank_metadata(self.bank_dir, vocab_size, d_model, self.dtype, self.layers)
        self.bank_arrays = {
            layer: MemoryMappedTensor.empty(
                (vocab_size, d_model),
                dtype=self.dtype,
                filename=self.bank_path(layer),
                existsok=True,
            )
            for layer in self.layers
        }

    def save(self, save_tensors: bool = True, save_faiss: bool = True, use_cosine: bool = False) -> None:
        ensure_dir(self.bank_dir)
        if save_faiss:
            import faiss
            for layer, array in self.bank_arrays.items():
                index = build_faiss_index(array, use_cosine=use_cosine)
                faiss.write_index(index, str(self.faiss_path(layer)))
        if not save_tensors:
            for layer in self.layers:
                path = self.bank_path(layer)
                if path.exists():
                    path.unlink()
            meta_path = bank_metadata_path(self.bank_dir)
            if meta_path.exists():
                meta_path.unlink()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-layer vocab activation bank.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="HF hidden_states indices to use. If omitted, use all layers including embeddings."
    )
    p.add_argument("--bank_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--vocab_batch_size", type=int, default=256)
    p.add_argument("--normalize_bank", action="store_true")
    p.add_argument("--skip_save_tensors", action="store_true", help="Save only FAISS indices and not the raw bank tensors.")
    p.add_argument("--skip_save_faiss", action="store_true", help="Save only raw bank tensors and not FAISS indices.")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("-attn_implementation", type=str, default="eager")
    p.add_argument("--anchor_token_id", type=int, default=None,
                    help="Token ID to prepend as anchor. Defaults to tokenizer.bos_token_id.")
    return p.parse_args()


def resolve_anchor_id(args: argparse.Namespace, tokenizer) -> tuple[int, str]:
    if args.anchor_token_id is not None:
        return int(args.anchor_token_id), "argument --anchor_token_id"

    for attr in ("bos_token_id", "cls_token_id", "eos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            return int(token_id), f"tokenizer.{attr}"

    raise ValueError(
        "Could not determine a default anchor token id from the tokenizer. "
        "Please pass --anchor_token_id explicitly."
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args)

    anchor_id, anchor_source = resolve_anchor_id(args, tokenizer)
    anchor_text = tokenizer.convert_ids_to_tokens(anchor_id)
    print(f"Anchor token selected from {anchor_source}: id={anchor_id} text={anchor_text!r}")

    if args.layers is None:
        args.layers = list(range(num_hidden_state_slots(model)))
    print(f"Using layers: {args.layers}")
    validate_layers(args.layers, model)

    vocab_size = int(tokenizer.vocab_size)
    d_model = hidden_size_from_model(model)
    bank_dtype = get_dtype(args.dtype)
    print(f"Raw bank storage dtype: {args.dtype}")
    bank = ActivationBank(args.bank_dir, args.layers, dtype=bank_dtype)
    bank.initialize(vocab_size, d_model)

    token_ids = list(range(vocab_size))
    for ids in batch_iter(token_ids, args.vocab_batch_size):
        input_ids = torch.tensor(ids, dtype=torch.long, device=args.device).unsqueeze(1)
        input_ids = torch.cat([torch.ones_like(input_ids) * anchor_id, input_ids], dim=-1)
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        for layer in args.layers:
            vecs = outputs.hidden_states[layer][:, 1, :]
            # if layer > 0:
            #     vecs -= outputs.hidden_states[layer-1][:, 0, :]
            vecs = maybe_normalize(vecs, args.normalize_bank)
            bank.bank_arrays[layer][ids] = vecs.to(bank_dtype).cpu()

    bank.save(
        save_tensors=not args.skip_save_tensors,
        save_faiss=not args.skip_save_faiss,
        use_cosine=args.normalize_bank,
    )
    print(f"Saved activation bank to {args.bank_dir}")
    if not args.skip_save_faiss:
        print("FAISS indices written per layer.")


if __name__ == "__main__":
    main()
