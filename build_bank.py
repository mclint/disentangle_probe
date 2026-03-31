#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence

import torch
torch.set_float32_matmul_precision('high')

from common import (
    batch_iter,
    build_faiss_index,
    ensure_dir,
    get_dtype,
    hidden_size_from_model,
    load_model_and_tokenizer,
    maybe_normalize,
    to_numpy_f32,
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
    def __init__(self, bank_dir: str, layers: Sequence[int]):
        self.bank_dir = Path(bank_dir)
        self.layers = list(layers)
        self.bank_tensors: Dict[int, torch.Tensor] = {}

    def bank_path(self, layer: int) -> Path:
        return self.bank_dir / f"layer_{layer}.pt"

    def faiss_path(self, layer: int) -> Path:
        return self.bank_dir / f"layer_{layer}.faiss"

    def save(self, save_tensors: bool = True, save_faiss: bool = True, use_cosine: bool = False) -> None:
        ensure_dir(self.bank_dir)
        if save_tensors:
            for layer, tensor in self.bank_tensors.items():
                torch.save(tensor.to(torch.float32).cpu(), self.bank_path(layer))
        if save_faiss:
            import faiss
            for layer, tensor in self.bank_tensors.items():
                index = build_faiss_index(to_numpy_f32(tensor), use_cosine=use_cosine)
                faiss.write_index(index, str(self.faiss_path(layer)))


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


@torch.no_grad()
def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args)

    anchor_id = args.anchor_token_id if args.anchor_token_id is not None else tokenizer.bos_token_id
    anchor_text = tokenizer.convert_ids_to_tokens(anchor_id)
    print(f"Anchor token: id={anchor_id} text={anchor_text!r}")

    if args.layers is None:
        args.layers = list(range(num_hidden_state_slots(model)))
    print(f"Using layers: {args.layers}")
    validate_layers(args.layers, model)

    vocab_size = int(tokenizer.vocab_size)
    d_model = hidden_size_from_model(model)
    bank = ActivationBank(args.bank_dir, args.layers)
    bank.bank_tensors = {
        l: torch.empty((vocab_size, d_model), dtype=get_dtype(args.dtype), device=args.device)
        for l in args.layers
    }

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
            bank.bank_tensors[layer][ids] = vecs

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
