#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

torch.set_float32_matmul_precision("high")

from build_contextual_bank import collate_prompt_examples, load_input_examples
from common import (
    ReconstructionMetricsAccumulator,
    TokenDistributionSpec,
    build_distribution_token_mask,
    ensure_dir,
    hidden_size_from_model,
    load_bank_tensor,
    load_distribution_specs,
    load_model_and_tokenizer,
    lookup_prototypes,
    maybe_normalize,
    num_hidden_state_slots,
    split_examples_deterministically,
    to_numpy_f32,
    validate_layers,
)


BASELINE_PROTOTYPE = "prototype_only"
BASELINE_CODEBOOK = "prototype_plus_codebook"
BASELINE_PCA = "prototype_plus_pca"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prototype and residual reconstruction baselines.")
    parser.add_argument("--model", type=str, required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--dataset_file", type=str, default=None)
    source_group.add_argument("--hf_dataset", type=str, default=None)
    parser.add_argument("--bank_dir", type=str, required=True)
    parser.add_argument("--distribution_config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="HF hidden_states indices to use. If omitted, use all layers including embeddings.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_prompt_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--hf_revision", type=str, default=None)
    parser.add_argument("--hf_streaming", action="store_true")
    parser.add_argument(
        "--normalize_bank",
        action="store_true",
        help="Normalize bank vectors at evaluation time so prototypes match normalized hidden states.",
    )
    parser.add_argument("--normalize_states", action="store_true")
    parser.add_argument("--fit_fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codebook_k", type=int, default=8)
    parser.add_argument("--pca_rank", type=int, default=8)
    parser.add_argument("--coverage_cosine_thresholds", type=float, nargs="+", default=[0.90, 0.95, 0.98])
    parser.add_argument("--coverage_mse_thresholds", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.codebook_k <= 0:
        raise ValueError("--codebook_k must be positive.")
    if args.pca_rank <= 0:
        raise ValueError("--pca_rank must be positive.")
    if not 0.0 < float(args.fit_fraction) < 1.0:
        raise ValueError("--fit_fraction must be strictly between 0 and 1.")
    if not args.coverage_cosine_thresholds:
        raise ValueError("Provide at least one --coverage_cosine_thresholds value.")
    if not args.coverage_mse_thresholds:
        raise ValueError("Provide at least one --coverage_mse_thresholds value.")
    if bool(args.normalize_bank) != bool(args.normalize_states):
        raise ValueError(
            "Residual evaluation requires bank and hidden states to live in the same space. "
            "Set both --normalize_bank and --normalize_states, or neither."
        )


def _mask_tensor(mask_values: Sequence[bool], seq_len: int, device: str) -> torch.Tensor:
    if seq_len <= 0:
        return torch.zeros(0, dtype=torch.bool, device=device)
    return torch.tensor(list(mask_values[:seq_len]), dtype=torch.bool, device=device)


def _build_distribution_masks(
    examples: Sequence[Any],
    tokenizer,
    distributions: Sequence[TokenDistributionSpec],
) -> List[Dict[str, List[bool]]]:
    return [
        {spec.name: build_distribution_token_mask(example, tokenizer, spec) for spec in distributions}
        for example in examples
    ]


def _empty_layer_tensor(hidden_size: int) -> torch.Tensor:
    return torch.empty((0, hidden_size), dtype=torch.float32)


def _fit_torch_kmeans(
    residuals: torch.Tensor,
    k: int,
    *,
    seed: int,
    max_iters: int = 25,
) -> torch.Tensor:
    n_samples = int(residuals.shape[0])
    k = min(int(k), n_samples)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    centroids = residuals[torch.randperm(n_samples, generator=generator)[:k]].clone()

    for _ in range(max_iters):
        squared_distances = torch.sum(
            (residuals.unsqueeze(1) - centroids.unsqueeze(0)) ** 2,
            dim=-1,
        )
        assignments = torch.argmin(squared_distances, dim=1)
        next_centroids = centroids.clone()
        for cluster_idx in range(k):
            cluster_mask = assignments == cluster_idx
            if torch.any(cluster_mask):
                next_centroids[cluster_idx] = residuals[cluster_mask].mean(dim=0)
        if torch.allclose(next_centroids, centroids):
            centroids = next_centroids
            break
        centroids = next_centroids
    return centroids


def fit_residual_codebook(
    residuals: torch.Tensor,
    *,
    k: int,
    seed: int,
) -> Dict[str, Any]:
    residuals = residuals.detach().to(torch.float32).cpu()
    if residuals.ndim != 2:
        raise ValueError("Residual codebook fitting expects a [num_tokens, hidden_size] tensor.")
    if residuals.shape[0] == 0:
        return {"available": False, "backend": None, "centroids": _empty_layer_tensor(int(residuals.shape[-1]))}

    k_eff = min(int(k), int(residuals.shape[0]))
    if int(residuals.shape[0]) < max(32, 4 * k_eff):
        centroids = _fit_torch_kmeans(residuals, k_eff, seed=seed)
        return {
            "available": True,
            "backend": "torch",
            "centroids": centroids,
            "num_centroids": int(centroids.shape[0]),
        }

    try:
        import faiss

        kmeans = faiss.Kmeans(
            int(residuals.shape[1]),
            int(k_eff),
            niter=25,
            verbose=False,
            seed=int(seed),
            gpu=False,
        )
        kmeans.train(to_numpy_f32(residuals))
        centroids = torch.from_numpy(kmeans.centroids.reshape(k_eff, residuals.shape[1])).to(torch.float32)
        backend = "faiss"
    except Exception:
        centroids = _fit_torch_kmeans(residuals, k_eff, seed=seed)
        backend = "torch"

    return {
        "available": True,
        "backend": backend,
        "centroids": centroids,
        "num_centroids": int(centroids.shape[0]),
    }


def reconstruct_with_codebook(
    residuals: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    if residuals.numel() == 0:
        return residuals.clone()
    if centroids.numel() == 0:
        return torch.zeros_like(residuals)
    squared_distances = torch.sum(
        (residuals.unsqueeze(1) - centroids.unsqueeze(0)) ** 2,
        dim=-1,
    )
    nearest = torch.argmin(squared_distances, dim=1)
    return centroids[nearest]


def fit_low_rank_residual_model(
    residuals: torch.Tensor,
    *,
    rank: int,
) -> Dict[str, Any]:
    residuals = residuals.detach().to(torch.float32).cpu()
    if residuals.ndim != 2:
        raise ValueError("PCA fitting expects a [num_tokens, hidden_size] tensor.")
    hidden_size = int(residuals.shape[1])
    mean = residuals.mean(dim=0) if residuals.shape[0] > 0 else torch.zeros(hidden_size, dtype=torch.float32)
    if residuals.shape[0] == 0:
        return {"available": False, "mean": mean, "basis": _empty_layer_tensor(hidden_size).transpose(0, 1), "rank": 0}

    centered = residuals - mean
    max_rank = min(int(rank), int(centered.shape[0]), hidden_size)
    if max_rank <= 0 or torch.allclose(centered, torch.zeros_like(centered)):
        basis = torch.empty((hidden_size, 0), dtype=torch.float32)
        return {"available": True, "mean": mean, "basis": basis, "rank": 0}

    try:
        _, _, basis = torch.pca_lowrank(centered, q=max_rank, center=False)
        basis = basis[:, :max_rank].to(torch.float32)
    except RuntimeError:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[:max_rank, :].transpose(0, 1).contiguous().to(torch.float32)

    return {
        "available": True,
        "mean": mean,
        "basis": basis,
        "rank": int(basis.shape[1]),
    }


def reconstruct_with_pca(
    residuals: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    if residuals.numel() == 0:
        return residuals.clone()
    centered = residuals - mean.unsqueeze(0)
    if basis.numel() == 0:
        return mean.unsqueeze(0).expand_as(residuals)
    coefficients = centered @ basis
    return mean.unsqueeze(0) + (coefficients @ basis.transpose(0, 1))


@torch.no_grad()
def collect_fit_residuals(
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    examples: Sequence[Any],
    masks: Sequence[Dict[str, List[bool]]],
    distributions: Sequence[TokenDistributionSpec],
    bank_tensors: Dict[int, torch.Tensor],
    hidden_size: int,
) -> tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, Dict[int, int]]]:
    residual_chunks: Dict[str, Dict[int, List[torch.Tensor]]] = {
        spec.name: {layer: [] for layer in args.layers}
        for spec in distributions
    }
    token_counts: Dict[str, Dict[int, int]] = {
        spec.name: {layer: 0 for layer in args.layers}
        for spec in distributions
    }

    for start in range(0, len(examples), args.batch_size):
        batch_examples = examples[start : start + args.batch_size]
        batch_masks = masks[start : start + args.batch_size]
        input_ids, attention_mask, lengths = collate_prompt_examples(batch_examples, tokenizer.pad_token_id, args.device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_by_layer = {
            layer: maybe_normalize(outputs.hidden_states[layer].detach().to(torch.float32), args.normalize_states)
            for layer in args.layers
        }

        for row, (example, mask_by_distribution, seq_len) in enumerate(zip(batch_examples, batch_masks, lengths)):
            if seq_len <= 0:
                continue
            token_ids = torch.tensor(list(example.input_ids or [])[:seq_len], dtype=torch.long, device=args.device)
            residual_by_layer: Dict[int, torch.Tensor] = {}
            for layer in args.layers:
                hidden_states = hidden_by_layer[layer][row, :seq_len, :]
                prototypes = lookup_prototypes(bank_tensors[layer], token_ids, device=hidden_states.device).to(torch.float32)
                residual_by_layer[layer] = hidden_states - prototypes

            for spec in distributions:
                mask = _mask_tensor(mask_by_distribution[spec.name], seq_len, args.device)
                if not torch.any(mask):
                    continue
                selected_count = int(mask.sum().item())
                for layer in args.layers:
                    residual_chunks[spec.name][layer].append(residual_by_layer[layer][mask].cpu())
                    token_counts[spec.name][layer] += selected_count

    fit_residuals: Dict[str, Dict[int, torch.Tensor]] = {
        spec.name: {} for spec in distributions
    }
    for spec in distributions:
        for layer in args.layers:
            chunks = residual_chunks[spec.name][layer]
            fit_residuals[spec.name][layer] = torch.cat(chunks, dim=0) if chunks else _empty_layer_tensor(hidden_size)
    return fit_residuals, token_counts


def build_residual_models(
    *,
    args: argparse.Namespace,
    distributions: Sequence[TokenDistributionSpec],
    fit_residuals: Dict[str, Dict[int, torch.Tensor]],
) -> Dict[str, Dict[int, Dict[str, Dict[str, Any]]]]:
    models: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]] = {
        spec.name: {} for spec in distributions
    }
    for spec in distributions:
        for layer in args.layers:
            residuals = fit_residuals[spec.name][layer]
            models[spec.name][layer] = {
                BASELINE_CODEBOOK: fit_residual_codebook(residuals, k=args.codebook_k, seed=args.seed),
                BASELINE_PCA: fit_low_rank_residual_model(residuals, rank=args.pca_rank),
            }
    return models


def _metric_accumulators(
    *,
    args: argparse.Namespace,
    distributions: Sequence[TokenDistributionSpec],
) -> Dict[str, Dict[int, Dict[str, ReconstructionMetricsAccumulator]]]:
    return {
        spec.name: {
            layer: {
                BASELINE_PROTOTYPE: ReconstructionMetricsAccumulator(
                    cosine_thresholds=args.coverage_cosine_thresholds,
                    mse_thresholds=args.coverage_mse_thresholds,
                ),
                BASELINE_CODEBOOK: ReconstructionMetricsAccumulator(
                    cosine_thresholds=args.coverage_cosine_thresholds,
                    mse_thresholds=args.coverage_mse_thresholds,
                ),
                BASELINE_PCA: ReconstructionMetricsAccumulator(
                    cosine_thresholds=args.coverage_cosine_thresholds,
                    mse_thresholds=args.coverage_mse_thresholds,
                ),
            }
            for layer in args.layers
        }
        for spec in distributions
    }


@torch.no_grad()
def evaluate_split(
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    examples: Sequence[Any],
    masks: Sequence[Dict[str, List[bool]]],
    distributions: Sequence[TokenDistributionSpec],
    bank_tensors: Dict[int, torch.Tensor],
    residual_models: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]],
) -> tuple[Dict[str, Dict[int, Dict[str, ReconstructionMetricsAccumulator]]], Dict[str, Dict[int, int]]]:
    metrics = _metric_accumulators(args=args, distributions=distributions)
    eval_token_counts: Dict[str, Dict[int, int]] = {
        spec.name: {layer: 0 for layer in args.layers}
        for spec in distributions
    }

    for start in range(0, len(examples), args.batch_size):
        batch_examples = examples[start : start + args.batch_size]
        batch_masks = masks[start : start + args.batch_size]
        input_ids, attention_mask, lengths = collate_prompt_examples(batch_examples, tokenizer.pad_token_id, args.device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_by_layer = {
            layer: maybe_normalize(outputs.hidden_states[layer].detach().to(torch.float32), args.normalize_states)
            for layer in args.layers
        }

        for row, (example, mask_by_distribution, seq_len) in enumerate(zip(batch_examples, batch_masks, lengths)):
            if seq_len <= 0:
                continue
            token_ids = torch.tensor(list(example.input_ids or [])[:seq_len], dtype=torch.long, device=args.device)
            hidden_row: Dict[int, torch.Tensor] = {}
            prototype_row: Dict[int, torch.Tensor] = {}
            residual_row: Dict[int, torch.Tensor] = {}
            for layer in args.layers:
                hidden_states = hidden_by_layer[layer][row, :seq_len, :]
                prototypes = lookup_prototypes(bank_tensors[layer], token_ids, device=hidden_states.device).to(torch.float32)
                hidden_row[layer] = hidden_states
                prototype_row[layer] = prototypes
                residual_row[layer] = hidden_states - prototypes

            for spec in distributions:
                mask = _mask_tensor(mask_by_distribution[spec.name], seq_len, args.device)
                if not torch.any(mask):
                    continue
                selected_count = int(mask.sum().item())
                for layer in args.layers:
                    eval_token_counts[spec.name][layer] += selected_count

                    targets = hidden_row[layer][mask].cpu()
                    prototypes = prototype_row[layer][mask].cpu()
                    residuals = residual_row[layer][mask].cpu()
                    metrics[spec.name][layer][BASELINE_PROTOTYPE].update(targets, prototypes)

                    codebook_model = residual_models[spec.name][layer][BASELINE_CODEBOOK]
                    if codebook_model.get("available", False):
                        codebook_residuals = reconstruct_with_codebook(residuals, codebook_model["centroids"])
                        metrics[spec.name][layer][BASELINE_CODEBOOK].update(targets, prototypes + codebook_residuals)

                    pca_model = residual_models[spec.name][layer][BASELINE_PCA]
                    if pca_model.get("available", False):
                        pca_residuals = reconstruct_with_pca(residuals, pca_model["mean"], pca_model["basis"])
                        metrics[spec.name][layer][BASELINE_PCA].update(targets, prototypes + pca_residuals)

    return metrics, eval_token_counts


def _baseline_fit_metadata(baseline_name: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
    if baseline_name == BASELINE_CODEBOOK:
        return {
            "model_available": bool(model_info.get("available", False)),
            "backend": model_info.get("backend"),
            "num_centroids": model_info.get("num_centroids"),
        }
    if baseline_name == BASELINE_PCA:
        return {
            "model_available": bool(model_info.get("available", False)),
            "rank": model_info.get("rank"),
        }
    return {"model_available": True}


def write_layer_metrics_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    cosine_thresholds: Sequence[float],
    mse_thresholds: Sequence[float],
) -> None:
    fieldnames = [
        "distribution",
        "layer",
        "baseline",
        "fit_token_count",
        "eval_token_count",
        "token_count",
        "model_available",
        "backend",
        "num_centroids",
        "rank",
        "cosine_mean",
        "cosine_median",
        "mse_mean_per_dim",
        "explained_variance",
    ]
    fieldnames.extend(f"coverage_cosine_{threshold:g}" for threshold in cosine_thresholds)
    fieldnames.extend(f"coverage_mse_per_dim_{threshold:g}" for threshold in mse_thresholds)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_residual_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_args(args)
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    if args.layers is None:
        args.layers = list(range(num_hidden_state_slots(model)))
    print(f"Using layers: {args.layers}")
    validate_layers(args.layers, model)

    distributions = load_distribution_specs(args.distribution_config)
    print(f"Loaded distributions: {[spec.name for spec in distributions]}")

    examples, dataset_meta = load_input_examples(args, tokenizer)
    fit_examples, eval_examples = split_examples_deterministically(examples, args.fit_fraction, args.seed)
    fit_masks = _build_distribution_masks(fit_examples, tokenizer, distributions)
    eval_masks = _build_distribution_masks(eval_examples, tokenizer, distributions)

    bank_tensors = {
        layer: load_bank_tensor(args.bank_dir, layer, device="cpu", normalize=args.normalize_bank)
        for layer in args.layers
    }
    hidden_size = hidden_size_from_model(model)

    fit_residuals, fit_token_counts = collect_fit_residuals(
        args=args,
        model=model,
        tokenizer=tokenizer,
        examples=fit_examples,
        masks=fit_masks,
        distributions=distributions,
        bank_tensors=bank_tensors,
        hidden_size=hidden_size,
    )
    residual_models = build_residual_models(
        args=args,
        distributions=distributions,
        fit_residuals=fit_residuals,
    )
    metrics, eval_token_counts = evaluate_split(
        args=args,
        model=model,
        tokenizer=tokenizer,
        examples=eval_examples,
        masks=eval_masks,
        distributions=distributions,
        bank_tensors=bank_tensors,
        residual_models=residual_models,
    )

    results: Dict[str, Dict[str, Any]] = {}
    csv_rows: List[Dict[str, Any]] = []
    for spec in distributions:
        layer_payloads: Dict[str, Any] = {}
        for layer in args.layers:
            layer_result = {
                "fit_token_count": int(fit_token_counts[spec.name][layer]),
                "eval_token_count": int(eval_token_counts[spec.name][layer]),
                "baselines": {},
            }
            for baseline_name in (BASELINE_PROTOTYPE, BASELINE_CODEBOOK, BASELINE_PCA):
                metric_payload = metrics[spec.name][layer][baseline_name].finalize()
                fit_metadata = _baseline_fit_metadata(
                    baseline_name,
                    residual_models[spec.name][layer].get(baseline_name, {}),
                )
                baseline_payload = {**fit_metadata, **metric_payload}
                layer_result["baselines"][baseline_name] = baseline_payload

                flat_row = {
                    "distribution": spec.name,
                    "layer": int(layer),
                    "baseline": baseline_name,
                    "fit_token_count": int(fit_token_counts[spec.name][layer]),
                    "eval_token_count": int(eval_token_counts[spec.name][layer]),
                    "token_count": baseline_payload["token_count"],
                    "model_available": baseline_payload.get("model_available"),
                    "backend": baseline_payload.get("backend"),
                    "num_centroids": baseline_payload.get("num_centroids"),
                    "rank": baseline_payload.get("rank"),
                    "cosine_mean": baseline_payload["cosine_mean"],
                    "cosine_median": baseline_payload["cosine_median"],
                    "mse_mean_per_dim": baseline_payload["mse_mean_per_dim"],
                    "explained_variance": baseline_payload["explained_variance"],
                }
                for threshold_key, value in baseline_payload["coverage_cosine"].items():
                    flat_row[f"coverage_cosine_{threshold_key}"] = value
                for threshold_key, value in baseline_payload["coverage_mse_per_dim"].items():
                    flat_row[f"coverage_mse_per_dim_{threshold_key}"] = value
                csv_rows.append(flat_row)
            layer_payloads[str(layer)] = layer_result
        results[spec.name] = layer_payloads

    summary = {
        "model": args.model,
        "bank_dir": args.bank_dir,
        "out_dir": args.out_dir,
        "layers": [int(layer) for layer in args.layers],
        "normalize_bank": bool(args.normalize_bank),
        "normalize_states": bool(args.normalize_states),
        "fit_fraction": float(args.fit_fraction),
        "seed": int(args.seed),
        "codebook_k": int(args.codebook_k),
        "pca_rank": int(args.pca_rank),
        "coverage_cosine_thresholds": [float(threshold) for threshold in args.coverage_cosine_thresholds],
        "coverage_mse_thresholds": [float(threshold) for threshold in args.coverage_mse_thresholds],
        "num_examples_loaded": int(len(examples)),
        "num_fit_examples": int(len(fit_examples)),
        "num_eval_examples": int(len(eval_examples)),
        "dataset_format": "chat",
        "dataset_source": dataset_meta["dataset_source"],
        "dataset_name": dataset_meta["dataset_name"],
        "dataset_split": dataset_meta["dataset_split"],
        "dataset_revision": dataset_meta["dataset_revision"],
        "distributions": [asdict(spec) for spec in distributions],
        "results": results,
    }

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_layer_metrics_csv(
        out_dir / "layer_metrics.csv",
        csv_rows,
        cosine_thresholds=args.coverage_cosine_thresholds,
        mse_thresholds=args.coverage_mse_thresholds,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_residual_evaluation(args)
    print(
        f"Wrote residual evaluation summary for {summary['num_eval_examples']} held-out examples "
        f"to {Path(args.out_dir) / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
