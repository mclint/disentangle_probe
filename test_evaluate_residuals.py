from __future__ import annotations

import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import evaluate_residuals
from common import ReconstructionMetricsAccumulator, lookup_prototypes, split_examples_deterministically


class TinyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self._vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2}
        self._special_chars = set("<>|:\n")

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def all_special_ids(self):
        ids = {self.pad_token_id, self.eos_token_id, self.bos_token_id}
        ids.update(self._get_id(ch) for ch in self._special_chars)
        return sorted(ids)

    def _get_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab)
        return self._vocab[token]

    def convert_ids_to_tokens(self, token_id: int) -> str:
        for token, idx in self._vocab.items():
            if idx == int(token_id):
                return token
        return str(token_id)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "".join(f"<|{msg['role']}|>:{msg['content']}\n" for msg in messages)
        if add_generation_prompt:
            rendered += "<|assistant|>:"
        if tokenize:
            return [self._get_id(ch) for ch in rendered]
        return rendered

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        return_tensors=None,
        truncation: bool = False,
        max_length: int | None = None,
    ):
        tokens = []
        offsets = []
        for idx, ch in enumerate(text):
            tokens.append(self._get_id(ch))
            offsets.append((idx, idx + 1))
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
            offsets = offsets[:max_length]
        payload = {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
        }
        if return_offsets_mapping:
            payload["offset_mapping"] = offsets
        return payload


class ResidualToyModel:
    def __init__(self, tokenizer: TinyTokenizer):
        self.tokenizer = tokenizer
        self.config = SimpleNamespace(hidden_size=4, num_hidden_layers=1, use_cache=True)
        positive_chars = {"a", "c", "e", "g", "i", "k"}
        self.positive_ids = {tokenizer._get_id(ch) for ch in positive_chars}

    def eval(self):
        return self

    def _prototype(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = input_ids.to(torch.float32)
        return torch.stack([
            x,
            0.1 * x,
            torch.ones_like(x),
            torch.zeros_like(x),
        ], dim=-1)

    def _residual(self, input_ids: torch.Tensor) -> torch.Tensor:
        positive = torch.zeros_like(input_ids, dtype=torch.float32)
        for token_id in self.positive_ids:
            positive = positive + (input_ids == int(token_id)).to(torch.float32)
        sign = torch.where(positive > 0.0, 1.0, -1.0)
        return torch.stack([
            torch.zeros_like(sign),
            torch.zeros_like(sign),
            torch.zeros_like(sign),
            sign,
        ], dim=-1)

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        return_dict: bool = True,
    ):
        prototype = self._prototype(input_ids)
        h0 = prototype
        h1 = prototype + self._residual(input_ids)
        return SimpleNamespace(hidden_states=(h0, h1))


def seed_tokenizer_from_messages(tokenizer: TinyTokenizer, rows) -> None:
    for row in rows:
        rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)


def write_toy_bank(bank_dir: Path, tokenizer: TinyTokenizer) -> None:
    vocab_size = tokenizer.vocab_size
    hidden_size = 4
    token_ids = torch.arange(vocab_size, dtype=torch.float32)
    prototype = torch.stack([
        token_ids,
        0.1 * token_ids,
        torch.ones_like(token_ids),
        torch.zeros_like(token_ids),
    ], dim=-1)
    torch.save(prototype.clone(), bank_dir / "layer_0.pt")
    torch.save(prototype.clone(), bank_dir / "layer_1.pt")


class ResidualEvaluationTests(unittest.TestCase):
    def test_split_examples_deterministically(self):
        fit_a, eval_a = split_examples_deterministically(list(range(10)), fit_fraction=0.6, seed=7)
        fit_b, eval_b = split_examples_deterministically(list(range(10)), fit_fraction=0.6, seed=7)
        self.assertEqual(fit_a, fit_b)
        self.assertEqual(eval_a, eval_b)
        self.assertEqual(len(fit_a), 6)
        self.assertEqual(len(eval_a), 4)

    def test_lookup_prototypes_exact(self):
        bank = torch.tensor([
            [0.0, 0.0],
            [1.0, 1.5],
            [2.0, 2.5],
            [3.0, 3.5],
        ])
        token_ids = torch.tensor([3, 1, 2], dtype=torch.long)
        gathered = lookup_prototypes(bank, token_ids, device="cpu")
        expected = torch.tensor([
            [3.0, 3.5],
            [1.0, 1.5],
            [2.0, 2.5],
        ])
        self.assertTrue(torch.allclose(gathered, expected))

    def test_codebook_and_pca_improve_over_prototype(self):
        targets = torch.tensor([
            [1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ])
        prototypes = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ])
        residuals = targets - prototypes

        codebook_model = evaluate_residuals.fit_residual_codebook(residuals, k=2, seed=42)
        pca_model = evaluate_residuals.fit_low_rank_residual_model(residuals, rank=1)

        prototype_metrics = ReconstructionMetricsAccumulator(cosine_thresholds=[0.99], mse_thresholds=[0.01])
        prototype_metrics.update(targets, prototypes)
        codebook_metrics = ReconstructionMetricsAccumulator(cosine_thresholds=[0.99], mse_thresholds=[0.01])
        codebook_metrics.update(targets, prototypes + evaluate_residuals.reconstruct_with_codebook(residuals, codebook_model["centroids"]))
        pca_metrics = ReconstructionMetricsAccumulator(cosine_thresholds=[0.99], mse_thresholds=[0.01])
        pca_metrics.update(targets, prototypes + evaluate_residuals.reconstruct_with_pca(residuals, pca_model["mean"], pca_model["basis"]))

        prototype_summary = prototype_metrics.finalize()
        codebook_summary = codebook_metrics.finalize()
        pca_summary = pca_metrics.finalize()

        self.assertLess(prototype_summary["cosine_mean"], codebook_summary["cosine_mean"])
        self.assertLess(prototype_summary["cosine_mean"], pca_summary["cosine_mean"])
        self.assertTrue(math.isclose(codebook_summary["explained_variance"], 1.0, rel_tol=1e-6, abs_tol=1e-6))
        self.assertTrue(math.isclose(pca_summary["explained_variance"], 1.0, rel_tol=1e-6, abs_tol=1e-6))

    def test_end_to_end_residual_evaluation_smoke(self):
        rows = [
            {"messages": [{"role": "user", "content": "ab"}, {"role": "assistant", "content": "cd"}]},
            {"messages": [{"role": "user", "content": "ef"}, {"role": "assistant", "content": "ga"}]},
            {"messages": [{"role": "user", "content": "bc"}, {"role": "assistant", "content": "de"}]},
            {"messages": [{"role": "user", "content": "fg"}, {"role": "assistant", "content": "ab"}]},
            {"messages": [{"role": "user", "content": "ce"}, {"role": "assistant", "content": "gb"}]},
            {"messages": [{"role": "user", "content": "da"}, {"role": "assistant", "content": "fc"}]},
        ]

        tokenizer = TinyTokenizer()
        seed_tokenizer_from_messages(tokenizer, rows)
        model = ResidualToyModel(tokenizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "dataset.jsonl"
            dataset_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            config_path = tmp_path / "distributions.json"
            config_path.write_text(
                json.dumps({
                    "distributions": [
                        {
                            "name": "content_only",
                            "include_content_tokens": True,
                            "include_template_control_tokens": False,
                            "include_special_tokens": False,
                        }
                    ]
                }),
                encoding="utf-8",
            )
            bank_dir = tmp_path / "bank"
            bank_dir.mkdir()
            write_toy_bank(bank_dir, tokenizer)
            out_dir = tmp_path / "residual_eval"

            args = Namespace(
                model="tiny-model",
                dataset_file=str(dataset_path),
                hf_dataset=None,
                bank_dir=str(bank_dir),
                distribution_config=str(config_path),
                out_dir=str(out_dir),
                layers=[0, 1],
                device="cpu",
                dtype="float32",
                max_examples=None,
                max_prompt_tokens=256,
                batch_size=2,
                hf_split="train",
                hf_revision=None,
                hf_streaming=False,
                normalize_bank=False,
                normalize_states=False,
                fit_fraction=0.5,
                seed=42,
                codebook_k=2,
                pca_rank=1,
                coverage_cosine_thresholds=[0.9, 0.95, 0.98],
                coverage_mse_thresholds=[0.01, 0.05, 0.1],
                trust_remote_code=False,
                attn_implementation=None,
            )

            with patch("evaluate_residuals.load_model_and_tokenizer", return_value=(model, tokenizer)):
                summary = evaluate_residuals.run_residual_evaluation(args)

            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "layer_metrics.csv").exists())
            self.assertEqual(summary["num_fit_examples"], 3)
            self.assertEqual(summary["num_eval_examples"], 3)

            summary_json = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            layer0 = summary_json["results"]["content_only"]["0"]["baselines"]
            layer1 = summary_json["results"]["content_only"]["1"]["baselines"]

            self.assertEqual(layer0[evaluate_residuals.BASELINE_PROTOTYPE]["token_count"], summary_json["results"]["content_only"]["0"]["eval_token_count"])
            self.assertGreater(layer1[evaluate_residuals.BASELINE_CODEBOOK]["cosine_mean"], layer1[evaluate_residuals.BASELINE_PROTOTYPE]["cosine_mean"])
            self.assertGreater(layer1[evaluate_residuals.BASELINE_PCA]["cosine_mean"], layer1[evaluate_residuals.BASELINE_PROTOTYPE]["cosine_mean"])
            self.assertTrue(math.isfinite(layer1[evaluate_residuals.BASELINE_CODEBOOK]["explained_variance"]))
            self.assertTrue(math.isfinite(layer1[evaluate_residuals.BASELINE_PCA]["explained_variance"]))
            self.assertEqual(layer1[evaluate_residuals.BASELINE_CODEBOOK]["num_centroids"], 2)
            self.assertEqual(layer1[evaluate_residuals.BASELINE_PCA]["rank"], 1)

    def test_end_to_end_residual_evaluation_smoke_with_normalized_bank(self):
        rows = [
            {"messages": [{"role": "user", "content": "ab"}, {"role": "assistant", "content": "cd"}]},
            {"messages": [{"role": "user", "content": "ef"}, {"role": "assistant", "content": "ga"}]},
            {"messages": [{"role": "user", "content": "bc"}, {"role": "assistant", "content": "de"}]},
            {"messages": [{"role": "user", "content": "fg"}, {"role": "assistant", "content": "ab"}]},
            {"messages": [{"role": "user", "content": "ce"}, {"role": "assistant", "content": "gb"}]},
            {"messages": [{"role": "user", "content": "da"}, {"role": "assistant", "content": "fc"}]},
        ]

        tokenizer = TinyTokenizer()
        seed_tokenizer_from_messages(tokenizer, rows)
        model = ResidualToyModel(tokenizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "dataset.jsonl"
            dataset_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            config_path = tmp_path / "distributions.json"
            config_path.write_text(
                json.dumps({
                    "distributions": [
                        {
                            "name": "content_only",
                            "include_content_tokens": True,
                            "include_template_control_tokens": False,
                            "include_special_tokens": False,
                        }
                    ]
                }),
                encoding="utf-8",
            )
            bank_dir = tmp_path / "bank"
            bank_dir.mkdir()
            write_toy_bank(bank_dir, tokenizer)
            out_dir = tmp_path / "residual_eval"

            args = Namespace(
                model="tiny-model",
                dataset_file=str(dataset_path),
                hf_dataset=None,
                bank_dir=str(bank_dir),
                distribution_config=str(config_path),
                out_dir=str(out_dir),
                layers=[0, 1],
                device="cpu",
                dtype="float32",
                max_examples=None,
                max_prompt_tokens=256,
                batch_size=2,
                hf_split="train",
                hf_revision=None,
                hf_streaming=False,
                normalize_bank=True,
                normalize_states=True,
                fit_fraction=0.5,
                seed=42,
                codebook_k=2,
                pca_rank=1,
                coverage_cosine_thresholds=[0.9, 0.95, 0.98],
                coverage_mse_thresholds=[0.01, 0.05, 0.1],
                trust_remote_code=False,
                attn_implementation=None,
            )

            with patch("evaluate_residuals.load_model_and_tokenizer", return_value=(model, tokenizer)):
                summary = evaluate_residuals.run_residual_evaluation(args)

            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "layer_metrics.csv").exists())
            self.assertEqual(summary["num_fit_examples"], 3)
            self.assertEqual(summary["num_eval_examples"], 3)

            summary_json = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            layer0 = summary_json["results"]["content_only"]["0"]["baselines"]
            layer1 = summary_json["results"]["content_only"]["1"]["baselines"]

            self.assertEqual(layer0[evaluate_residuals.BASELINE_PROTOTYPE]["token_count"], summary_json["results"]["content_only"]["0"]["eval_token_count"])
            self.assertGreaterEqual(layer1[evaluate_residuals.BASELINE_CODEBOOK]["cosine_mean"], layer1[evaluate_residuals.BASELINE_PROTOTYPE]["cosine_mean"])
            self.assertGreaterEqual(layer1[evaluate_residuals.BASELINE_PCA]["cosine_mean"], layer1[evaluate_residuals.BASELINE_PROTOTYPE]["cosine_mean"])
            self.assertTrue(math.isfinite(layer1[evaluate_residuals.BASELINE_CODEBOOK]["explained_variance"]))
            self.assertTrue(math.isfinite(layer1[evaluate_residuals.BASELINE_PCA]["explained_variance"]))


if __name__ == "__main__":
    unittest.main()
