from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import build_contextual_bank
from common import (
    PromptExample,
    SparseTokenCodebookAccumulator,
    SparseTokenStatsAccumulator,
    TokenDistributionSpec,
    build_distribution_token_mask,
    prepare_prompt_examples_from_chat_objects,
)


class TinyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self._vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2}
        self._special_chars = set("<>|:\n")

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
        out = {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
        }
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


class TinyModel:
    def __init__(self):
        self.config = SimpleNamespace(hidden_size=4, num_hidden_layers=2, use_cache=True)

    def eval(self):
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        return_dict: bool = True,
    ):
        x = input_ids.to(torch.float32)
        prev = torch.zeros_like(x)
        prev[:, 1:] = x[:, :-1]
        pos = torch.arange(x.size(1), device=x.device, dtype=torch.float32).unsqueeze(0).expand_as(x)
        h0 = torch.stack([x, pos, prev, x + prev], dim=-1)
        h1 = torch.stack([x + 0.5 * prev, pos + prev, x - prev, (x + 1.0) * (pos + 1.0)], dim=-1)
        h2 = torch.stack([x + prev, 0.5 * pos + prev, 2.0 * x - prev, (prev + 1.0) * (pos + 1.0)], dim=-1)
        return SimpleNamespace(hidden_states=(h0, h1, h2))


class ContextualBankTests(unittest.TestCase):
    def test_dataset_source_validation(self):
        local_args = Namespace(
            dataset_file="data.jsonl",
            hf_dataset=None,
            batch_size=1,
            codebook_k=2,
            codebook_min_count=2,
        )
        build_contextual_bank._validate_args(local_args)

        hf_args = Namespace(
            dataset_file=None,
            hf_dataset="HuggingFaceH4/capybara",
            batch_size=1,
            codebook_k=2,
            codebook_min_count=2,
        )
        build_contextual_bank._validate_args(hf_args)

        with self.assertRaisesRegex(ValueError, "exactly one dataset source"):
            build_contextual_bank._validate_args(Namespace(
                dataset_file="data.jsonl",
                hf_dataset="HuggingFaceH4/capybara",
                batch_size=1,
                codebook_k=2,
                codebook_min_count=2,
            ))

        with self.assertRaisesRegex(ValueError, "exactly one dataset source"):
            build_contextual_bank._validate_args(Namespace(
                dataset_file=None,
                hf_dataset=None,
                batch_size=1,
                codebook_k=2,
                codebook_min_count=2,
            ))

    def test_distribution_filtering(self):
        tokenizer = TinyTokenizer()
        token_ids = [11, 12, tokenizer._get_id("<"), 13]
        example = PromptExample(
            raw_text="",
            input_ids=token_ids,
            attention_mask=[1, 1, 1, 1],
            content_token_mask=[True, False, False, True],
            template_control_token_mask=[False, True, True, False],
        )

        content_only = TokenDistributionSpec(
            name="content_only",
            include_content_tokens=True,
            include_template_control_tokens=False,
            include_special_tokens=False,
        )
        template_only = TokenDistributionSpec(
            name="template_only",
            include_content_tokens=False,
            include_template_control_tokens=True,
            include_special_tokens=False,
        )
        template_with_special = TokenDistributionSpec(
            name="template_special",
            include_content_tokens=False,
            include_template_control_tokens=True,
            include_special_tokens=True,
        )

        self.assertEqual(build_distribution_token_mask(example, tokenizer, content_only), [True, False, False, True])
        self.assertEqual(build_distribution_token_mask(example, tokenizer, template_only), [False, True, False, False])
        self.assertEqual(build_distribution_token_mask(example, tokenizer, template_with_special), [False, True, True, False])

    def test_hf_chat_row_normalization(self):
        tokenizer = TinyTokenizer()
        examples = prepare_prompt_examples_from_chat_objects(
            objects=[
                {
                    "messages": [
                        {"role": "system", "content": "Guide"},
                        {"role": "user", "content": "ab"},
                        {"role": "assistant", "content": "cb"},
                    ]
                }
            ],
            tokenizer=tokenizer,
            max_prompts=None,
            max_prompt_tokens=256,
            add_generation_prompt=False,
            source_metadata={"dataset_source": "huggingface", "dataset_name": "HuggingFaceH4/capybara"},
        )

        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example.metadata["dataset_source"], "huggingface")
        self.assertEqual(example.metadata["dataset_name"], "HuggingFaceH4/capybara")
        self.assertIn("messages", example.metadata)
        self.assertGreater(sum(int(x) for x in example.content_token_mask or []), 0)
        self.assertGreater(sum(int(x) for x in example.template_control_token_mask or []), 0)
        self.assertEqual(len(example.input_ids or []), len(example.content_token_mask or []))
        self.assertEqual(len(example.input_ids or []), len(example.template_control_token_mask or []))

    def test_sparse_stats_matches_exact_moments(self):
        acc = SparseTokenStatsAccumulator(hidden_size=2)
        acc.update(7, torch.tensor([1.0, 2.0]))
        acc.update(7, torch.tensor([3.0, 4.0]))
        payload = acc.finalize(torch.float32)

        self.assertTrue(torch.equal(payload["token_ids"], torch.tensor([7], dtype=torch.long)))
        self.assertTrue(torch.equal(payload["counts"], torch.tensor([2], dtype=torch.long)))
        self.assertTrue(torch.allclose(payload["mean"], torch.tensor([[2.0, 3.0]])))
        self.assertTrue(torch.allclose(payload["variance"], torch.tensor([[1.0, 1.0]])))

    def test_codebook_low_count_fallback_and_count_conservation(self):
        stats = SparseTokenStatsAccumulator(hidden_size=2)
        codebook = SparseTokenCodebookAccumulator(hidden_size=2, k=2)

        low_count_vectors = [torch.tensor([1.0, 1.0]), torch.tensor([3.0, 3.0])]
        for vec in low_count_vectors:
            stats.update(5, vec)
            codebook.update(5, vec)

        high_count_vectors = [
            torch.tensor([0.0, 0.0]),
            torch.tensor([10.0, 10.0]),
            torch.tensor([0.1, 0.1]),
            torch.tensor([10.1, 10.1]),
        ]
        for vec in high_count_vectors:
            stats.update(9, vec)
            codebook.update(9, vec)

        payload = codebook.finalize(torch.float32, stats_accumulator=stats, min_count=3)
        row_low = int((payload["token_ids"] == 5).nonzero(as_tuple=False)[0, 0].item())
        row_high = int((payload["token_ids"] == 9).nonzero(as_tuple=False)[0, 0].item())

        self.assertEqual(int(payload["active_clusters"][row_low].item()), 1)
        self.assertEqual(int(payload["cluster_counts"][row_low].sum().item()), 2)
        self.assertTrue(torch.allclose(payload["centroids"][row_low, 0], torch.tensor([2.0, 2.0])))

        self.assertEqual(int(payload["active_clusters"][row_high].item()), 2)
        self.assertEqual(int(payload["cluster_counts"][row_high].sum().item()), 4)
        self.assertEqual(tuple(payload["centroids"].shape), (2, 2, 2))

    def test_end_to_end_contextual_bank_smoke(self):
        tokenizer = TinyTokenizer()
        model = TinyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "dataset.jsonl"
            dataset_path.write_text(
                json.dumps({"messages": [{"role": "user", "content": "ab"}, {"role": "assistant", "content": "cb"}]}) + "\n"
                + json.dumps({"messages": [{"role": "user", "content": "xb"}, {"role": "assistant", "content": "yb"}]}) + "\n",
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
            out_dir = tmp_path / "artifacts"

            args = Namespace(
                model="tiny-model",
                dataset_file=str(dataset_path),
                distribution_config=str(config_path),
                out_dir=str(out_dir),
                layers=[0, 1, 2],
                device="cpu",
                dtype="float32",
                max_examples=None,
                max_prompt_tokens=256,
                batch_size=2,
                normalize_states=False,
                codebook_k=2,
                codebook_min_count=2,
                trust_remote_code=False,
                attn_implementation=None,
            )

            with patch("build_contextual_bank.load_model_and_tokenizer", return_value=(model, tokenizer)):
                build_contextual_bank.run_contextual_bank(args)

            meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["dataset_format"], "chat")
            self.assertEqual(meta["dataset_source"], "local_jsonl")
            self.assertIsNone(meta["dataset_name"])
            self.assertEqual(meta["layers"], [0, 1, 2])

            stats_path = out_dir / "distribution_content_only" / "layer_1_stats.pt"
            codebook_path = out_dir / "distribution_content_only" / "layer_1_codebook.pt"
            self.assertTrue(stats_path.exists())
            self.assertTrue(codebook_path.exists())

            stats_payload = torch.load(stats_path, map_location="cpu")
            codebook_payload = torch.load(codebook_path, map_location="cpu")
            self.assertTrue(torch.all(stats_payload["token_ids"][:-1] <= stats_payload["token_ids"][1:]))

            token_b = tokenizer._get_id("b")
            row = int((stats_payload["token_ids"] == token_b).nonzero(as_tuple=False)[0, 0].item())
            self.assertEqual(int(stats_payload["counts"][row].item()), 4)
            self.assertGreater(float(stats_payload["variance"][row].abs().sum().item()), 0.0)
            self.assertEqual(int(codebook_payload["active_clusters"][row].item()), 2)
            self.assertEqual(int(codebook_payload["cluster_counts"][row].sum().item()), 4)

    def test_end_to_end_hf_contextual_bank_smoke(self):
        tokenizer = TinyTokenizer()
        model = TinyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
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
            out_dir = tmp_path / "artifacts_hf"

            args = Namespace(
                model="tiny-model",
                dataset_file=None,
                hf_dataset="HuggingFaceH4/capybara",
                hf_split="train",
                hf_revision="main",
                hf_streaming=False,
                distribution_config=str(config_path),
                out_dir=str(out_dir),
                layers=[0, 1, 2],
                device="cpu",
                dtype="float32",
                max_examples=2,
                max_prompt_tokens=256,
                batch_size=2,
                normalize_states=False,
                codebook_k=2,
                codebook_min_count=2,
                trust_remote_code=False,
                attn_implementation=None,
            )

            hf_rows = [
                {"messages": [{"role": "user", "content": "ab"}, {"role": "assistant", "content": "cb"}]},
                {"messages": [{"role": "system", "content": "hint"}, {"role": "user", "content": "xb"}, {"role": "assistant", "content": "yb"}]},
            ]

            with patch("build_contextual_bank.load_model_and_tokenizer", return_value=(model, tokenizer)):
                with patch("build_contextual_bank.load_hf_chat_objects", return_value=hf_rows):
                    build_contextual_bank.run_contextual_bank(args)

            meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["dataset_source"], "huggingface")
            self.assertEqual(meta["dataset_name"], "HuggingFaceH4/capybara")
            self.assertEqual(meta["dataset_split"], "train")
            self.assertEqual(meta["dataset_revision"], "main")
            self.assertEqual(meta["num_examples_processed"], 2)

            stats_path = out_dir / "distribution_content_only" / "layer_1_stats.pt"
            codebook_path = out_dir / "distribution_content_only" / "layer_1_codebook.pt"
            stats_payload = torch.load(stats_path, map_location="cpu")
            codebook_payload = torch.load(codebook_path, map_location="cpu")

            self.assertTrue(torch.all(stats_payload["token_ids"][:-1] <= stats_payload["token_ids"][1:]))
            token_b = tokenizer._get_id("b")
            row = int((stats_payload["token_ids"] == token_b).nonzero(as_tuple=False)[0, 0].item())
            self.assertGreaterEqual(int(stats_payload["counts"][row].item()), 4)
            self.assertGreater(float(stats_payload["variance"][row].abs().sum().item()), 0.0)
            self.assertEqual(int(codebook_payload["cluster_counts"][row].sum().item()), int(stats_payload["counts"][row].item()))

    def test_normalized_contextual_bank_records_state_space_and_changes_payloads(self):
        tokenizer = TinyTokenizer()
        model = TinyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "dataset.jsonl"
            dataset_path.write_text(
                json.dumps({"messages": [{"role": "user", "content": "ab"}, {"role": "assistant", "content": "cb"}]}) + "\n"
                + json.dumps({"messages": [{"role": "user", "content": "xb"}, {"role": "assistant", "content": "yb"}]}) + "\n",
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
            raw_out_dir = tmp_path / "artifacts_raw"
            norm_out_dir = tmp_path / "artifacts_norm"

            raw_args = Namespace(
                model="tiny-model",
                dataset_file=str(dataset_path),
                hf_dataset=None,
                hf_split="train",
                hf_revision=None,
                hf_streaming=False,
                distribution_config=str(config_path),
                out_dir=str(raw_out_dir),
                layers=[0, 1, 2],
                device="cpu",
                dtype="float32",
                max_examples=None,
                max_prompt_tokens=256,
                batch_size=2,
                normalize_states=False,
                codebook_k=2,
                codebook_min_count=2,
                trust_remote_code=False,
                attn_implementation=None,
            )
            norm_args = Namespace(
                model="tiny-model",
                dataset_file=str(dataset_path),
                hf_dataset=None,
                hf_split="train",
                hf_revision=None,
                hf_streaming=False,
                distribution_config=str(config_path),
                out_dir=str(norm_out_dir),
                layers=[0, 1, 2],
                device="cpu",
                dtype="float32",
                max_examples=None,
                max_prompt_tokens=256,
                batch_size=2,
                normalize_states=True,
                codebook_k=2,
                codebook_min_count=2,
                trust_remote_code=False,
                attn_implementation=None,
            )

            with patch("build_contextual_bank.load_model_and_tokenizer", return_value=(model, tokenizer)):
                build_contextual_bank.run_contextual_bank(raw_args)
            with patch("build_contextual_bank.load_model_and_tokenizer", return_value=(model, tokenizer)):
                build_contextual_bank.run_contextual_bank(norm_args)

            raw_meta = json.loads((raw_out_dir / "meta.json").read_text(encoding="utf-8"))
            norm_meta = json.loads((norm_out_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(raw_meta["state_space"], "raw")
            self.assertEqual(norm_meta["state_space"], "normalized")
            self.assertFalse(raw_meta["posthoc_normalization_supported"])
            self.assertFalse(norm_meta["posthoc_normalization_supported"])

            raw_stats = torch.load(raw_out_dir / "distribution_content_only" / "layer_1_stats.pt", map_location="cpu")
            norm_stats = torch.load(norm_out_dir / "distribution_content_only" / "layer_1_stats.pt", map_location="cpu")
            token_b = tokenizer._get_id("b")
            row_raw = int((raw_stats["token_ids"] == token_b).nonzero(as_tuple=False)[0, 0].item())
            row_norm = int((norm_stats["token_ids"] == token_b).nonzero(as_tuple=False)[0, 0].item())

            self.assertTrue(torch.equal(raw_stats["counts"], norm_stats["counts"]))
            self.assertFalse(torch.allclose(raw_stats["mean"][row_raw], norm_stats["mean"][row_norm]))


if __name__ == "__main__":
    unittest.main()
