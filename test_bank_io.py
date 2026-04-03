from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from common import build_faiss_index, load_bank_tensor, save_bank_metadata
from probe_states import ActivationBank as ProbeActivationBank


class BankIOTests(unittest.TestCase):
    def test_load_bank_tensor_can_normalize_raw_pt_bank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_dir = Path(tmpdir)
            bank = torch.tensor([
                [3.0, 4.0],
                [5.0, 12.0],
            ], dtype=torch.float32)
            torch.save(bank, bank_dir / "layer_0.pt")

            loaded = load_bank_tensor(bank_dir, layer=0, device="cpu", normalize=True)

            self.assertTrue(torch.allclose(torch.linalg.norm(loaded, dim=-1), torch.ones(2), atol=1e-6))
            self.assertTrue(torch.allclose(loaded[0], torch.tensor([0.6, 0.8]), atol=1e-6))

    def test_build_faiss_index_normalizes_raw_vectors_for_cosine(self):
        vectors = torch.tensor([
            [10.0, 0.0],
            [0.0, 5.0],
        ], dtype=torch.float32)
        index = build_faiss_index(vectors, use_cosine=True)
        query = torch.tensor([1.0, 0.0], dtype=torch.float32)

        scores, ids = index.search(query.unsqueeze(0).numpy().astype("float32"), 1)

        self.assertEqual(int(ids[0, 0]), 0)
        self.assertTrue(math.isclose(float(scores[0, 0]), 1.0, rel_tol=1e-6, abs_tol=1e-6))

    def test_probe_bank_rebuilds_on_saved_faiss_metric_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_dir = Path(tmpdir)
            bank = torch.tensor([
                [10.0, 0.0],
                [1.0, 1.0],
            ], dtype=torch.float32)
            torch.save(bank, bank_dir / "layer_0.pt")
            save_bank_metadata(
                bank_dir,
                vocab_size=2,
                hidden_size=2,
                dtype=torch.float32,
                layers=[0],
                tensor_storage="raw",
                faiss_metric="l2",
            )

            import faiss

            faiss.write_index(build_faiss_index(bank, use_cosine=False), str(bank_dir / "layer_0.faiss"))

            activation_bank = ProbeActivationBank(str(bank_dir), [0])
            activation_bank.load(device="cpu", use_faiss=True, use_cosine=True)

            query = torch.tensor([1.0, 1.0], dtype=torch.float32)
            query = query / torch.linalg.norm(query)
            nn_id, score, rank_true = activation_bank.search(0, query, true_token_id=1, topk_true_rank=2, use_cosine=True)

            self.assertEqual(nn_id, 1)
            self.assertEqual(rank_true, 1)
            self.assertGreater(score, 0.99)

    def test_probe_bank_rebuilds_legacy_faiss_without_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_dir = Path(tmpdir)
            bank = torch.tensor([
                [10.0, 0.0],
                [1.0, 1.0],
            ], dtype=torch.float32)
            torch.save(bank, bank_dir / "layer_0.pt")
            (bank_dir / "bank_meta.json").write_text(
                json.dumps(
                    {
                        "shape": [2, 2],
                        "dtype": "float32",
                        "layers": [0],
                    }
                ),
                encoding="utf-8",
            )

            import faiss

            faiss.write_index(build_faiss_index(bank, use_cosine=False), str(bank_dir / "layer_0.faiss"))

            activation_bank = ProbeActivationBank(str(bank_dir), [0])
            activation_bank.load(device="cpu", use_faiss=True, use_cosine=True)

            query = torch.tensor([1.0, 1.0], dtype=torch.float32)
            query = query / torch.linalg.norm(query)
            nn_id, score, rank_true = activation_bank.search(0, query, true_token_id=1, topk_true_rank=2, use_cosine=True)

            self.assertEqual(nn_id, 1)
            self.assertEqual(rank_true, 1)
            self.assertGreater(score, 0.99)


if __name__ == "__main__":
    unittest.main()
