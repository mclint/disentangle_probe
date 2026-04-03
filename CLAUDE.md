# CLAUDE.md

## Bank normalization contract

- `build_bank.py` always stores raw bank tensors on disk.
- `build_bank.py --normalize_bank` builds cosine-compatible `.faiss` indices, but does not normalize the stored `.memmap` bank.
- `probe_states.py` and `probe_states_hybrid.py` use cosine similarity only when both `--normalize_bank` and `--normalize_queries` are set.
- `evaluate_residuals.py` uses cosine-space prototypes only when both `--normalize_bank` and `--normalize_states` are set.
- `build_contextual_bank.py --normalize_states` is different: it aggregates normalized hidden states directly, so the stored sparse contextual stats/codebooks live in normalized space and cannot be recreated by post-hoc normalization of raw aggregates.

## Bank artifacts

- `layer_<L>.memmap` stores the raw bank tensor for each layer.
- `layer_<L>.faiss` stores at most one FAISS index metric for each layer.
- `bank_meta.json` records:
  - `shape`
  - `dtype`
  - `layers`
  - `tensor_storage`
  - `faiss_metric`

## FAISS reuse

- Probe scripts reuse a saved `.faiss` only when `bank_meta.json` records a matching metric.
- If the metric is missing or mismatched, probing rebuilds an in-memory FAISS index from the raw bank tensors.
- Legacy banks without `tensor_storage` metadata are treated as raw storage.
- Legacy `.faiss` files without `faiss_metric` metadata are ignored and rebuilt when raw bank tensors are available.
