# AGENTS.md

## Project overview

This package measures **hidden-state disentanglement** in Hugging Face causal LMs. For each layer, it builds a vocabulary activation bank (one vector per token) and then probes prefill/decode hidden states via nearest-neighbor lookup. A state is "disentangled" if its NN in the bank matches the true token.

- For this workspace, always use the VSCode-configured Python interpreter instead of the shell default Python.
- Follow the workspace fallback logic from `.vscode/settings.json`, which selects the project conda environment for this folder.

## Key files

- `build_bank.py` — builds per-layer activation bank (raw `.memmap` tensors + optional `.faiss` indices)
- `build_contextual_bank.py` — builds sparse contextualized token stats and codebooks from chat datasets
- `probe_states.py` — probes prefill/decode states against the bank (standard attention models)
- `probe_states_hybrid.py` — same as above but handles hybrid attention (e.g. Qwen 3.5 with mixed full/linear attention layers)
- `common.py` — shared utilities: model loading, FAISS wrappers, prompt preparation, attention summarisation, output writing
- `configs/` — example distribution configs for contextualized-bank runs
- `notebooks/` — analysis/debug notebooks, including `analyze_results.ipynb`, `debug.ipynb`, and `context_bank_diagnostics.ipynb`

## Layer convention (HuggingFace hidden_states indexing)

- `hidden_states[0]` = embeddings (no transformer block, no attention)
- `hidden_states[L]` = output of transformer block `L-1`
- Total slots = `num_hidden_layers + 1`

## Anchor token in build_bank.py

Each vocab token is preceded by an anchor token when extracting activations. The `--anchor_token_id` arg controls this; if omitted it defaults to `tokenizer.bos_token_id`. Known values:
- Qwen 2.5: `248045` (`<|im_start|>`)
- Llama: `1` (`<s>`)

## Hybrid attention (Qwen 3.5)

Qwen 3.5 alternates full-attention and linear-attention (GatedDeltaNet) layers. With `full_attention_interval=4` and 32 layers, only 8 use full attention (block indices 3, 7, 11, 15, 19, 23, 27, 31 = hidden-state layers 4, 8, 12, 16, 20, 24, 28, 32).

- `outputs.attentions` is a tuple of length 8 (only full-attention layers), not 32
- `model.config.layer_types` lists `"full_attention"` or `"linear_attention"` per block
- Use `probe_states_hybrid.py` for these models; it auto-detects the layout
- Hidden states are available for all 32 layers regardless of attention type
- Disentanglement probing works on any layer; attention stats are only available for full-attention layers

## Running

```bash
# Build contextualized bank from Capybara
python build_contextual_bank.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --hf_dataset HuggingFaceH4/capybara --hf_split train_sft \
  --distribution_config ./configs/distributions.content_only.json \
  --out_dir ./context_bank_capybara_debug --layers 0 1 8 --max_examples 128 \
  --batch_size 4 --codebook_k 4 --codebook_min_count 4 --normalize_states --device cuda:0

# Build bank (auto-detects anchor from BOS, or override)
python build_bank.py --model Qwen/Qwen3-4B --layers 4 8 12 16 20 24 28 32 \
  --bank_dir ./bank_qwen35_4b --vocab_batch_size 256 --normalize_bank --device cuda:0

# Probe (hybrid model)
python probe_states_hybrid.py --model Qwen/Qwen3-4B --prompts_file prompts.txt \
  --bank_dir ./bank_qwen35_4b --out_jsonl ./probe_results.jsonl \
  --layers 4 8 12 16 20 24 28 32 --normalize_bank --normalize_queries \
  --use_faiss --collect_attention_stats --device cuda:0

# Probe (standard model)
python probe_states.py --model Qwen/Qwen2.5-1.5B --prompts_file prompts.txt \
  --bank_dir ./bank_qwen25_15b --out_jsonl ./probe_results.jsonl \
  --layers 0 1 8 16 24 --normalize_bank --normalize_queries \
  --use_faiss --collect_attention_stats --device cuda:0
```

## Dependencies

torch, transformers, faiss-cpu, numpy, jupyter, plotly, matplotlib, pandas

## Keeping docs in sync

When changing CLI flags, output schema, or adding new scripts, update both `README.md` and `CLAUDE.md` to match.

## Common flags

- `--normalize_bank` in `build_bank.py`: build cosine-compatible `.faiss` indices while keeping bank tensors raw on disk
- `--normalize_bank` + `--normalize_queries`: use cosine similarity in probing by normalizing raw bank vectors at probe time
- `--normalize_bank` + `--normalize_states`: use cosine-space prototypes in `evaluate_residuals.py`
- `--normalize_states` in `build_contextual_bank.py`: aggregate normalized hidden states directly; contextual sparse stats/codebooks are therefore stored in normalized space and cannot be post-hoc converted from raw aggregates
- `--use_faiss`: reuse saved `.faiss` only when its recorded metric matches, otherwise rebuild from raw bank tensors
- `--collect_attention_stats`: extract attention entropy, top-k mass, recency, BOS mass, etc.
- `--prompt_format chat` + `--add_generation_prompt`: for chat-templated prompts (JSONL input)
- `--trust_remote_code`: required for some model architectures
