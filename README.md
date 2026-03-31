# Disentanglement Probe Package

This package measures **hidden-state disentanglement** in Hugging Face causal LMs. For each layer, it builds a vocabulary activation bank (one vector per token) and then probes prefill/decode hidden states via nearest-neighbor lookup. A state is "disentangled" if its nearest neighbor in the bank matches the true token.

## Files

- `build_bank.py` — builds per-layer activation bank (.pt tensors + .faiss indices)
- `probe_states.py` — probes prefill/decode states against the bank (standard attention models)
- `probe_states_hybrid.py` — same as above but handles hybrid attention (e.g. Qwen 3.5 with mixed full/linear attention layers)
- `common.py` — shared utilities: model loading, FAISS wrappers, prompt preparation, attention summarisation, output writing
- `analyze_results.ipynb` — analysis and visualisation notebook

## Layer convention

We follow Hugging Face `hidden_states` indexing exactly:

- `hidden_states[0]` = embeddings (no transformer block, no attention)
- `hidden_states[L]` = output of transformer block `L-1`
- Total slots = `num_hidden_layers + 1`

## Labels

For a token-position state at layer `l`:

- `disentangled` if nearest-neighbor token id equals the true token id
- `entangled` otherwise

We retain both:

- `is_special_token`
- `is_template_control_token`

`is_template_control_token` is only relevant for chat-templated prefill inputs. Decode tokens are always marked `False`.

## Installation

```bash
pip install torch transformers faiss-cpu numpy jupyter plotly matplotlib pandas
```

## Device loading

The model loader tries to place the model directly on the configured device during `from_pretrained(...)` using `device_map={"": <device>}`. If that is not supported in your setup, it falls back to a normal load followed by `model.to(device)`.

## Anchor token in build_bank.py

Each vocab token is preceded by an anchor token when extracting activations. The `--anchor_token_id` flag controls this; if omitted it defaults to `tokenizer.bos_token_id`. Known values:

- Qwen 2.5: `248045` (`<|im_start|>`)
- Llama: `1` (`<s>`)

## Build the activation bank

```bash
python build_bank.py \
  --model Qwen/Qwen2.5-1.5B \
  --layers 0 1 8 16 24 \
  --bank_dir ./bank_qwen25_15b \
  --vocab_batch_size 256 \
  --normalize_bank \
  --device cuda:0
```

This writes:

- `layer_<L>.pt` raw bank tensors
- `layer_<L>.faiss` FAISS indices

Use `--skip_save_tensors` or `--skip_save_faiss` to omit either output.

## Probe raw-text prompts (standard model)

```bash
python probe_states.py \
  --model Qwen/Qwen2.5-1.5B \
  --prompts_file prompts.txt \
  --bank_dir ./bank_qwen25_15b \
  --out_jsonl ./probe_results.jsonl \
  --layers 0 1 8 16 24 \
  --prompt_format raw \
  --max_prompt_tokens 256 \
  --max_new_tokens 32 \
  --normalize_bank \
  --normalize_queries \
  --use_faiss \
  --collect_attention_stats \
  --device cuda:0
```

## Probe chat-templated prompts

For `--prompt_format chat`, the prompts file must be JSONL, one conversation per line. Each line can be either:

```json
[{"role":"system","content":"You are helpful."},{"role":"user","content":"Explain gravity simply."}]
```

or

```json
{"messages":[{"role":"user","content":"Explain gravity simply."}]}
```

Then run:

```bash
python probe_states.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompts_file chat_prompts.jsonl \
  --bank_dir ./bank_qwen25_15b \
  --out_jsonl ./probe_results_chat.jsonl \
  --layers 0 1 8 16 24 \
  --prompt_format chat \
  --add_generation_prompt \
  --max_prompt_tokens 512 \
  --max_new_tokens 64 \
  --normalize_bank \
  --normalize_queries \
  --use_faiss \
  --collect_attention_stats \
  --device cuda:0
```

Template-control tokens are identified by rendering the chat template to text, tokenizing with offset mappings, and marking tokens that fall outside user-provided message content spans.

## Hybrid attention models (e.g. Qwen 3.5)

Qwen 3.5 alternates full-attention and linear-attention (GatedDeltaNet) layers. Use `probe_states_hybrid.py` for these models; it auto-detects the layout from `model.config.layer_types`.

- `outputs.attentions` contains only full-attention layers (e.g. 8 out of 32)
- Hidden states are available for all layers regardless of attention type
- Disentanglement probing works on any layer; attention stats are only available for full-attention layers

```bash
# Build bank
python build_bank.py \
  --model Qwen/Qwen3-4B \
  --layers 4 8 12 16 20 24 28 32 \
  --bank_dir ./bank_qwen35_4b \
  --vocab_batch_size 256 \
  --normalize_bank \
  --device cuda:0

# Probe
python probe_states_hybrid.py \
  --model Qwen/Qwen3-4B \
  --prompts_file prompts.txt \
  --bank_dir ./bank_qwen35_4b \
  --out_jsonl ./probe_results.jsonl \
  --layers 4 8 12 16 20 24 28 32 \
  --normalize_bank \
  --normalize_queries \
  --use_faiss \
  --collect_attention_stats \
  --device cuda:0
```

## Common flags

| Flag | Description |
|------|-------------|
| `--normalize_bank` + `--normalize_queries` | Use cosine similarity (both must be set) |
| `--use_faiss` | Use `.faiss` index files for fast NN search |
| `--collect_attention_stats` | Extract attention entropy, top-k mass, recency, BOS mass, etc. |
| `--prompt_format chat` + `--add_generation_prompt` | For chat-templated prompts (JSONL input) |
| `--trust_remote_code` | Required for some model architectures |
| `--dtype {float16,bfloat16,float32}` | Model precision (default: `float16`) |
| `--attn_implementation {eager,sdpa,flash_attention_2,flash_attention_3}` | Attention backend override |
| `--temperature` | Decode sampling temperature (default: `0.0` = greedy) |
| `--topk_true_rank` | How many top-k neighbours to search for the true token's rank (default: `10`) |
| `--max_prompts` | Limit number of prompts to process |
| `--anchor_token_id` | Override anchor token for `build_bank.py` (default: BOS token) |
| `--skip_save_tensors` / `--skip_save_faiss` | Omit `.pt` or `.faiss` output in `build_bank.py` |

## Notes

- For cosine similarity, use both `--normalize_bank` and `--normalize_queries`.
- With `--use_faiss`, probing uses the saved `.faiss` files when present.
- If only raw `.pt` bank tensors exist, the probe scripts can still build in-memory FAISS indices.
- Attention summaries are only available for layers >= 1 (standard models) or full-attention layers (hybrid models).
- Template-control detection is best-effort and assumes message content appears verbatim in the rendered chat template.

## Output schema

Each JSONL row contains:

| Field | Type | Description |
|-------|------|-------------|
| `prompt_index` | int | Index of the prompt in the input file |
| `phase` | str | `"prefill"` or `"decode"` |
| `position_in_phase` | int | Token position within phase |
| `global_position` | int | Token position in the full sequence |
| `token_id` | int | True token id |
| `token_text` | str | Decoded token text |
| `is_special_token` | bool | Whether token is a special token |
| `is_template_control_token` | bool | Whether token is a chat-template control token |
| `layer` | int | Hidden-state layer index |
| `nn_token_id` | int | Nearest-neighbor token id from the bank |
| `nn_token_text` | str | Decoded nearest-neighbor token text |
| `disentangled` | bool | `True` if `nn_token_id == token_id` |
| `score` | float | Similarity (cosine) or negative L2 distance |
| `rank_of_true` | int or null | Rank of true token in top-k, if found |
| `attn_entropy` | float or null | Shannon entropy of attention weights |
| `attn_top1_mass` | float or null | Mass on the highest-attention position |
| `attn_top5_mass` | float or null | Mass on top 5 positions |
| `attn_recent_8_mass` | float or null | Mass on last 8 positions |
| `attn_recent_32_mass` | float or null | Mass on last 32 positions |
| `attn_bos_mass` | float or null | Mass on BOS position |
| `attn_prev_mass` | float or null | Mass on immediately previous position |
| `attn_mean_distance` | float or null | Mean distance-weighted attention position |

Attention fields are present only when `--collect_attention_stats` is set and attention is available for the given layer.
