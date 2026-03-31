# Disentanglement Probe Package

This package builds a layer-wise vocabulary activation bank for a Hugging Face causal LM and then probes prompt-prefill and decode states against that bank.

## Files

- `build_bank.py`: builds the activation bank and optional FAISS indices
- `probe_states.py`: probes prefill/decode hidden states and writes JSONL results
- `common.py`: shared helpers, prompt preparation, and FAISS utilities
- `analyze_results.ipynb`: starter notebook for analysis and plots

## Layer convention

We follow Hugging Face indexing exactly:

- `hidden_states[0]` = embeddings
- `hidden_states[1]` = output of transformer block 0
- `hidden_states[2]` = output of transformer block 1
- ...

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
pip install torch transformers faiss-cpu jupyter plotly matplotlib pandas
```

## Device loading

The model loader tries to place the model directly on the configured device during `from_pretrained(...)` using `device_map={"": <device>}`. If that is not supported in your setup, it falls back to a normal load followed by `model.to(device)`.

## Build the activation bank

```bash
python build_bank.py   --model Qwen/Qwen2.5-1.5B   --layers 0 1 8 16 24   --bank_dir ./bank_qwen25_15b   --vocab_batch_size 256   --normalize_bank   --device cuda:0
```

This writes:

- `layer_<L>.pt` raw bank tensors
- `layer_<L>.faiss` FAISS indices

## Probe raw-text prompts

```bash
python probe_states.py   --model Qwen/Qwen2.5-1.5B   --prompts_file prompts.txt   --bank_dir ./bank_qwen25_15b   --out_jsonl ./probe_results.jsonl   --layers 0 1 8 16 24   --prompt_format raw   --max_prompt_tokens 256   --max_new_tokens 32   --normalize_bank   --normalize_queries   --use_faiss   --collect_attention_stats   --device cuda:0
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
python probe_states.py   --model Qwen/Qwen2.5-1.5B-Instruct   --prompts_file chat_prompts.jsonl   --bank_dir ./bank_qwen25_15b   --out_jsonl ./probe_results_chat.jsonl   --layers 0 1 8 16 24   --prompt_format chat   --add_generation_prompt   --max_prompt_tokens 512   --max_new_tokens 64   --normalize_bank   --normalize_queries   --use_faiss   --collect_attention_stats   --device cuda:0
```

Template-control tokens are identified by rendering the chat template to text, tokenizing with offset mappings, and marking tokens that fall outside user-provided message content spans.

## Notes

- For cosine similarity, use both `--normalize_bank` and `--normalize_queries`.
- With `--use_faiss`, probing uses the saved `.faiss` files when present.
- If only raw `.pt` bank tensors exist, `probe_states.py` can still build in-memory FAISS indices.
- Attention summaries are only available for layers `>= 1`.
- Template-control detection is best-effort and assumes message content appears verbatim in the rendered chat template.

## Output schema

Each JSONL row contains:

- prompt index
- phase (`prefill` or `decode`)
- position in phase
- global position
- token id / token text
- `is_special_token`
- `is_template_control_token`
- layer
- nearest-neighbor token id / text
- `disentangled`
- similarity or distance-based score
- optional rank of true token within top-k
- optional attention summary statistics
