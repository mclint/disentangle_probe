from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class PromptExample:
    raw_text: str
    input_ids: Optional[List[int]] = None
    attention_mask: Optional[List[int]] = None
    content_token_mask: Optional[List[bool]] = None
    template_control_token_mask: Optional[List[bool]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProbeRecord:
    prompt_index: int
    phase: str
    position_in_phase: int
    global_position: int
    token_id: int
    token_text: str
    is_special_token: bool
    is_template_control_token: bool
    layer: int
    nn_token_id: int
    nn_token_text: str
    disentangled: bool
    score: float
    rank_of_true: Optional[int] = None
    attn_entropy: Optional[float] = None
    attn_top1_mass: Optional[float] = None
    attn_top5_mass: Optional[float] = None
    attn_recent_8_mass: Optional[float] = None
    attn_recent_32_mass: Optional[float] = None
    attn_bos_mass: Optional[float] = None
    attn_prev_mass: Optional[float] = None
    attn_mean_distance: Optional[float] = None


def get_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def jsonl_write(path: str, records: Iterable[ProbeRecord]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model_kwargs = {
        "torch_dtype": get_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }

    # Put the model directly on the requested device.
    # For a single-GPU run, this is cleaner than loading on CPU then moving.
    if args.device.startswith("cuda"):
        model_kwargs["device_map"] = {"": args.device}
    else:
        model_kwargs["device_map"] = {"": "cpu"}

    # SDPA does not support output_attentions=True, so switch to eager
    # whenever we plan to collect attention statistics.
    attn_impl = getattr(args, "attn_implementation", None)
    collect_attention_stats = getattr(args, "collect_attention_stats", None)

    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl
    elif collect_attention_stats:
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs,
    )

    model.eval()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    return model, tokenizer


def hidden_size_from_model(model) -> int:
    if hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)
    if hasattr(model.config, "n_embd"):
        return int(model.config.n_embd)
    raise ValueError("Could not infer hidden size from model config.")


def num_hidden_state_slots(model) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers) + 1
    if hasattr(model.config, "n_layer"):
        return int(model.config.n_layer) + 1
    raise ValueError("Could not infer number of hidden state slots.")


def validate_layers(requested_layers: Sequence[int], model) -> None:
    total = num_hidden_state_slots(model)
    bad = [l for l in requested_layers if l < 0 or l >= total]
    if bad:
        raise ValueError(f"Invalid layers {bad}; valid range is [0, {total - 1}].")


def batch_iter(xs: Sequence[int], batch_size: int):
    for i in range(0, len(xs), batch_size):
        yield xs[i : i + batch_size]


def decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.convert_ids_to_tokens(int(token_id))
    except Exception:
        return str(token_id)


def is_special_token_id(tokenizer, token_id: int) -> bool:
    return int(token_id) in set(getattr(tokenizer, "all_special_ids", []) or [])


def maybe_normalize(x: torch.Tensor, normalize: bool) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1) if normalize else x


def to_numpy_f32(x: torch.Tensor) -> np.ndarray:
    return x.detach().to(torch.float32).cpu().contiguous().numpy().astype("float32", copy=False)


class FaissLayerIndex:
    def __init__(self, index, use_cosine: bool):
        self.index = index
        self.use_cosine = use_cosine

    def search(self, query: torch.Tensor, true_token_id: int, topk_true_rank: int) -> Tuple[int, float, Optional[int]]:
        q = to_numpy_f32(query).reshape(1, -1)
        scores, ids = self.index.search(q, max(1, topk_true_rank))
        nn_id = int(ids[0, 0])
        score = float(scores[0, 0])
        rank_true = None
        matches = np.where(ids[0] == int(true_token_id))[0]
        if len(matches) > 0:
            rank_true = int(matches[0]) + 1
        return nn_id, score, rank_true


def build_faiss_index(vectors: np.ndarray, use_cosine: bool):
    import faiss

    dim = int(vectors.shape[1])
    if use_cosine:
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def knn_top1_and_true_rank(
    query: torch.Tensor,
    bank: torch.Tensor,
    true_token_id: int,
    use_cosine: bool,
    topk_true_rank: int,
) -> Tuple[int, float, Optional[int]]:
    if use_cosine:
        sims = bank @ query
        top_scores, top_ids = torch.topk(sims, k=max(1, topk_true_rank), dim=0)
        nn_id = int(top_ids[0].item())
        score = float(top_scores[0].item())
    else:
        dists = torch.sum((bank - query.unsqueeze(0)) ** 2, dim=-1)
        top_scores, top_ids = torch.topk(-dists, k=max(1, topk_true_rank), dim=0)
        nn_id = int(top_ids[0].item())
        score = float(top_scores[0].item())

    rank_true = None
    matches = (top_ids == int(true_token_id)).nonzero(as_tuple=False)
    if matches.numel() > 0:
        rank_true = int(matches[0, 0].item()) + 1
    return nn_id, score, rank_true


def _find_all_occurrences(haystack: str, needle: str) -> List[Tuple[int, int]]:
    if not needle:
        return []
    spans: List[Tuple[int, int]] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(needle)))
        start = idx + len(needle)
    return spans


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(int(s), int(e)) for s, e in merged]


def _char_in_any_span(start: int, end: int, spans: Sequence[Tuple[int, int]]) -> bool:
    for s, e in spans:
        if end > s and start < e:
            return True
    return False


def _build_content_mask_from_offsets(rendered_text: str, offsets: Sequence[Tuple[int, int]], messages: Sequence[Dict[str, Any]]) -> List[bool]:
    content_spans: List[Tuple[int, int]] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
            content = "".join(parts)
        if not isinstance(content, str) or not content:
            continue
        content_spans.extend(_find_all_occurrences(rendered_text, content))
    content_spans = _merge_spans(content_spans)
    mask: List[bool] = []
    for start, end in offsets:
        if start == end:
            mask.append(False)
        else:
            mask.append(_char_in_any_span(start, end, content_spans))
    return mask


def _tokenize_with_offsets(tokenizer, text: str):
    out = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = list(out["input_ids"])
    offsets = [tuple(x) for x in out["offset_mapping"]]
    return input_ids, offsets


def parse_chat_messages(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict) and "messages" in obj:
        obj = obj["messages"]
    if not isinstance(obj, list):
        raise ValueError("Chat input must be a list of messages or an object with a 'messages' field.")
    normalized: List[Dict[str, Any]] = []
    for msg in obj:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise ValueError("Each message must be a dict with 'role' and 'content'.")
        normalized.append({"role": msg["role"], "content": msg["content"]})
    return normalized


def prepare_prompt_examples(
    prompts_file: str,
    tokenizer,
    prompt_format: str,
    max_prompts: Optional[int],
    max_prompt_tokens: int,
    add_generation_prompt: bool,
) -> List[PromptExample]:
    examples: List[PromptExample] = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if max_prompts is not None:
        lines = lines[:max_prompts]

    if prompt_format == "raw":
        for line in lines:
            enc = tokenizer(
                line,
                return_tensors=None,
                truncation=True,
                max_length=max_prompt_tokens,
                add_special_tokens=True,
            )
            ids = list(enc["input_ids"])
            examples.append(PromptExample(
                raw_text=line,
                input_ids=ids,
                attention_mask=list(enc.get("attention_mask", [1] * len(ids))),
                content_token_mask=[True] * len(ids),
                template_control_token_mask=[False] * len(ids),
                metadata={"prompt_format": "raw"},
            ))
        return examples

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Selected prompt_format='chat' but tokenizer has no apply_chat_template().")

    for line in lines:
        obj = json.loads(line)
        messages = parse_chat_messages(obj)
        rendered_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        input_ids, offsets = _tokenize_with_offsets(tokenizer, rendered_text)
        if len(input_ids) > max_prompt_tokens:
            input_ids = input_ids[:max_prompt_tokens]
            offsets = offsets[:max_prompt_tokens]
        content_mask = _build_content_mask_from_offsets(rendered_text, offsets, messages)
        content_mask = content_mask[: len(input_ids)]
        template_mask = [not x for x in content_mask]
        examples.append(PromptExample(
            raw_text=rendered_text,
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            content_token_mask=content_mask,
            template_control_token_mask=template_mask,
            metadata={"prompt_format": "chat", "messages": messages},
        ))
    return examples


def print_summary(records: Sequence[ProbeRecord]) -> None:
    if not records:
        print("No records.")
        return
    grouped: Dict[Tuple[str, int, bool, bool], List[ProbeRecord]] = {}
    for r in records:
        grouped.setdefault((r.phase, r.layer, r.is_special_token, r.is_template_control_token), []).append(r)
    print("\n=== Summary ===")
    for (phase, layer, is_special, is_template), rs in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        n = len(rs)
        disent = sum(int(r.disentangled) for r in rs)
        mean_score = sum(r.score for r in rs) / max(1, n)
        tok_label = "special" if is_special else "non_special"
        template_label = "template" if is_template else "content_or_decode"
        print(
            f"phase={phase:7s} layer={layer:3d} tokens={tok_label:11s} source={template_label:17s} "
            f"n={n:6d} disentangled_rate={disent/max(1,n):.4f} mean_score={mean_score:.4f}"
        )
