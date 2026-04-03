from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tensordict import MemoryMappedTensor
except ImportError:  # pragma: no cover - exercised indirectly in environments without tensordict
    MemoryMappedTensor = None


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


@dataclass(frozen=True)
class TokenDistributionSpec:
    name: str
    include_content_tokens: bool
    include_template_control_tokens: bool
    include_special_tokens: bool = False

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "TokenDistributionSpec":
        if not isinstance(obj, dict):
            raise ValueError("Each distribution entry must be a JSON object.")
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Each distribution must have a non-empty string 'name'.")
        if not all(ch.isalnum() or ch in "._-" for ch in name):
            raise ValueError(
                f"Distribution name {name!r} contains unsupported characters. "
                "Use letters, numbers, '.', '_' or '-'."
            )

        include_content_tokens = bool(obj.get("include_content_tokens", False))
        include_template_control_tokens = bool(obj.get("include_template_control_tokens", False))
        include_special_tokens = bool(obj.get("include_special_tokens", False))

        if not (include_content_tokens or include_template_control_tokens):
            raise ValueError(
                f"Distribution {name!r} must enable at least one token source: "
                "'include_content_tokens' or 'include_template_control_tokens'."
            )

        return cls(
            name=name,
            include_content_tokens=include_content_tokens,
            include_template_control_tokens=include_template_control_tokens,
            include_special_tokens=include_special_tokens,
        )

    def matches(
        self,
        *,
        is_content_token: bool,
        is_template_control_token: bool,
        is_special_token: bool,
    ) -> bool:
        if is_special_token and not self.include_special_tokens:
            return False
        return (
            (self.include_content_tokens and is_content_token)
            or (self.include_template_control_tokens and is_template_control_token)
        )


T = TypeVar("T")


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


def batch_iter(xs: Sequence[T], batch_size: int):
    for i in range(0, len(xs), batch_size):
        yield xs[i : i + batch_size]


def decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.convert_ids_to_tokens(int(token_id))
    except Exception:
        return str(token_id)


def is_special_token_id(tokenizer, token_id: int) -> bool:
    return int(token_id) in set(getattr(tokenizer, "all_special_ids", []) or [])


def load_distribution_specs(path: str | Path) -> List[TokenDistributionSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Distribution config must be a JSON object.")
    raw_distributions = payload.get("distributions")
    if not isinstance(raw_distributions, list) or not raw_distributions:
        raise ValueError("Distribution config must contain a non-empty 'distributions' list.")

    specs = [TokenDistributionSpec.from_dict(obj) for obj in raw_distributions]
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate distribution names found: {duplicates}")
    return specs


def build_distribution_token_mask(
    example: PromptExample,
    tokenizer,
    distribution: TokenDistributionSpec,
) -> List[bool]:
    input_ids = list(example.input_ids or [])
    content_mask = list(example.content_token_mask or [])
    template_mask = list(example.template_control_token_mask or [])
    mask: List[bool] = []
    for pos, token_id in enumerate(input_ids):
        is_content_token = bool(content_mask[pos]) if pos < len(content_mask) else False
        is_template_control_token = bool(template_mask[pos]) if pos < len(template_mask) else False
        mask.append(distribution.matches(
            is_content_token=is_content_token,
            is_template_control_token=is_template_control_token,
            is_special_token=is_special_token_id(tokenizer, token_id),
        ))
    return mask


def maybe_normalize(x: torch.Tensor, normalize: bool) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1) if normalize else x


def to_numpy_f32(x: torch.Tensor) -> np.ndarray:
    return x.detach().to(torch.float32).cpu().contiguous().numpy().astype("float32", copy=False)


def dtype_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


class RunningMoments:
    def __init__(self, hidden_size: int):
        self.count = 0
        self.mean = torch.zeros(hidden_size, dtype=torch.float32)
        self.m2 = torch.zeros(hidden_size, dtype=torch.float32)

    def update(self, vector: torch.Tensor) -> None:
        x = vector.detach().to(torch.float32).cpu()
        self.count += 1
        delta = x - self.mean
        self.mean += delta / float(self.count)
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def variance(self) -> torch.Tensor:
        if self.count <= 0:
            return torch.zeros_like(self.mean)
        return self.m2 / float(self.count)


class SparseTokenStatsAccumulator:
    def __init__(self, hidden_size: int):
        self.hidden_size = int(hidden_size)
        self._states: Dict[int, RunningMoments] = {}

    def update(self, token_id: int, vector: torch.Tensor) -> None:
        token_id = int(token_id)
        state = self._states.get(token_id)
        if state is None:
            state = RunningMoments(self.hidden_size)
            self._states[token_id] = state
        state.update(vector)

    def get_state(self, token_id: int) -> Optional[RunningMoments]:
        return self._states.get(int(token_id))

    def token_ids(self) -> List[int]:
        return sorted(self._states)

    def finalize(self, storage_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        token_ids = self.token_ids()
        if not token_ids:
            return {
                "token_ids": torch.empty(0, dtype=torch.long),
                "counts": torch.empty(0, dtype=torch.long),
                "mean": torch.empty((0, self.hidden_size), dtype=storage_dtype),
                "variance": torch.empty((0, self.hidden_size), dtype=storage_dtype),
            }

        counts = torch.tensor([self._states[token_id].count for token_id in token_ids], dtype=torch.long)
        mean = torch.stack([self._states[token_id].mean for token_id in token_ids], dim=0).to(storage_dtype)
        variance = torch.stack([self._states[token_id].variance() for token_id in token_ids], dim=0).to(storage_dtype)
        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "counts": counts,
            "mean": mean,
            "variance": variance,
        }


class TokenCodebookState:
    def __init__(self, hidden_size: int, k: int):
        self.hidden_size = int(hidden_size)
        self.k = int(k)
        self.active_clusters = 0
        self.cluster_counts = torch.zeros(self.k, dtype=torch.long)
        self.centroids = torch.zeros((self.k, self.hidden_size), dtype=torch.float32)
        self.m2 = torch.zeros((self.k, self.hidden_size), dtype=torch.float32)

    def update(self, vector: torch.Tensor) -> None:
        x = vector.detach().to(torch.float32).cpu()
        if self.active_clusters == 0:
            self._assign_new_cluster(0, x)
            self.active_clusters = 1
            return

        active_centroids = self.centroids[: self.active_clusters]
        distances = torch.sum((active_centroids - x.unsqueeze(0)) ** 2, dim=-1)
        nearest_idx = int(torch.argmin(distances).item())

        if self.active_clusters < self.k and float(distances[nearest_idx].item()) > 1e-12:
            new_idx = self.active_clusters
            self._assign_new_cluster(new_idx, x)
            self.active_clusters += 1
            return

        self._update_cluster(nearest_idx, x)

    def _assign_new_cluster(self, cluster_idx: int, vector: torch.Tensor) -> None:
        self.cluster_counts[cluster_idx] = 1
        self.centroids[cluster_idx] = vector
        self.m2[cluster_idx].zero_()

    def _update_cluster(self, cluster_idx: int, vector: torch.Tensor) -> None:
        prev_count = int(self.cluster_counts[cluster_idx].item())
        if prev_count == 0:
            self._assign_new_cluster(cluster_idx, vector)
            return

        next_count = prev_count + 1
        centroid = self.centroids[cluster_idx]
        delta = vector - centroid
        centroid = centroid + (delta / float(next_count))
        delta2 = vector - centroid
        self.centroids[cluster_idx] = centroid
        self.m2[cluster_idx] += delta * delta2
        self.cluster_counts[cluster_idx] = next_count

    def variances(self) -> torch.Tensor:
        variances = torch.zeros_like(self.centroids)
        for idx in range(self.active_clusters):
            count = int(self.cluster_counts[idx].item())
            if count > 0:
                variances[idx] = self.m2[idx] / float(count)
        return variances


class SparseTokenCodebookAccumulator:
    def __init__(self, hidden_size: int, k: int):
        self.hidden_size = int(hidden_size)
        self.k = int(k)
        self._states: Dict[int, TokenCodebookState] = {}

    def update(self, token_id: int, vector: torch.Tensor) -> None:
        token_id = int(token_id)
        state = self._states.get(token_id)
        if state is None:
            state = TokenCodebookState(self.hidden_size, self.k)
            self._states[token_id] = state
        state.update(vector)

    def finalize(
        self,
        storage_dtype: torch.dtype,
        stats_accumulator: SparseTokenStatsAccumulator,
        min_count: int,
    ) -> Dict[str, torch.Tensor]:
        token_ids = stats_accumulator.token_ids()
        if not token_ids:
            return {
                "token_ids": torch.empty(0, dtype=torch.long),
                "active_clusters": torch.empty(0, dtype=torch.long),
                "cluster_counts": torch.empty((0, self.k), dtype=torch.long),
                "centroids": torch.empty((0, self.k, self.hidden_size), dtype=storage_dtype),
                "cluster_variance": torch.empty((0, self.k, self.hidden_size), dtype=storage_dtype),
            }

        active_clusters = torch.zeros(len(token_ids), dtype=torch.long)
        cluster_counts = torch.zeros((len(token_ids), self.k), dtype=torch.long)
        centroids = torch.zeros((len(token_ids), self.k, self.hidden_size), dtype=torch.float32)
        cluster_variance = torch.zeros((len(token_ids), self.k, self.hidden_size), dtype=torch.float32)

        for row, token_id in enumerate(token_ids):
            stats_state = stats_accumulator.get_state(token_id)
            codebook_state = self._states.get(token_id)
            if stats_state is None:
                continue

            total_count = int(stats_state.count)
            if total_count < int(min_count) or codebook_state is None or codebook_state.active_clusters == 0:
                active_clusters[row] = 1 if total_count > 0 else 0
                cluster_counts[row, 0] = total_count
                centroids[row, 0] = stats_state.mean
                cluster_variance[row, 0] = stats_state.variance()
                continue

            active_clusters[row] = int(codebook_state.active_clusters)
            cluster_counts[row] = codebook_state.cluster_counts
            centroids[row] = codebook_state.centroids
            cluster_variance[row] = codebook_state.variances()

        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "active_clusters": active_clusters,
            "cluster_counts": cluster_counts,
            "centroids": centroids.to(storage_dtype),
            "cluster_variance": cluster_variance.to(storage_dtype),
        }


def bank_memmap_path(bank_dir: str | Path, layer: int) -> Path:
    return Path(bank_dir) / f"layer_{layer}.memmap"


def bank_npy_path(bank_dir: str | Path, layer: int) -> Path:
    return Path(bank_dir) / f"layer_{layer}.npy"


def bank_pt_path(bank_dir: str | Path, layer: int) -> Path:
    return Path(bank_dir) / f"layer_{layer}.pt"


def bank_metadata_path(bank_dir: str | Path) -> Path:
    return Path(bank_dir) / "bank_meta.json"


def save_bank_metadata(
    bank_dir: str | Path,
    vocab_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    layers: Sequence[int],
) -> None:
    payload = {
        "shape": [int(vocab_size), int(hidden_size)],
        "dtype": dtype_name(dtype),
        "layers": [int(layer) for layer in layers],
    }
    bank_metadata_path(bank_dir).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_bank_metadata(bank_dir: str | Path) -> Dict[str, Any]:
    path = bank_metadata_path(bank_dir)
    if not path.exists():
        raise FileNotFoundError(f"Activation bank metadata not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_bank_path(bank_dir: str | Path, layer: int) -> Path:
    memmap_path = bank_memmap_path(bank_dir, layer)
    if memmap_path.exists():
        return memmap_path
    npy_path = bank_npy_path(bank_dir, layer)
    if npy_path.exists():
        return npy_path
    pt_path = bank_pt_path(bank_dir, layer)
    if pt_path.exists():
        return pt_path
    return memmap_path


def load_bank_tensor(bank_dir: str | Path, layer: int, device: str) -> torch.Tensor:
    path = resolve_bank_path(bank_dir, layer)
    if not path.exists():
        raise FileNotFoundError(f"Activation bank for layer {layer} not found under {bank_dir}")
    if path.suffix == ".memmap":
        if MemoryMappedTensor is None:
            raise ImportError("tensordict is required to read .memmap activation banks.")
        metadata = load_bank_metadata(bank_dir)
        tensor = MemoryMappedTensor.from_filename(
            path,
            dtype=get_dtype(metadata["dtype"]),
            shape=torch.Size(metadata["shape"]),
        )
        return tensor.to(device)
    if path.suffix == ".npy":
        array = np.load(path)
        return torch.from_numpy(array).to(device)
    return torch.load(path, map_location=device)


def load_bank_vectors_for_faiss(bank_dir: str | Path, layer: int) -> np.ndarray | torch.Tensor:
    path = resolve_bank_path(bank_dir, layer)
    if not path.exists():
        raise FileNotFoundError(f"Activation bank for layer {layer} not found under {bank_dir}")
    if path.suffix == ".memmap":
        if MemoryMappedTensor is None:
            raise ImportError("tensordict is required to read .memmap activation banks.")
        metadata = load_bank_metadata(bank_dir)
        return MemoryMappedTensor.from_filename(
            path,
            dtype=get_dtype(metadata["dtype"]),
            shape=torch.Size(metadata["shape"]),
        )
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    tensor = torch.load(path, map_location="cpu")
    return to_numpy_f32(tensor)


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


def build_faiss_index(vectors: np.ndarray | torch.Tensor, use_cosine: bool, add_batch_size: int = 16384):
    import faiss

    dim = int(vectors.shape[1])
    if use_cosine:
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    total = int(vectors.shape[0])
    for start in range(0, total, add_batch_size):
        batch_vectors = vectors[start : start + add_batch_size]
        if torch.is_tensor(batch_vectors):
            batch = to_numpy_f32(batch_vectors)
        else:
            batch = np.asarray(batch_vectors, dtype="float32")
        index.add(batch)
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


def prepare_chat_prompt_example(
    obj: Any,
    tokenizer,
    max_prompt_tokens: int,
    add_generation_prompt: bool,
    metadata: Optional[Dict[str, Any]] = None,
) -> PromptExample:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Selected chat prompt processing but tokenizer has no apply_chat_template().")

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
    example_metadata = {"prompt_format": "chat", "messages": messages}
    if metadata:
        example_metadata.update(metadata)
    return PromptExample(
        raw_text=rendered_text,
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        content_token_mask=content_mask,
        template_control_token_mask=template_mask,
        metadata=example_metadata,
    )


def prepare_prompt_examples_from_chat_objects(
    objects: Sequence[Any],
    tokenizer,
    max_prompts: Optional[int],
    max_prompt_tokens: int,
    add_generation_prompt: bool,
    source_metadata: Optional[Dict[str, Any]] = None,
) -> List[PromptExample]:
    examples: List[PromptExample] = []
    objs = list(objects)
    if max_prompts is not None:
        objs = objs[:max_prompts]
    for idx, obj in enumerate(objs):
        metadata = dict(source_metadata or {})
        metadata["source_index"] = int(idx)
        examples.append(prepare_chat_prompt_example(
            obj=obj,
            tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            add_generation_prompt=add_generation_prompt,
            metadata=metadata,
        ))
    return examples


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

    objects = [json.loads(line) for line in lines]
    return prepare_prompt_examples_from_chat_objects(
        objects=objects,
        tokenizer=tokenizer,
        max_prompts=None,
        max_prompt_tokens=max_prompt_tokens,
        add_generation_prompt=add_generation_prompt,
    )


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
