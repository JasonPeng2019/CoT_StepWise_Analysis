from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from typing import Any


JsonDict = dict[str, Any]


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _serialize(v) for k, v in asdict(value).items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


def _deserialize_field(field_type: Any, value: Any) -> Any:
    origin = getattr(field_type, "__origin__", None)
    args = getattr(field_type, "__args__", ())
    if value is None:
        return None
    if origin is list and args:
        return [_deserialize_field(args[0], item) for item in value]
    if origin is dict and len(args) == 2:
        return {
            _deserialize_field(args[0], key): _deserialize_field(args[1], item)
            for key, item in value.items()
        }
    if origin is tuple and args:
        return tuple(_deserialize_field(args[0], item) for item in value)
    if getattr(field_type, "__module__", "") == "typing" and args:
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            return _deserialize_field(non_none[0], value)
    if isinstance(field_type, type) and issubclass(field_type, Record):
        return field_type.from_dict(value)
    return value


@dataclass(slots=True)
class Record:
    def to_dict(self) -> JsonDict:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "Record":
        kwargs = {}
        for item in fields(cls):
            if item.name not in payload:
                continue
            kwargs[item.name] = _deserialize_field(item.type, payload[item.name])
        return cls(**kwargs)


@dataclass(slots=True)
class RawRollout(Record):
    cond: str
    answer: str | None
    parse_ok: bool
    filter_sim: float | None = None


@dataclass(slots=True)
class Problem(Record):
    problem_id: str
    dataset: str
    phase: int
    prompt: str
    gold_answer: str
    vocabulary: dict[str, list[str]] = field(default_factory=dict)
    supporting_facts: list[list[int]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Trace(Record):
    trace_id: str
    problem_id: str
    token_ids: list[int]
    text: str
    gen_seed: int
    gen_config_hash: str


@dataclass(slots=True)
class Span(Record):
    span_id: str
    trace_id: str
    clause_id: str
    tok_start: int
    tok_end: int
    text: str
    is_window: bool
    task_tokens: list[int] = field(default_factory=list)
    frame_tokens: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Signals(Record):
    span_id: str
    E: float | None = None
    P: float | None = None
    Pchained: float | None = None
    G: float | None = None
    Pprime: float | None = None
    PprimeChained: float | None = None
    D: float | None = None
    Dchained: float | None = None
    D_ig: float | None = None
    c: float | None = None
    position: int | None = None
    length: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Label(Record):
    span_id: str
    A: float
    real_dist: dict[str, float]
    diff_dist: dict[str, float]
    raw_rollouts: list[RawRollout]
    R_real: int
    R_diff_net: int
    R_gross: int
    parse_fail_rate: float
    metric: str


@dataclass(slots=True)
class ConstructReport(Record):
    span_id: str
    diff_shift: float | None
    similar_shift: float | None
    has_task_tokens: bool


@dataclass(slots=True)
class SplitAssignment(Record):
    problem_id: str
    fold: int
    role: str


@dataclass(slots=True)
class ModelFacts(Record):
    L: int
    H: int
    Hkv: int
    group_size: int
    d: int
    d_head: int
    eps: float
    norm_type: str
    rope_theta: float | None
    model_type: str
    attn_impl: str


@dataclass(slots=True)
class RunManifest(Record):
    git_sha: str
    config_hash: str
    lockfile_hash: str
    model_facts: ModelFacts | None
    seeds: dict[str, int]
    phase: int
    tier: str
    started_at: str


def dump_records(records: list[Record]) -> list[JsonDict]:
    return [record.to_dict() for record in records]


def load_records(record_type: type[Record], payloads: list[JsonDict]) -> list[Record]:
    return [record_type.from_dict(payload) for payload in payloads]
