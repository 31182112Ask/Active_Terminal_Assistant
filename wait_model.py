from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


BUCKET_SPECS = [
    {"label": "0s-5s", "upper": 5.0, "representative": 3.0},
    {"label": "5s-10s", "upper": 10.0, "representative": 8.0},
    {"label": "10s-20s", "upper": 20.0, "representative": 15.0},
    {"label": "20s-30s", "upper": 30.0, "representative": 25.0},
    {"label": "30s-45s", "upper": 45.0, "representative": 38.0},
    {"label": "45s-60s", "upper": 60.0, "representative": 53.0},
    {"label": "1m-2m", "upper": 120.0, "representative": 90.0},
    {"label": "2m-3m", "upper": 180.0, "representative": 150.0},
    {"label": "3m-5m", "upper": 300.0, "representative": 240.0},
    {"label": "5m-7m", "upper": 420.0, "representative": 360.0},
    {"label": "7m-10m", "upper": 600.0, "representative": 510.0},
    {"label": "10m-15m", "upper": 900.0, "representative": 750.0},
    {"label": "15m-20m", "upper": 1_200.0, "representative": 1_050.0},
    {"label": "20m-30m", "upper": 1_800.0, "representative": 1_500.0},
    {"label": "30m-45m", "upper": 2_700.0, "representative": 2_250.0},
    {"label": "45m-1h", "upper": 3_600.0, "representative": 3_150.0},
    {"label": "1h-1.5h", "upper": 5_400.0, "representative": 4_500.0},
    {"label": "1.5h-2h", "upper": 7_200.0, "representative": 6_300.0},
    {"label": "2h-3h", "upper": 10_800.0, "representative": 9_000.0},
    {"label": "3h-4h", "upper": 14_400.0, "representative": 12_600.0},
    {"label": "4h-6h", "upper": 21_600.0, "representative": 18_000.0},
    {"label": "6h-8h", "upper": 28_800.0, "representative": 25_200.0},
    {"label": "8h-12h", "upper": 43_200.0, "representative": 36_000.0},
    {"label": "12h-18h", "upper": 64_800.0, "representative": 54_000.0},
    {"label": "18h-24h", "upper": 86_400.0, "representative": 75_600.0},
    {"label": "1d-2d", "upper": 172_800.0, "representative": 129_600.0},
    {"label": "2d-3d", "upper": 259_200.0, "representative": 216_000.0},
    {"label": "3d-4d", "upper": 345_600.0, "representative": 302_400.0},
    {"label": "4d-6d", "upper": 518_400.0, "representative": 432_000.0},
    {"label": "6d-8d", "upper": 691_200.0, "representative": 604_800.0},
    {"label": "8d-10d", "upper": 864_000.0, "representative": 777_600.0},
]
MAX_WAIT_SECONDS = BUCKET_SPECS[-1]["upper"]
SUPPRESS_BUCKET = len(BUCKET_SPECS)
NUM_BUCKETS = SUPPRESS_BUCKET + 1
SUPPRESS_LABEL = "suppress"


TOKEN_PATTERN = re.compile(r"[a-z0-9_']+|[\u4e00-\u9fff]+", flags=re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
RELATIVE_WAIT_PATTERN = re.compile(
    r"(?P<value>\d+|[零一二两三四五六七八九十]+)\s*(?P<unit>seconds?|secs?|minutes?|mins?|hours?|days?|秒|分钟|分|小時|小时|天)(?:\s*later|\s*from now|\s*后)?",
    flags=re.IGNORECASE,
)

PATTERN_STRINGS = {
    "question": r"\?|吗|呢|么|請問|请问",
    "reminder": r"remind|follow up|follow-up|check back|circle back|nudge|ping|poke|check in|提醒|跟进|回头找|再问|再找|再戳|催我",
    "busy": r"busy|in meetings|in a meeting|heads down|offline|sleep|sleeping|commute|class|deep work|忙|开会|会议|睡|离线|上课|通勤|埋头",
    "do_not_contact": r"don't ping|do not ping|don't follow up|leave it here|stay quiet|no need to follow up|别再提醒|先别找我|不要再跟进|不用再问|先安静|先别聊",
    "closure": r"that's all|all good|we're done|resolved|solved|nothing else|thanks,? that's enough|先这样|没事了|搞定了|就这样|不用了|谢谢够了",
    "urgent": r"urgent|asap|prod|production|sev[- ]?1|incident|outage|down|blocked|紧急|马上|宕机|故障|阻塞|线上",
    "deliverable": r"draft|doc|document|file|deck|slides|proposal|brief|ticket|patch|report|草稿|文档|文件|方案|汇报|补丁|报告",
    "time_tonight": r"tonight|this evening|later tonight|今晚|今天晚上|今晚再",
    "time_morning": r"tomorrow morning|morning|明早|明天早上|早上",
    "time_afternoon": r"afternoon|after lunch|tomorrow afternoon|下午|午后|午饭后|明天下午",
    "time_evening": r"evening|tonight|tomorrow evening|晚上|今晚|明晚",
    "time_relative": r"\b\d+\s*(minute|minutes|min|mins|hour|hours|day|days)\b|\d+\s*(分钟|小時|小时|天)后",
    "status": r"status|update|check if|let me know|回报|更新|进展|状态",
    "emotion": r"overwhelmed|anxious|need space|grief|drained|let me breathe|我想缓缓|先让我静一静|情绪|焦虑|难过",
    "thanks": r"thanks|thank you|appreciate it|多谢|谢谢|麻烦了",
}
PATTERNS = {name: re.compile(expr, flags=re.IGNORECASE) for name, expr in PATTERN_STRINGS.items()}


def normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text.strip().lower())


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_text(text))


def stable_hash(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def parse_number_token(token: str) -> float | None:
    if token.isdigit():
        return float(token)
    mapping = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if token in mapping:
        return float(mapping[token])
    if "十" in token:
        if token == "十":
            return 10.0
        if token.startswith("十"):
            return 10.0 + mapping.get(token[1:], 0)
        if token.endswith("十"):
            return mapping.get(token[0], 0) * 10.0
        left, right = token.split("十", 1)
        return mapping.get(left, 0) * 10.0 + mapping.get(right, 0)
    return None


def seconds_to_log_wait(wait_seconds: float) -> float:
    return math.log1p(max(0.0, min(wait_seconds, MAX_WAIT_SECONDS)))


def log_wait_to_seconds(log_wait: float) -> float:
    return max(0.0, min(math.expm1(log_wait), MAX_WAIT_SECONDS))


def wait_seconds_to_bucket(wait_seconds: float | None, suppress: bool = False) -> int:
    if suppress or wait_seconds is None:
        return SUPPRESS_BUCKET
    clipped = max(0.0, min(wait_seconds, MAX_WAIT_SECONDS))
    for bucket_id, spec in enumerate(BUCKET_SPECS):
        if clipped <= spec["upper"]:
            return bucket_id
    return len(BUCKET_SPECS) - 1


def bucket_to_bounds(bucket_id: int) -> tuple[float, float]:
    if bucket_id == SUPPRESS_BUCKET:
        return MAX_WAIT_SECONDS, MAX_WAIT_SECONDS
    lower = 0.0 if bucket_id == 0 else BUCKET_SPECS[bucket_id - 1]["upper"]
    upper = BUCKET_SPECS[bucket_id]["upper"]
    return lower, upper


def bucket_to_label(bucket_id: int) -> str:
    if bucket_id == SUPPRESS_BUCKET:
        return SUPPRESS_LABEL
    return BUCKET_SPECS[bucket_id]["label"]


def humanize_wait(wait_seconds: float | None) -> str:
    if wait_seconds is None:
        return "suppress"
    seconds = max(0.0, wait_seconds)
    if seconds < 90:
        return f"{int(round(seconds))}s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{int(round(minutes))}m"
    hours = minutes / 60.0
    if hours < 36:
        return f"{hours:.1f}h"
    days = hours / 24.0
    return f"{days:.1f}d"


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def next_daypart_wait_seconds(hour_local: int, weekday: int, slot: str) -> float:
    del weekday
    schedule = {
        "after_lunch": 14.0,
        "afternoon": 15.0,
        "evening": 19.5,
        "tonight": 20.0,
        "tomorrow_morning": 33.0,
        "tomorrow_afternoon": 39.0,
        "tomorrow_evening": 44.0,
        "next_business_morning": 57.0,
    }
    target = schedule[slot]
    now = float(hour_local)
    wait_hours = target - now
    while wait_hours <= 0.0:
        wait_hours += 24.0
    return wait_hours * 3_600.0


def build_relative_phrase(wait_seconds: float, lang: str, holdout: bool = False) -> str:
    seconds = int(round(wait_seconds))
    if seconds < 3_600:
        if lang == "zh":
            return f"{seconds}秒后" if not holdout else f"{seconds}秒之后"
        return f"in {seconds} seconds" if not holdout else f"{seconds} seconds from now"
    minutes = wait_seconds / 60.0
    if minutes < 360:
        mins = int(round(minutes))
        if lang == "zh":
            return f"{mins}分钟后" if not holdout else f"{mins}分钟之后"
        return f"in {mins} minutes" if not holdout else f"{mins} minutes from now"
    hours = minutes / 60.0
    if hours < 48:
        hrs = int(round(hours))
        if lang == "zh":
            return f"{hrs}小时后" if not holdout else f"大约{hrs}小时后"
        return f"in {hrs} hours" if not holdout else f"about {hrs} hours later"
    days = int(round(hours / 24.0))
    if lang == "zh":
        return f"{days}天后"
    return f"in {days} days"


def extract_relative_wait_seconds(text: str) -> float | None:
    match = RELATIVE_WAIT_PATTERN.search(normalize_text(text))
    if not match:
        return None
    parsed_value = parse_number_token(match.group("value"))
    if parsed_value is None:
        return None
    value = float(parsed_value)
    unit = match.group("unit")
    if unit.startswith(("second", "sec")) or unit == "秒":
        multiplier = 1.0
    elif unit.startswith(("minute", "min")) or unit in {"分钟", "分"}:
        multiplier = 60.0
    elif unit.startswith("hour") or unit in {"小時", "小时"}:
        multiplier = 3_600.0
    else:
        multiplier = 86_400.0
    return min(MAX_WAIT_SECONDS, value * multiplier)


def extract_time_hint_seconds(text: str, hour_local: int, weekday: int) -> float | None:
    normalized = normalize_text(text)
    relative_wait = extract_relative_wait_seconds(normalized)
    if relative_wait is not None:
        return relative_wait
    if re.search(r"tomorrow morning|early tomorrow|when tomorrow morning starts|明早|明天早上", normalized):
        return next_daypart_wait_seconds(hour_local, weekday, "tomorrow_morning")
    if re.search(r"tomorrow afternoon|by tomorrow afternoon|once tomorrow afternoon opens up|明天下午", normalized):
        return next_daypart_wait_seconds(hour_local, weekday, "tomorrow_afternoon")
    if re.search(r"tomorrow evening|tomorrow night|by tomorrow night|once tomorrow evening arrives|明晚|明天晚上", normalized):
        return next_daypart_wait_seconds(hour_local, weekday, "tomorrow_evening")
    if re.search(r"after lunch|once lunch is over|post-lunch|午饭后|午后", normalized):
        return next_daypart_wait_seconds(hour_local, weekday, "after_lunch")
    if re.search(r"this afternoon|later this afternoon|later in the afternoon|once the afternoon opens up|今天下午|下午", normalized):
        return next_daypart_wait_seconds(hour_local, weekday, "afternoon")
    if re.search(r"later tonight|this evening|once tonight settles down|今晚|今天晚上", normalized):
        return next_daypart_wait_seconds(hour_local, weekday, "tonight")
    if re.search(r"next business morning|next workday morning|next working morning|next business-day morning|下个工作日早上|下一个工作日上午", normalized):
        return next_daypart_wait_seconds(hour_local, weekday, "next_business_morning")
    return None


@dataclass
class ConversationExample:
    turns: list[dict[str, str]]
    hour_local: int
    weekday: int
    scenario: str = ""
    split: str = ""
    language: str = "en"
    wait_seconds: float | None = None
    bucket_id: int | None = None
    suppress: bool = False
    meta: dict[str, Any] | None = None


@dataclass
class FeaturizerConfig:
    context_dim: int = 768
    assistant_dim: int = 384
    user_dim: int = 384
    char_dim: int = 768
    char_ngram: int = 3

    @property
    def input_dim(self) -> int:
        return self.context_dim + self.assistant_dim + self.user_dim + self.char_dim + 33


class WaitTimeFeaturizer:
    numeric_feature_names = [
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "turn_count",
        "assistant_len_log",
        "user_len_log",
        "ctx_question",
        "assistant_question",
        "user_question",
        "ctx_reminder",
        "user_reminder",
        "assistant_reminder",
        "ctx_busy",
        "user_busy",
        "ctx_dnc",
        "user_dnc",
        "ctx_closure",
        "user_closure",
        "ctx_urgent",
        "ctx_deliverable",
        "ctx_tonight",
        "ctx_morning",
        "ctx_afternoon",
        "ctx_evening",
        "ctx_relative_time",
        "ctx_status",
        "ctx_emotion",
        "time_hint_log",
        "time_hint_bucket",
        "force_suppress_signal",
        "anchor_explicit_wait",
        "explicit_resume_signal",
    ]

    def __init__(self, config: FeaturizerConfig | None = None) -> None:
        self.config = config or FeaturizerConfig()

    def get_config(self) -> dict[str, Any]:
        return asdict(self.config)

    def _hash_tokens(self, tokens: list[str], dim: int, namespace: str) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        if not tokens:
            return vec
        features = list(tokens)
        if len(tokens) > 1:
            features.extend(f"{tokens[i]}__{tokens[i + 1]}" for i in range(len(tokens) - 1))
        for feature in features:
            signed_hash = stable_hash(f"{namespace}:{feature}")
            index = signed_hash % dim
            sign = 1.0 if ((signed_hash >> 7) & 1) == 0 else -1.0
            vec[index] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _hash_chars(self, text: str, dim: int, ngram: int) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        normalized = normalize_text(text)
        if len(normalized) < ngram:
            return vec
        for start in range(len(normalized) - ngram + 1):
            gram = normalized[start : start + ngram]
            signed_hash = stable_hash(f"char:{gram}")
            index = signed_hash % dim
            sign = 1.0 if ((signed_hash >> 11) & 1) == 0 else -1.0
            vec[index] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _pattern(self, name: str, text: str) -> float:
        return float(PATTERNS[name].search(text) is not None)

    def _structured_features(self, example: ConversationExample) -> np.ndarray:
        turns = example.turns
        context_text = "\n".join(f"{turn['role']}: {turn['text']}" for turn in turns)
        last_assistant = next((turn["text"] for turn in reversed(turns) if turn["role"] == "assistant"), "")
        last_user = next((turn["text"] for turn in reversed(turns) if turn["role"] == "user"), "")
        norm_ctx = normalize_text(context_text)
        norm_assistant = normalize_text(last_assistant)
        norm_user = normalize_text(last_user)
        signals = extract_policy_signals(example)

        hour_angle = (example.hour_local % 24) / 24.0 * (2.0 * math.pi)
        weekday_angle = (example.weekday % 7) / 7.0 * (2.0 * math.pi)
        features = np.array(
            [
                math.sin(hour_angle),
                math.cos(hour_angle),
                math.sin(weekday_angle),
                math.cos(weekday_angle),
                min(len(turns), 8) / 8.0,
                min(math.log1p(len(norm_assistant)), 6.0) / 6.0,
                min(math.log1p(len(norm_user)), 6.0) / 6.0,
                self._pattern("question", norm_ctx),
                self._pattern("question", norm_assistant),
                self._pattern("question", norm_user),
                self._pattern("reminder", norm_ctx),
                self._pattern("reminder", norm_user),
                self._pattern("reminder", norm_assistant),
                self._pattern("busy", norm_ctx),
                self._pattern("busy", norm_user),
                self._pattern("do_not_contact", norm_ctx),
                self._pattern("do_not_contact", norm_user),
                self._pattern("closure", norm_ctx),
                self._pattern("closure", norm_user),
                self._pattern("urgent", norm_ctx),
                self._pattern("deliverable", norm_ctx),
                self._pattern("time_tonight", norm_ctx),
                self._pattern("time_morning", norm_ctx),
                self._pattern("time_afternoon", norm_ctx),
                self._pattern("time_evening", norm_ctx),
                self._pattern("time_relative", norm_ctx),
                self._pattern("status", norm_ctx),
                self._pattern("emotion", norm_ctx),
                signals["time_hint_log"],
                signals["time_hint_bucket"],
                float(signals["force_suppress"]),
                float(signals["anchor_explicit_wait"]),
                float(signals["explicit_resume"]),
            ],
            dtype=np.float32,
        )
        return features

    def transform_one(self, example: ConversationExample) -> np.ndarray:
        turns = example.turns
        context_text = "\n".join(f"{turn['role']}: {turn['text']}" for turn in turns)
        last_assistant = next((turn["text"] for turn in reversed(turns) if turn["role"] == "assistant"), "")
        last_user = next((turn["text"] for turn in reversed(turns) if turn["role"] == "user"), "")
        context_tokens = tokenize(context_text)
        assistant_tokens = tokenize(last_assistant)
        user_tokens = tokenize(last_user)

        parts = [
            self._hash_tokens(context_tokens, self.config.context_dim, "ctx"),
            self._hash_tokens(assistant_tokens, self.config.assistant_dim, "assistant"),
            self._hash_tokens(user_tokens, self.config.user_dim, "user"),
            self._hash_chars(context_text, self.config.char_dim, self.config.char_ngram),
            self._structured_features(example),
        ]
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    def transform_many(self, examples: list[ConversationExample]) -> np.ndarray:
        return np.stack([self.transform_one(example) for example in examples], axis=0)


class TinyWaitNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size")
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = float(dropout)
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                ]
            )
            previous = hidden
        self.encoder = nn.Sequential(*layers)
        self.bucket_head = nn.Linear(previous, NUM_BUCKETS)
        self.wait_head = nn.Linear(previous, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(features)
        bucket_logits = self.bucket_head(hidden)
        log_wait = self.wait_head(hidden).squeeze(-1)
        return bucket_logits, log_wait


class KeywordWaitBaseline:
    def predict_one(self, example: ConversationExample) -> dict[str, Any]:
        text = normalize_text("\n".join(f"{turn['role']}: {turn['text']}" for turn in example.turns))
        if PATTERNS["do_not_contact"].search(text) or PATTERNS["closure"].search(text):
            return {
                "bucket_id": SUPPRESS_BUCKET,
                "wait_seconds": None,
                "suppress": True,
                "label": SUPPRESS_LABEL,
            }
        if PATTERNS["urgent"].search(text):
            wait_seconds = 300.0
        elif PATTERNS["time_morning"].search(text):
            wait_seconds = next_daypart_wait_seconds(example.hour_local, example.weekday, "tomorrow_morning")
        elif PATTERNS["time_afternoon"].search(text):
            wait_seconds = next_daypart_wait_seconds(example.hour_local, example.weekday, "afternoon")
        elif PATTERNS["time_tonight"].search(text):
            wait_seconds = next_daypart_wait_seconds(example.hour_local, example.weekday, "tonight")
        elif PATTERNS["busy"].search(text) or PATTERNS["deliverable"].search(text):
            wait_seconds = 21_600.0
        elif PATTERNS["question"].search(text):
            wait_seconds = 2_700.0
        else:
            wait_seconds = 43_200.0
        bucket_id = wait_seconds_to_bucket(wait_seconds)
        return {
            "bucket_id": bucket_id,
            "wait_seconds": wait_seconds,
            "suppress": False,
            "label": bucket_to_label(bucket_id),
        }

    def predict_many(self, examples: list[ConversationExample]) -> list[dict[str, Any]]:
        return [self.predict_one(example) for example in examples]


def extract_policy_signals(example: ConversationExample) -> dict[str, Any]:
    turns = example.turns
    context_text = "\n".join(f"{turn['role']}: {turn['text']}" for turn in turns)
    normalized = normalize_text(context_text)
    has_reminder = PATTERNS["reminder"].search(normalized) is not None
    has_status = PATTERNS["status"].search(normalized) is not None
    has_question = PATTERNS["question"].search(normalized) is not None
    has_urgent = PATTERNS["urgent"].search(normalized) is not None
    has_do_not_contact = PATTERNS["do_not_contact"].search(normalized) is not None
    has_closure = PATTERNS["closure"].search(normalized) is not None
    time_hint_seconds = extract_time_hint_seconds(normalized, example.hour_local, example.weekday)
    explicit_resume = bool(
        time_hint_seconds is not None
        and re.search(r"until|by|around|at|等到|到了|到|之后|以后|then", normalized) is not None
    )
    blocked_again_tonight = re.search(r"again tonight|今晚别再|今晚不要再", normalized) is not None
    force_suppress = bool(
        blocked_again_tonight
        or (has_do_not_contact and not explicit_resume)
        or (has_closure and not has_reminder and not has_status and not has_urgent)
    )
    anchor_explicit_wait = bool(time_hint_seconds is not None and (has_reminder or has_status or has_question or explicit_resume))
    time_hint_log = 0.0 if time_hint_seconds is None else seconds_to_log_wait(time_hint_seconds) / seconds_to_log_wait(MAX_WAIT_SECONDS)
    time_hint_bucket = (
        0.0
        if time_hint_seconds is None
        else wait_seconds_to_bucket(time_hint_seconds) / max(len(BUCKET_SPECS) - 1, 1)
    )
    return {
        "has_reminder": has_reminder,
        "has_status": has_status,
        "has_question": has_question,
        "has_urgent": has_urgent,
        "has_do_not_contact": has_do_not_contact,
        "has_closure": has_closure,
        "time_hint_seconds": time_hint_seconds,
        "explicit_resume": explicit_resume,
        "force_suppress": force_suppress,
        "anchor_explicit_wait": anchor_explicit_wait,
        "time_hint_log": time_hint_log,
        "time_hint_bucket": time_hint_bucket,
    }


def apply_prediction_guardrails(example: ConversationExample, prediction: dict[str, Any]) -> dict[str, Any]:
    signals = extract_policy_signals(example)
    probabilities = prediction.get("probabilities")
    if signals["force_suppress"]:
        return {
            "bucket_id": SUPPRESS_BUCKET,
            "wait_seconds": None,
            "suppress": True,
            "label": SUPPRESS_LABEL,
            "probabilities": probabilities,
        }
    if signals["anchor_explicit_wait"] and signals["time_hint_seconds"] is not None:
        wait_seconds = float(signals["time_hint_seconds"])
        bucket_id = wait_seconds_to_bucket(wait_seconds)
        return {
            "bucket_id": bucket_id,
            "wait_seconds": wait_seconds,
            "suppress": False,
            "label": bucket_to_label(bucket_id),
            "probabilities": probabilities,
        }
    if prediction["suppress"] and signals["has_urgent"] and signals["has_question"]:
        wait_seconds = min(signals["time_hint_seconds"] or 900.0, 900.0)
        bucket_id = wait_seconds_to_bucket(wait_seconds)
        return {
            "bucket_id": bucket_id,
            "wait_seconds": wait_seconds,
            "suppress": False,
            "label": bucket_to_label(bucket_id),
            "probabilities": probabilities,
        }
    return prediction


def apply_guardrails(examples: list[ConversationExample], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [apply_prediction_guardrails(example, prediction) for example, prediction in zip(examples, predictions)]


def decode_prediction(
    bucket_logits: torch.Tensor,
    log_wait: torch.Tensor,
    suppress_threshold: float = 0.52,
) -> list[dict[str, Any]]:
    probabilities = torch.softmax(bucket_logits, dim=-1).detach().cpu().numpy()
    log_wait_np = log_wait.detach().cpu().numpy()
    outputs: list[dict[str, Any]] = []
    for class_probs, log_wait_value in zip(probabilities, log_wait_np):
        predicted_class = int(class_probs.argmax())
        suppress_prob = float(class_probs[SUPPRESS_BUCKET])
        if predicted_class == SUPPRESS_BUCKET or suppress_prob >= suppress_threshold:
            outputs.append(
                {
                    "bucket_id": SUPPRESS_BUCKET,
                    "wait_seconds": None,
                    "suppress": True,
                    "label": SUPPRESS_LABEL,
                    "probabilities": class_probs.tolist(),
                }
            )
            continue
        wait_seconds = float(int(round(log_wait_to_seconds(float(log_wait_value)))))
        bucket_id = wait_seconds_to_bucket(wait_seconds)
        outputs.append(
            {
                "bucket_id": bucket_id,
                "wait_seconds": wait_seconds,
                "suppress": False,
                "label": bucket_to_label(bucket_id),
                "suppress_probability": suppress_prob,
                "probabilities": class_probs.tolist(),
            }
        )
    return outputs


def save_model_bundle(
    path: str,
    model: TinyWaitNet,
    featurizer: WaitTimeFeaturizer,
    metadata: dict[str, Any],
    suppress_threshold: float,
) -> None:
    bundle = {
        "state_dict": model.state_dict(),
        "model_config": {
            "input_dim": featurizer.config.input_dim,
            "hidden_dims": model.hidden_dims,
            "dropout": model.dropout,
        },
        "featurizer_config": featurizer.get_config(),
        "metadata": metadata,
        "suppress_threshold": suppress_threshold,
    }
    torch.save(bundle, path)


def load_model_bundle(path: str, device: str | torch.device | None = "auto") -> tuple[TinyWaitNet, WaitTimeFeaturizer, dict[str, Any]]:
    resolved_device = resolve_device(device)
    bundle = torch.load(path, map_location=resolved_device)
    featurizer = WaitTimeFeaturizer(FeaturizerConfig(**bundle["featurizer_config"]))
    model = TinyWaitNet(
        input_dim=featurizer.config.input_dim,
        hidden_dims=tuple(bundle["model_config"]["hidden_dims"]),
        dropout=float(bundle["model_config"].get("dropout", 0.10)),
    )
    model.load_state_dict(bundle["state_dict"])
    model.to(resolved_device)
    model.eval()
    metadata = {
        "metadata": bundle.get("metadata", {}),
        "suppress_threshold": bundle.get("suppress_threshold", 0.52),
    }
    return model, featurizer, metadata


def predict_wait_time(
    model: TinyWaitNet,
    featurizer: WaitTimeFeaturizer,
    example: ConversationExample,
    device: str | torch.device | None = "auto",
    suppress_threshold: float = 0.52,
) -> dict[str, Any]:
    features = featurizer.transform_one(example)
    resolved_device = resolve_device(device)
    tensor = torch.from_numpy(features).unsqueeze(0).to(resolved_device)
    model.eval()
    with torch.no_grad():
        bucket_logits, log_wait = model(tensor)
    raw_prediction = decode_prediction(bucket_logits, log_wait, suppress_threshold=suppress_threshold)[0]
    return apply_prediction_guardrails(example, raw_prediction)
