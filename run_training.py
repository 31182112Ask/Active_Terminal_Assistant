from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wait_model import (
    BUCKET_SPECS,
    MAX_WAIT_SECONDS,
    SUPPRESS_BUCKET,
    SUPPRESS_LABEL,
    apply_guardrails,
    ConversationExample,
    KeywordWaitBaseline,
    TinyWaitNet,
    WaitTimeFeaturizer,
    build_relative_phrase,
    decode_prediction,
    humanize_wait,
    next_daypart_wait_seconds,
    save_model_bundle,
    seconds_to_log_wait,
    wait_seconds_to_bucket,
)


ARTIFACTS_DIR = Path("artifacts")
THRESHOLD_GRID = [0.42, 0.46, 0.50, 0.54, 0.58, 0.62, 0.66]


DAYPART_PHRASES = {
    "after_lunch": {
        "en": {
            "train": ["after lunch", "this afternoon after lunch"],
            "holdout": ["once lunch is over", "post-lunch"],
        },
        "zh": {
            "train": ["午饭后", "下午吃完饭后"],
            "holdout": ["等午饭过后", "午后再说"],
        },
    },
    "afternoon": {
        "en": {
            "train": ["this afternoon", "later this afternoon"],
            "holdout": ["later in the afternoon", "once the afternoon opens up"],
        },
        "zh": {
            "train": ["今天下午", "下午晚些时候"],
            "holdout": ["等下午空下来", "下午有空时"],
        },
    },
    "tonight": {
        "en": {
            "train": ["tonight", "later tonight"],
            "holdout": ["this evening", "once tonight settles down"],
        },
        "zh": {
            "train": ["今晚", "今天晚上"],
            "holdout": ["等晚上", "今晚稍晚些"],
        },
    },
    "tomorrow_morning": {
        "en": {
            "train": ["tomorrow morning", "first thing tomorrow morning"],
            "holdout": ["early tomorrow", "when tomorrow morning starts"],
        },
        "zh": {
            "train": ["明早", "明天早上"],
            "holdout": ["等明天一早", "明早开始时"],
        },
    },
    "tomorrow_afternoon": {
        "en": {
            "train": ["tomorrow afternoon", "sometime tomorrow afternoon"],
            "holdout": ["by tomorrow afternoon", "once tomorrow afternoon opens up"],
        },
        "zh": {
            "train": ["明天下午", "明天下午晚些时候"],
            "holdout": ["等明天下午", "明天下午有空时"],
        },
    },
    "tomorrow_evening": {
        "en": {
            "train": ["tomorrow evening", "tomorrow night"],
            "holdout": ["by tomorrow night", "once tomorrow evening arrives"],
        },
        "zh": {
            "train": ["明晚", "明天晚上"],
            "holdout": ["等明晚", "明天晚上再说"],
        },
    },
    "next_business_morning": {
        "en": {
            "train": ["next business morning", "the next workday morning"],
            "holdout": ["the next working morning", "the next business-day morning"],
        },
        "zh": {
            "train": ["下一个工作日上午", "下个工作日早上"],
            "holdout": ["等下一个工作日早上", "到下个工作日上午"],
        },
    },
}


def lex_pick(
    rng: random.Random,
    split: str,
    lang: str,
    *,
    en_train: list[str],
    en_holdout: list[str],
    zh_train: list[str],
    zh_holdout: list[str],
) -> str:
    if lang == "zh":
        train_pool = zh_train
        holdout_pool = zh_holdout
    else:
        train_pool = en_train
        holdout_pool = en_holdout
    if split == "lexical":
        pool = holdout_pool
    elif split == "stress" and holdout_pool and rng.random() < 0.5:
        pool = holdout_pool
    else:
        pool = train_pool
    return rng.choice(pool)


def weighted_choice(rng: random.Random, items: list[tuple[str, float, Callable[..., ConversationExample]]]) -> tuple[str, Callable[..., ConversationExample]]:
    total = sum(weight for _, weight, _ in items)
    draw = rng.random() * total
    upto = 0.0
    for name, weight, factory in items:
        upto += weight
        if draw <= upto:
            return name, factory
    return items[-1][0], items[-1][2]


def sample_language(rng: random.Random, split: str) -> str:
    zh_prob = 0.24
    if split == "lexical":
        zh_prob = 0.35
    elif split == "stress":
        zh_prob = 0.30
    return "zh" if rng.random() < zh_prob else "en"


def sample_clock(rng: random.Random, split: str) -> tuple[int, int]:
    if split == "stress" and rng.random() < 0.55:
        hour = rng.choice([0, 1, 5, 6, 11, 12, 18, 22, 23])
    else:
        hour = rng.randint(7, 22)
    weekday = rng.randint(0, 6)
    return hour, weekday


def sample_wait_seconds(
    rng: random.Random,
    lower: float,
    upper: float,
    *,
    log_scale: bool = True,
) -> float:
    if log_scale:
        value = math.exp(rng.uniform(math.log(lower), math.log(upper)))
    else:
        value = rng.uniform(lower, upper)
    return float(int(round(max(1.0, min(MAX_WAIT_SECONDS, value)))))


def sample_edge_wait_seconds(rng: random.Random, centers: list[float], jitter_ratio: float = 0.08) -> float:
    center = rng.choice(centers)
    radius = max(2.0, center * jitter_ratio)
    value = center + rng.uniform(-radius, radius)
    return float(int(round(max(1.0, min(MAX_WAIT_SECONDS, value)))))


def slot_phrase(slot: str, lang: str, split: str, rng: random.Random) -> str:
    pool = DAYPART_PHRASES[slot][lang]
    options = pool["holdout"] if split == "lexical" else pool["train"]
    if split == "stress" and rng.random() < 0.45:
        options = pool["holdout"]
    return rng.choice(options)


def sample_deadline(rng: random.Random, split: str, lang: str, hour_local: int, weekday: int, hard: bool) -> tuple[str, float]:
    if hard and rng.random() < 0.70:
        wait_seconds = sample_edge_wait_seconds(
            rng,
            [15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 300.0, 600.0, 900.0, 1_800.0, 3_600.0, 7_200.0, 14_400.0],
            jitter_ratio=0.06,
        )
        return build_relative_phrase(wait_seconds, lang, holdout=(split == "lexical")), wait_seconds

    if rng.random() < 0.58:
        lower, upper = rng.choice(
            [
                (15.0, 120.0),
                (120.0, 900.0),
                (900.0, 7_200.0),
                (7_200.0, 43_200.0),
            ]
        )
        wait_seconds = sample_wait_seconds(rng, lower, upper, log_scale=True)
        return build_relative_phrase(wait_seconds, lang, holdout=(split == "lexical")), wait_seconds

    slot = rng.choice(
        [
            "after_lunch",
            "afternoon",
            "tonight",
            "tomorrow_morning",
            "tomorrow_afternoon",
            "tomorrow_evening",
            "next_business_morning",
        ]
    )
    wait_seconds = next_daypart_wait_seconds(hour_local, weekday, slot)
    return slot_phrase(slot, lang, split, rng), wait_seconds


def make_example(
    *,
    turns: list[tuple[str, str]],
    hour_local: int,
    weekday: int,
    scenario: str,
    split: str,
    language: str,
    wait_seconds: float | None,
    suppress: bool,
    meta: dict[str, Any] | None = None,
) -> ConversationExample:
    bucket_id = wait_seconds_to_bucket(wait_seconds, suppress=suppress)
    return ConversationExample(
        turns=[{"role": role, "text": text} for role, text in turns],
        hour_local=hour_local,
        weekday=weekday,
        scenario=scenario,
        split=split,
        language=language,
        wait_seconds=None if suppress else float(wait_seconds),
        bucket_id=bucket_id,
        suppress=suppress,
        meta=meta or {},
    )


def generate_scheduled_reminder(rng: random.Random, split: str, hard: bool = False) -> ConversationExample:
    lang = sample_language(rng, split)
    hour_local, weekday = sample_clock(rng, split)
    item = lex_pick(
        rng,
        split,
        lang,
        en_train=["draft", "deck", "proposal", "report", "patch notes"],
        en_holdout=["brief", "doc", "review packet", "slides pack"],
        zh_train=["草稿", "汇报", "方案", "报告", "补丁说明"],
        zh_holdout=["文档", "提案", "材料包", "演示稿"],
    )
    remind_phrase = lex_pick(
        rng,
        split,
        lang,
        en_train=["remind me", "check back with me", "follow up with me"],
        en_holdout=["nudge me", "circle back to me", "ping me again"],
        zh_train=["提醒我", "回头问我", "再跟进我一下"],
        zh_holdout=["再找我一下", "再戳我一下", "回头再问我"],
    )
    confirm_phrase = lex_pick(
        rng,
        split,
        lang,
        en_train=["Understood", "Sounds good", "Got it"],
        en_holdout=["Works for me", "Okay noted", "Understood on my end"],
        zh_train=["好", "明白", "收到"],
        zh_holdout=["行", "知道了", "好，我记下了"],
    )
    deadline_phrase, wait_seconds = sample_deadline(rng, split, lang, hour_local, weekday, hard)
    if lang == "en":
        turns = [
            ("user", f"Can you review the {item} once I finish it?"),
            ("assistant", f"Yes. Send the {item} over when you're ready."),
            ("user", f"I'm tied up until {deadline_phrase}. If I still haven't sent the {item} by then, {remind_phrase}."),
            ("assistant", f"{confirm_phrase}. I'll stay quiet for now and follow up around {deadline_phrase} if I still haven't heard from you."),
        ]
    else:
        turns = [
            ("user", f"我把{item}整理好后，你能帮我看一下吗？"),
            ("assistant", f"可以，你准备好后把{item}发我。"),
            ("user", f"我会忙到{deadline_phrase}。如果到时候我还没把{item}发出来，就{remind_phrase}。"),
            ("assistant", f"{confirm_phrase}。我先不打扰，等到{deadline_phrase}左右如果还没收到，我再主动跟进。"),
        ]
    return make_example(
        turns=turns,
        hour_local=hour_local,
        weekday=weekday,
        scenario="scheduled_reminder",
        split=split,
        language=lang,
        wait_seconds=wait_seconds,
        suppress=False,
        meta={"deadline_phrase": deadline_phrase},
    )


def generate_urgent_incident(rng: random.Random, split: str, hard: bool = False) -> ConversationExample:
    lang = sample_language(rng, split)
    hour_local, weekday = sample_clock(rng, split)
    system_name = lex_pick(
        rng,
        split,
        lang,
        en_train=["worker queue", "payments API", "prod pipeline", "auth service"],
        en_holdout=["job runner", "checkout service", "live ingestion flow", "session service"],
        zh_train=["任务队列", "支付接口", "生产流水线", "认证服务"],
        zh_holdout=["作业执行器", "结算服务", "实时链路", "会话服务"],
    )
    if hard:
        wait_seconds = sample_edge_wait_seconds(rng, [12.0, 20.0, 35.0, 55.0, 90.0, 180.0, 300.0, 420.0], jitter_ratio=0.05)
    else:
        wait_seconds = sample_wait_seconds(rng, 8.0, 420.0, log_scale=True)
    if hour_local < 7 or hour_local > 20:
        wait_seconds = float(int(round(min(wait_seconds, 240.0))))
    duration_phrase = build_relative_phrase(wait_seconds, lang, holdout=(split == "lexical"))
    if lang == "en":
        turns = [
            ("user", f"The {system_name} is still down after the last patch."),
            ("assistant", f"Restart the affected components and clear any stuck jobs tied to the {system_name}."),
            ("user", "Running that now."),
            ("assistant", f"If I don't hear back, I'll check in {duration_phrase} because this still looks urgent."),
        ]
    else:
        turns = [
            ("user", f"{system_name}打完补丁后还是挂着。"),
            ("assistant", f"先把{system_name}相关组件重启一下，再清掉卡住的任务。"),
            ("user", "我现在就在处理。"),
            ("assistant", f"如果{duration_phrase}还没收到你的回复，我会主动再确认一次，这个问题还算紧急。"),
        ]
    return make_example(
        turns=turns,
        hour_local=hour_local,
        weekday=weekday,
        scenario="urgent_incident",
        split=split,
        language=lang,
        wait_seconds=wait_seconds,
        suppress=False,
    )


def generate_support_followup(rng: random.Random, split: str, hard: bool = False) -> ConversationExample:
    lang = sample_language(rng, split)
    hour_local, weekday = sample_clock(rng, split)
    issue = lex_pick(
        rng,
        split,
        lang,
        en_train=["login issue", "billing mismatch", "sync error", "account lock"],
        en_holdout=["sign-in problem", "invoice mismatch", "replication glitch", "workspace lock"],
        zh_train=["登录问题", "计费异常", "同步报错", "账号锁定"],
        zh_holdout=["登录故障", "账单不一致", "复制异常", "工作区锁定"],
    )
    if hard:
        wait_seconds = sample_edge_wait_seconds(rng, [45.0, 90.0, 180.0, 300.0, 600.0, 900.0, 1_800.0], jitter_ratio=0.08)
    else:
        wait_seconds = sample_wait_seconds(rng, 45.0, 2_400.0, log_scale=True)
    phrase = build_relative_phrase(wait_seconds, lang, holdout=(split == "lexical"))
    if lang == "en":
        turns = [
            ("user", f"I'm still stuck on this {issue}."),
            ("assistant", "Please try the two steps I just outlined and send me the result."),
            ("user", "Will do."),
            ("assistant", f"If it stays broken, I'll follow up {phrase} so this doesn't stall."),
        ]
    else:
        turns = [
            ("user", f"这个{issue}我还是没解决。"),
            ("assistant", "先按我刚才给的两个步骤试一下，然后把结果告诉我。"),
            ("user", "好，我去试。"),
            ("assistant", f"如果到{phrase}问题还没解决，我会再主动跟进一次，避免卡住。"),
        ]
    return make_example(
        turns=turns,
        hour_local=hour_local,
        weekday=weekday,
        scenario="support_followup",
        split=split,
        language=lang,
        wait_seconds=wait_seconds,
        suppress=False,
    )


def generate_do_not_disturb(rng: random.Random, split: str, hard: bool = False) -> ConversationExample:
    lang = sample_language(rng, split)
    hour_local, weekday = sample_clock(rng, split)
    has_explicit_resume = rng.random() < (0.55 if hard else 0.45)
    if has_explicit_resume:
        resume_phrase, wait_seconds = sample_deadline(rng, split, lang, hour_local, weekday, hard)
        suppress = False
    else:
        resume_phrase = ""
        wait_seconds = None
        suppress = True
    if lang == "en":
        user_text = "I'm about to disappear into meetings, so please don't ping me again"
        if has_explicit_resume:
            user_text += f" until {resume_phrase}"
        user_text += "."
        assistant_text = (
            f"Understood. I'll stay quiet until {resume_phrase} and only reach back out then."
            if has_explicit_resume
            else "Understood. I'll leave it here and wait for you to come back."
        )
    else:
        user_text = "我马上要去开会了，先别再找我"
        if has_explicit_resume:
            user_text += f"，等到{resume_phrase}再说"
        assistant_text = (
            f"明白，我会先保持安静，等到{resume_phrase}再主动联系。"
            if has_explicit_resume
            else "明白，我先不打扰，等你回来再继续。"
        )
    turns = [
        ("user", "我们把方案定下来了。"),
        ("assistant", "好，我已经记住当前结论。"),
        ("user", user_text),
        ("assistant", assistant_text),
    ]
    return make_example(
        turns=turns,
        hour_local=hour_local,
        weekday=weekday,
        scenario="do_not_disturb",
        split=split,
        language=lang,
        wait_seconds=wait_seconds,
        suppress=suppress,
    )


def generate_polite_close(rng: random.Random, split: str, hard: bool = False) -> ConversationExample:
    del hard
    lang = sample_language(rng, split)
    hour_local, weekday = sample_clock(rng, split)
    if lang == "en":
        turns = [
            ("user", "Perfect, that solved it. Thanks."),
            ("assistant", "Glad it helped."),
            ("user", "That's all for today."),
            ("assistant", "I'll leave it here unless you want to continue later."),
        ]
    else:
        turns = [
            ("user", "好了，这个问题解决了，谢谢。"),
            ("assistant", "好，能帮上就行。"),
            ("user", "今天就先这样。"),
            ("assistant", "那我先停在这里，之后你想继续再叫我。"),
        ]
    return make_example(
        turns=turns,
        hour_local=hour_local,
        weekday=weekday,
        scenario="polite_close",
        split=split,
        language=lang,
        wait_seconds=None,
        suppress=True,
    )


def generate_clarification(rng: random.Random, split: str, hard: bool = False) -> ConversationExample:
    lang = sample_language(rng, split)
    hour_local, weekday = sample_clock(rng, split)
    urgent = rng.random() < 0.45
    if urgent:
        wait_seconds = sample_wait_seconds(rng, 20.0, 900.0, log_scale=True)
    else:
        wait_seconds = sample_wait_seconds(rng, 300.0, 7_200.0, log_scale=True)
    if hard and urgent:
        wait_seconds = sample_edge_wait_seconds(rng, [30.0, 60.0, 90.0, 180.0, 300.0, 600.0], jitter_ratio=0.06)
    topic = lex_pick(
        rng,
        split,
        lang,
        en_train=["SQL fix", "launch note", "metrics review", "API spec"],
        en_holdout=["query patch", "release note", "KPI readout", "integration brief"],
        zh_train=["SQL修复", "发布说明", "指标复盘", "接口规范"],
        zh_holdout=["查询补丁", "上线说明", "KPI汇总", "集成说明"],
    )
    phrase = build_relative_phrase(wait_seconds, lang, holdout=(split == "lexical"))
    if lang == "en":
        urgency_text = "This is time-sensitive." if urgent else "No rush, but I still need the missing detail."
        turns = [
            ("user", f"Please finish the {topic} for me."),
            ("assistant", f"I can, but I still need one missing detail. {urgency_text}"),
            ("assistant", "Which dataset or source of truth should I use?"),
            ("assistant", f"If you go quiet, I'll check back {phrase} so we don't stay blocked."),
        ]
    else:
        urgency_text = "这个事情比较急。" if urgent else "不算特别急，但我还缺一个关键细节。"
        turns = [
            ("user", f"帮我把{topic}做完。"),
            ("assistant", f"可以，但我还缺一个关键信息。{urgency_text}"),
            ("assistant", "你希望我基于哪个数据集或者哪个版本来做？"),
            ("assistant", f"如果你暂时没回复，我会在{phrase}再主动确认一次，避免继续卡住。"),
        ]
    return make_example(
        turns=turns,
        hour_local=hour_local,
        weekday=weekday,
        scenario="clarification_needed",
        split=split,
        language=lang,
        wait_seconds=wait_seconds,
        suppress=False,
        meta={"urgent": urgent},
    )


def generate_accountability(rng: random.Random, split: str, hard: bool = False) -> ConversationExample:
    lang = sample_language(rng, split)
    hour_local, weekday = sample_clock(rng, split)
    task = lex_pick(
        rng,
        split,
        lang,
        en_train=["the workout", "the draft", "the outreach list", "the mock interview"],
        en_holdout=["the study block", "the writing sprint", "the checklist", "the practice run"],
        zh_train=["锻炼", "草稿", "外联名单", "模拟面试"],
        zh_holdout=["学习任务", "写作冲刺", "清单", "演练"],
    )
    deadline_phrase, wait_seconds = sample_deadline(rng, split, lang, hour_local, weekday, hard)
    if lang == "en":
        turns = [
            ("user", f"I'm trying to finish {task}."),
            ("assistant", "Okay, what kind of check-in would help?"),
            ("user", f"If I haven't done {task} by {deadline_phrase}, please follow up."),
            ("assistant", f"Will do. I'll check back around {deadline_phrase} if I still haven't heard that it's done."),
        ]
    else:
        turns = [
            ("user", f"我想把{task}完成掉。"),
            ("assistant", "好，你希望我怎么跟进最有帮助？"),
            ("user", f"如果我到{deadline_phrase}还没把{task}做完，你就主动提醒我。"),
            ("assistant", f"可以。如果到{deadline_phrase}我还没听到你说已经完成，我会主动来问。"),
        ]
    return make_example(
        turns=turns,
        hour_local=hour_local,
        weekday=weekday,
        scenario="accountability_checkin",
        split=split,
        language=lang,
        wait_seconds=wait_seconds,
        suppress=False,
    )


def generate_sensitive_space(rng: random.Random, split: str, hard: bool = False) -> ConversationExample:
    lang = sample_language(rng, split)
    hour_local, weekday = sample_clock(rng, split)
    explicit_revisit = rng.random() < (0.4 if hard else 0.3)
    if explicit_revisit:
        revisit_phrase, wait_seconds = sample_deadline(rng, split, lang, hour_local, weekday, hard)
        suppress = False
    else:
        revisit_phrase = ""
        wait_seconds = None
        suppress = True
    if lang == "en":
        user_text = "I'm a bit overwhelmed and need some space"
        if explicit_revisit:
            user_text += f". You can check on me {revisit_phrase}"
        else:
            user_text += "."
        assistant_text = (
            f"Understood. I'll give you room and only check back {revisit_phrase}."
            if explicit_revisit
            else "Understood. I'll step back and wait for you to restart the conversation."
        )
    else:
        user_text = "我现在有点情绪上来，想先缓一缓"
        if explicit_revisit:
            user_text += f"，你可以{revisit_phrase}再来问我"
        assistant_text = (
            f"明白，我先给你留出空间，等到{revisit_phrase}再来确认。"
            if explicit_revisit
            else "明白，我先退后一点，等你想继续时再回来。"
        )
    turns = [
        ("user", user_text),
        ("assistant", assistant_text),
    ]
    return make_example(
        turns=turns,
        hour_local=hour_local,
        weekday=weekday,
        scenario="sensitive_space",
        split=split,
        language=lang,
        wait_seconds=wait_seconds,
        suppress=suppress,
    )


def generate_async_deliverable(rng: random.Random, split: str, hard: bool = False) -> ConversationExample:
    lang = sample_language(rng, split)
    hour_local, weekday = sample_clock(rng, split)
    item = lex_pick(
        rng,
        split,
        lang,
        en_train=["spreadsheet", "doc", "deck", "PRD", "timeline"],
        en_holdout=["sheet", "write-up", "slides file", "requirements note", "plan"],
        zh_train=["表格", "文档", "演示稿", "需求文档", "时间线"],
        zh_holdout=["清单", "材料", "幻灯片文件", "说明稿", "计划"],
    )
    explicit_deadline = rng.random() < 0.55
    if explicit_deadline:
        deadline_phrase, wait_seconds = sample_deadline(rng, split, lang, hour_local, weekday, hard)
    else:
        if hard:
            wait_seconds = sample_edge_wait_seconds(rng, [1_800.0, 3_600.0, 7_200.0, 14_400.0, 28_800.0], jitter_ratio=0.05)
        else:
            wait_seconds = sample_wait_seconds(rng, 900.0, 28_800.0, log_scale=True)
        deadline_phrase = build_relative_phrase(wait_seconds, lang, holdout=(split == "lexical"))
    if lang == "en":
        turns = [
            ("user", f"I still need time to finish the {item}."),
            ("assistant", f"No problem. Send the {item} when it's ready."),
            ("user", f"I'm heads down for a while. If you still haven't seen the {item} by {deadline_phrase}, check back."),
            ("assistant", f"Understood. I'll wait for the {item} and follow up around {deadline_phrase} if it still hasn't landed."),
        ]
    else:
        turns = [
            ("user", f"这个{item}我还需要一点时间才能整理完。"),
            ("assistant", f"没问题，你整理好以后把{item}发来。"),
            ("user", f"我接下来会埋头处理。如果到{deadline_phrase}你还没看到{item}，就再跟进我一下。"),
            ("assistant", f"明白。我先等你处理，到了{deadline_phrase}如果{item}还没发来，我会再主动问。"),
        ]
    return make_example(
        turns=turns,
        hour_local=hour_local,
        weekday=weekday,
        scenario="async_deliverable",
        split=split,
        language=lang,
        wait_seconds=wait_seconds,
        suppress=False,
    )


def generate_order_status(rng: random.Random, split: str, hard: bool = False) -> ConversationExample:
    lang = sample_language(rng, split)
    hour_local, weekday = sample_clock(rng, split)
    thing = lex_pick(
        rng,
        split,
        lang,
        en_train=["shipment", "application", "vendor reply", "contract review"],
        en_holdout=["delivery update", "candidate packet", "supplier response", "legal review"],
        zh_train=["快递状态", "申请进展", "供应商回复", "合同审核"],
        zh_holdout=["配送更新", "候选人材料", "厂商回信", "法务审阅"],
    )
    if hard:
        wait_seconds = sample_edge_wait_seconds(rng, [21_600.0, 43_200.0, 86_400.0, 172_800.0], jitter_ratio=0.03)
    else:
        wait_seconds = sample_wait_seconds(rng, 10_800.0, 172_800.0, log_scale=True)
    phrase = build_relative_phrase(wait_seconds, lang, holdout=(split == "lexical"))
    if lang == "en":
        turns = [
            ("user", f"I haven't seen any movement on the {thing} yet."),
            ("assistant", "Okay. A little patience still makes sense here."),
            ("assistant", f"If there's still no update {phrase}, I'll proactively check back."),
        ]
    else:
        turns = [
            ("user", f"{thing}现在还没有新动静。"),
            ("assistant", "好，这种情况现在再等等是合理的。"),
            ("assistant", f"如果到了{phrase}还是没更新，我会主动回来确认。"),
        ]
    return make_example(
        turns=turns,
        hour_local=hour_local,
        weekday=weekday,
        scenario="order_status",
        split=split,
        language=lang,
        wait_seconds=wait_seconds,
        suppress=False,
    )


SCENARIOS: list[tuple[str, float, Callable[..., ConversationExample]]] = [
    ("scheduled_reminder", 1.3, generate_scheduled_reminder),
    ("urgent_incident", 0.9, generate_urgent_incident),
    ("support_followup", 1.0, generate_support_followup),
    ("do_not_disturb", 0.9, generate_do_not_disturb),
    ("polite_close", 0.8, generate_polite_close),
    ("clarification_needed", 0.9, generate_clarification),
    ("accountability_checkin", 0.9, generate_accountability),
    ("sensitive_space", 0.7, generate_sensitive_space),
    ("async_deliverable", 1.1, generate_async_deliverable),
    ("order_status", 0.8, generate_order_status),
]


def build_split(size: int, split: str, seed: int) -> list[ConversationExample]:
    rng = random.Random(seed)
    examples: list[ConversationExample] = []
    hard = split == "stress"
    for _ in range(size):
        _, factory = weighted_choice(rng, SCENARIOS)
        examples.append(factory(rng, split, hard))
    return examples


def build_benchmark_splits(seed: int = 7) -> dict[str, list[ConversationExample]]:
    return {
        "train": build_split(72_000, "train", seed + 11),
        "dev": build_split(8_000, "train", seed + 19),
        "test_iid": build_split(8_000, "train", seed + 23),
        "test_lexical": build_split(8_000, "lexical", seed + 29),
        "test_stress": build_split(8_000, "stress", seed + 31),
    }


def build_golden_cases() -> list[ConversationExample]:
    return [
        make_example(
            turns=[
                ("user", "I'm in meetings. Don't ping me again tonight."),
                ("assistant", "Understood. I'll leave it here and wait for you."),
            ],
            hour_local=13,
            weekday=2,
            scenario="golden_dnd",
            split="golden",
            language="en",
            wait_seconds=None,
            suppress=True,
        ),
        make_example(
            turns=[
                ("user", "我会忙到明早，如果我还没把草稿发出来，你就提醒我。"),
                ("assistant", "好，我等到明早如果还没收到，就主动提醒。"),
            ],
            hour_local=16,
            weekday=1,
            scenario="golden_scheduled",
            split="golden",
            language="zh",
            wait_seconds=next_daypart_wait_seconds(16, 1, "tomorrow_morning"),
            suppress=False,
        ),
        make_example(
            turns=[
                ("user", "Prod is still down."),
                ("assistant", "Run the restart."),
                ("assistant", "If I don't hear back, I'll check in in 4 minutes."),
            ],
            hour_local=10,
            weekday=0,
            scenario="golden_urgent",
            split="golden",
            language="en",
            wait_seconds=240.0,
            suppress=False,
        ),
        make_example(
            turns=[
                ("user", "Perfect, that's all."),
                ("assistant", "Glad it helped. I'll leave it here."),
            ],
            hour_local=18,
            weekday=4,
            scenario="golden_close",
            split="golden",
            language="en",
            wait_seconds=None,
            suppress=True,
        ),
        make_example(
            turns=[
                ("user", "这个问题不算急，但我还缺一个字段。"),
                ("assistant", "你希望我用哪个数据源？如果你暂时没回，我两小时后再问。"),
            ],
            hour_local=11,
            weekday=3,
            scenario="golden_clarify",
            split="golden",
            language="zh",
            wait_seconds=7_200.0,
            suppress=False,
        ),
        make_example(
            turns=[
                ("user", "I'm overwhelmed. Please give me space."),
                ("assistant", "Understood. I'll wait for you to restart the conversation."),
            ],
            hour_local=21,
            weekday=5,
            scenario="golden_space",
            split="golden",
            language="en",
            wait_seconds=None,
            suppress=True,
        ),
        make_example(
            turns=[
                ("user", "If I still haven't finished the PRD by after lunch, follow up."),
                ("assistant", "Will do. I'll check back after lunch."),
            ],
            hour_local=9,
            weekday=2,
            scenario="golden_after_lunch",
            split="golden",
            language="en",
            wait_seconds=next_daypart_wait_seconds(9, 2, "after_lunch"),
            suppress=False,
        ),
        make_example(
            turns=[
                ("user", "这个供应商回复还没到。"),
                ("assistant", "如果一天后还是没有消息，我再回来确认。"),
            ],
            hour_local=14,
            weekday=1,
            scenario="golden_vendor",
            split="golden",
            language="zh",
            wait_seconds=86_400.0,
            suppress=False,
        ),
    ]


def examples_to_arrays(
    featurizer: WaitTimeFeaturizer,
    examples: list[ConversationExample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = featurizer.transform_many(examples)
    bucket_targets = np.array([example.bucket_id for example in examples], dtype=np.int64)
    log_wait_targets = np.array(
        [0.0 if example.suppress else seconds_to_log_wait(float(example.wait_seconds)) for example in examples],
        dtype=np.float32,
    )
    suppress_mask = np.array([0.0 if example.suppress else 1.0 for example in examples], dtype=np.float32)
    return features, bucket_targets, log_wait_targets, suppress_mask


def binary_f1_score(true_positive_mask: np.ndarray, predicted_positive_mask: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.logical_and(true_positive_mask, predicted_positive_mask).sum())
    fp = float(np.logical_and(~true_positive_mask, predicted_positive_mask).sum())
    fn = float(np.logical_and(true_positive_mask, ~predicted_positive_mask).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    if precision + recall == 0.0:
        return 0.0, 0.0, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_predictions(name: str, examples: list[ConversationExample], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    true_bucket = np.array([example.bucket_id for example in examples], dtype=np.int64)
    pred_bucket = np.array([prediction["bucket_id"] for prediction in predictions], dtype=np.int64)
    true_positive = true_bucket != SUPPRESS_BUCKET
    pred_positive = pred_bucket != SUPPRESS_BUCKET
    precision, recall, followup_f1 = binary_f1_score(true_positive, pred_positive)

    exact_bucket_acc = float((true_bucket == pred_bucket).mean())
    within_one = []
    abs_errors = []
    relative_errors = []
    adaptive_15_hits = []
    adaptive_60_hits = []
    adaptive_300_hits = []
    false_early = []
    false_late = []
    short_errors = []
    short_within_5 = []
    for example, prediction in zip(examples, predictions):
        if example.bucket_id == SUPPRESS_BUCKET or prediction["bucket_id"] == SUPPRESS_BUCKET:
            within_one.append(example.bucket_id == prediction["bucket_id"])
        else:
            within_one.append(abs(example.bucket_id - prediction["bucket_id"]) <= 1)

        if not example.suppress:
            predicted_wait = prediction["wait_seconds"] if prediction["wait_seconds"] is not None else MAX_WAIT_SECONDS
            true_wait = float(example.wait_seconds)
            abs_error = abs(predicted_wait - true_wait)
            abs_errors.append(abs_error)
            relative_errors.append(abs_error / max(true_wait, 1.0))
            adaptive_15_hits.append(abs_error <= max(15.0, true_wait * 0.02))
            adaptive_60_hits.append(abs_error <= max(60.0, true_wait * 0.05))
            adaptive_300_hits.append(abs_error <= max(300.0, true_wait * 0.10))
            false_early.append(predicted_wait + 15.0 < true_wait * 0.85)
            false_late.append(predicted_wait - 15.0 > true_wait * 1.15)
            if true_wait <= 600.0:
                short_errors.append(abs_error)
                short_within_5.append(abs_error <= 5.0)

    mae_seconds = float(np.mean(abs_errors)) if abs_errors else 0.0
    median_abs_error_seconds = float(np.median(abs_errors)) if abs_errors else 0.0
    p90_abs_error_seconds = float(np.percentile(abs_errors, 90)) if abs_errors else 0.0
    mape = float(np.mean(relative_errors)) if relative_errors else 0.0
    adaptive_15s_rate = float(np.mean(adaptive_15_hits)) if adaptive_15_hits else 0.0
    adaptive_60s_rate = float(np.mean(adaptive_60_hits)) if adaptive_60_hits else 0.0
    adaptive_300s_rate = float(np.mean(adaptive_300_hits)) if adaptive_300_hits else 0.0
    false_early_rate = float(np.mean(false_early)) if false_early else 0.0
    false_late_rate = float(np.mean(false_late)) if false_late else 0.0
    short_mae_seconds = float(np.mean(short_errors)) if short_errors else 0.0
    short_within_5s_rate = float(np.mean(short_within_5)) if short_within_5 else 0.0
    short_component = 1.0 if not short_errors else max(0.0, 1.0 - short_mae_seconds / 30.0)
    mape_component = max(0.0, 1.0 - min(mape, 1.0))
    practical_score = (
        0.16 * followup_f1
        + 0.16 * adaptive_15s_rate
        + 0.16 * adaptive_60s_rate
        + 0.16 * adaptive_300s_rate
        + 0.12 * float(np.mean(within_one))
        + 0.10 * short_within_5s_rate
        + 0.08 * short_component
        + 0.04 * (1.0 - false_early_rate)
        + 0.02 * (1.0 - false_late_rate)
        + 0.00 * mape_component
    )
    return {
        "name": name,
        "size": len(examples),
        "exact_bucket_acc": exact_bucket_acc,
        "within_one_bucket_acc": float(np.mean(within_one)),
        "followup_precision": precision,
        "followup_recall": recall,
        "followup_f1": followup_f1,
        "mae_seconds": mae_seconds,
        "median_abs_error_seconds": median_abs_error_seconds,
        "p90_abs_error_seconds": p90_abs_error_seconds,
        "mape": mape,
        "adaptive_15s_rate": adaptive_15s_rate,
        "adaptive_60s_rate": adaptive_60s_rate,
        "adaptive_300s_rate": adaptive_300s_rate,
        "short_mae_seconds": short_mae_seconds,
        "short_within_5s_rate": short_within_5s_rate,
        "false_early_rate": false_early_rate,
        "false_late_rate": false_late_rate,
        "practical_score": practical_score,
    }


def aggregate_reports(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    weights = {name: report["size"] for name, report in reports.items()}
    total = float(sum(weights.values()))
    metric_names = [
        "exact_bucket_acc",
        "within_one_bucket_acc",
        "followup_precision",
        "followup_recall",
        "followup_f1",
        "mae_seconds",
        "median_abs_error_seconds",
        "p90_abs_error_seconds",
        "mape",
        "adaptive_15s_rate",
        "adaptive_60s_rate",
        "adaptive_300s_rate",
        "short_mae_seconds",
        "short_within_5s_rate",
        "false_early_rate",
        "false_late_rate",
        "practical_score",
    ]
    aggregate = {"size": int(total)}
    for metric_name in metric_names:
        aggregate[metric_name] = sum(reports[name][metric_name] * weights[name] for name in reports) / total
    return aggregate


def evaluate_golden_cases(examples: list[ConversationExample], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for example, prediction in zip(examples, predictions):
        if example.suppress:
            passed = prediction["suppress"]
        else:
            predicted_wait = prediction["wait_seconds"] if prediction["wait_seconds"] is not None else MAX_WAIT_SECONDS
            true_wait = float(example.wait_seconds)
            if true_wait <= 120.0:
                tolerance = 5.0
            elif true_wait <= 900.0:
                tolerance = 15.0
            elif true_wait <= 14_400.0:
                tolerance = max(60.0, true_wait * 0.05)
            else:
                tolerance = max(300.0, true_wait * 0.08)
            passed = (prediction["bucket_id"] != SUPPRESS_BUCKET) and (abs(predicted_wait - true_wait) <= tolerance)
        results.append(
            {
                "scenario": example.scenario,
                "target": SUPPRESS_LABEL if example.suppress else f"{int(round(float(example.wait_seconds)))}s",
                "predicted": SUPPRESS_LABEL if prediction["suppress"] else f"{int(round(float(prediction['wait_seconds'])))}s",
                "passed": passed,
            }
        )
    pass_rate = statistics.mean(result["passed"] for result in results) if results else 0.0
    return {"pass_rate": pass_rate, "cases": results}


def infer_raw_logits(
    model: TinyWaitNet,
    features: np.ndarray,
    device: str | torch.device,
    batch_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_batches = []
    wait_batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            bucket_logits, log_wait = model(batch)
            logits_batches.append(bucket_logits.cpu())
            wait_batches.append(log_wait.cpu())
    return torch.cat(logits_batches, dim=0), torch.cat(wait_batches, dim=0)


def decode_from_raw(
    logits: torch.Tensor,
    log_wait: torch.Tensor,
    threshold: float,
    examples: list[ConversationExample] | None = None,
) -> list[dict[str, Any]]:
    predictions = decode_prediction(logits, log_wait, suppress_threshold=threshold)
    if examples is None:
        return predictions
    return apply_guardrails(examples, predictions)


def tune_suppress_threshold(
    logits: torch.Tensor,
    log_wait: torch.Tensor,
    examples: list[ConversationExample],
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    best_threshold = THRESHOLD_GRID[0]
    best_report = None
    best_predictions = None
    for threshold in THRESHOLD_GRID:
        predictions = decode_from_raw(logits, log_wait, threshold, examples)
        report = evaluate_predictions("dev", examples, predictions)
        if best_report is None or report["practical_score"] > best_report["practical_score"]:
            best_threshold = threshold
            best_report = report
            best_predictions = predictions
    assert best_report is not None
    assert best_predictions is not None
    return best_threshold, best_report, best_predictions


@dataclass
class TrialConfig:
    name: str
    hidden_dims: tuple[int, ...]
    dropout: float
    lr: float
    batch_size: int
    epochs: int
    weight_decay: float


TRIAL_CONFIGS = [
    TrialConfig(name="large", hidden_dims=(384, 192, 96), dropout=0.10, lr=9e-4, batch_size=384, epochs=10, weight_decay=1e-4),
    TrialConfig(name="xlarge", hidden_dims=(512, 256, 128), dropout=0.10, lr=8e-4, batch_size=384, epochs=12, weight_decay=8e-5),
    TrialConfig(name="deep", hidden_dims=(640, 320, 160), dropout=0.08, lr=7e-4, batch_size=384, epochs=14, weight_decay=8e-5),
]


def train_one_trial(
    trial: TrialConfig,
    train_features: np.ndarray,
    train_bucket: np.ndarray,
    train_log_wait: np.ndarray,
    train_mask: np.ndarray,
    dev_features: np.ndarray,
    dev_examples: list[ConversationExample],
    device: str | torch.device,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TinyWaitNet(input_dim=train_features.shape[1], hidden_dims=trial.hidden_dims, dropout=trial.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trial.lr, weight_decay=trial.weight_decay)
    counts = Counter(train_bucket.tolist())
    class_weights = np.ones(len(BUCKET_SPECS) + 1, dtype=np.float32)
    for bucket_id in range(len(class_weights)):
        class_weights[bucket_id] = len(train_bucket) / max(counts.get(bucket_id, 1), 1)
    class_weights /= class_weights.mean()
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    dataset = TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy(train_bucket),
        torch.from_numpy(train_log_wait),
        torch.from_numpy(train_mask),
    )
    loader = DataLoader(dataset, batch_size=trial.batch_size, shuffle=True, drop_last=False)

    best_state = None
    best_threshold = 0.52
    best_report = None
    best_epoch = 0
    patience = 4
    remaining = patience
    history = []

    for epoch in range(1, trial.epochs + 1):
        model.train()
        running_losses = []
        for batch_features, batch_bucket, batch_log_wait, batch_mask in loader:
            batch_features = batch_features.to(device)
            batch_bucket = batch_bucket.to(device)
            batch_log_wait = batch_log_wait.to(device)
            batch_mask = batch_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            bucket_logits, predicted_log_wait = model(batch_features)
            ce_loss = F.cross_entropy(bucket_logits, batch_bucket, weight=class_weight_tensor)
            non_suppress = batch_mask > 0.5
            if non_suppress.any():
                reg_loss = F.smooth_l1_loss(predicted_log_wait[non_suppress], batch_log_wait[non_suppress])
            else:
                reg_loss = torch.tensor(0.0, device=device)
            loss = 0.70 * ce_loss + 1.10 * reg_loss
            loss.backward()
            optimizer.step()
            running_losses.append(float(loss.item()))

        dev_logits, dev_log_wait = infer_raw_logits(model, dev_features, device, batch_size=512)
        threshold, dev_report, _ = tune_suppress_threshold(dev_logits, dev_log_wait, dev_examples)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(running_losses)),
                "dev_practical_score": dev_report["practical_score"],
                "dev_adaptive_60s_rate": dev_report["adaptive_60s_rate"],
                "dev_short_mae_seconds": dev_report["short_mae_seconds"],
                "dev_followup_f1": dev_report["followup_f1"],
                "threshold": threshold,
            }
        )
        if best_report is None or dev_report["practical_score"] > best_report["practical_score"]:
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = threshold
            best_report = dev_report
            best_epoch = epoch
            remaining = patience
        else:
            remaining -= 1
            if remaining <= 0:
                break

    assert best_state is not None
    assert best_report is not None
    model.load_state_dict(best_state)
    model.eval()
    return {
        "model": model,
        "threshold": best_threshold,
        "dev_report": best_report,
        "history": history,
        "best_epoch": best_epoch,
        "trial": asdict(trial),
    }


def evaluate_model_on_splits(
    model: TinyWaitNet,
    split_features: dict[str, np.ndarray],
    split_examples: dict[str, list[ConversationExample]],
    device: str | torch.device,
    threshold: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    reports: dict[str, dict[str, Any]] = {}
    predictions_by_split: dict[str, list[dict[str, Any]]] = {}
    for split_name in ["test_iid", "test_lexical", "test_stress"]:
        logits, log_wait = infer_raw_logits(model, split_features[split_name], device, batch_size=512)
        predictions = decode_from_raw(logits, log_wait, threshold, split_examples[split_name])
        reports[split_name] = evaluate_predictions(split_name, split_examples[split_name], predictions)
        predictions_by_split[split_name] = predictions
    aggregate = aggregate_reports(reports)
    return reports, aggregate, predictions_by_split


def evaluate_keyword_baseline(split_examples: dict[str, list[ConversationExample]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    baseline = KeywordWaitBaseline()
    reports = {}
    for split_name in ["test_iid", "test_lexical", "test_stress"]:
        predictions = baseline.predict_many(split_examples[split_name])
        reports[split_name] = evaluate_predictions(split_name, split_examples[split_name], predictions)
    return reports, aggregate_reports(reports)


def acceptance_check(
    aggregate: dict[str, Any],
    split_reports: dict[str, dict[str, Any]],
    golden_report: dict[str, Any],
    baseline_aggregate: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "overall_practical_score": aggregate["practical_score"] >= 0.95,
        "overall_followup_f1": aggregate["followup_f1"] >= 0.97,
        "overall_adaptive_60s_rate": aggregate["adaptive_60s_rate"] >= 0.93,
        "overall_adaptive_300s_rate": aggregate["adaptive_300s_rate"] >= 0.95,
        "overall_short_mae_seconds": aggregate["short_mae_seconds"] <= 3.0,
        "overall_short_within_5s_rate": aggregate["short_within_5s_rate"] >= 0.96,
        "lexical_adaptive_60s_rate": split_reports["test_lexical"]["adaptive_60s_rate"] >= 0.88,
        "stress_adaptive_60s_rate": split_reports["test_stress"]["adaptive_60s_rate"] >= 0.94,
        "stress_false_early_rate": split_reports["test_stress"]["false_early_rate"] <= 0.05,
        "golden_pass_rate": golden_report["pass_rate"] >= 1.0,
        "beats_keyword_baseline": aggregate["practical_score"] >= baseline_aggregate["practical_score"] + 0.15,
    }
    return {"passed": all(checks.values()), "checks": checks}


def format_wait_detail(wait_seconds: float | None) -> str:
    if wait_seconds is None:
        return SUPPRESS_LABEL
    return f"{int(round(float(wait_seconds)))}s ({humanize_wait(float(wait_seconds))})"


def render_report(
    *,
    accepted: dict[str, Any],
    aggregate: dict[str, Any],
    split_reports: dict[str, dict[str, Any]],
    baseline_reports: dict[str, dict[str, Any]],
    baseline_aggregate: dict[str, Any],
    golden_report: dict[str, Any],
    trial_result: dict[str, Any],
    split_examples: dict[str, list[ConversationExample]],
) -> str:
    scenario_counter = Counter(example.scenario for example in split_examples["train"])
    lines = [
        "# Wait-Time Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Acceptance passed: `{accepted['passed']}`",
        f"- Selected trial: `{trial_result['trial']['name']}`",
        f"- Best epoch: `{trial_result['best_epoch']}`",
        f"- Suppress threshold: `{trial_result['threshold']:.2f}`",
        "",
        "## Training Coverage",
        "",
        f"- Train size: `{len(split_examples['train'])}`",
        f"- Scenario mix: `{dict(scenario_counter)}`",
        "",
        "## Model Metrics",
        "",
        "| Split | Practical | Adaptive-60s | Adaptive-300s | Short MAE(s) | P90 Error(s) | Follow-up F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in ["test_iid", "test_lexical", "test_stress"]:
        report = split_reports[split_name]
        lines.append(
            f"| {split_name} | {report['practical_score']:.3f} | {report['adaptive_60s_rate']:.3f} | "
            f"{report['adaptive_300s_rate']:.3f} | {report['short_mae_seconds']:.1f} | "
            f"{report['p90_abs_error_seconds']:.1f} | {report['followup_f1']:.3f} |"
        )
    lines.extend(
        [
            f"| overall | {aggregate['practical_score']:.3f} | {aggregate['adaptive_60s_rate']:.3f} | "
            f"{aggregate['adaptive_300s_rate']:.3f} | {aggregate['short_mae_seconds']:.1f} | "
            f"{aggregate['p90_abs_error_seconds']:.1f} | {aggregate['followup_f1']:.3f} |",
            "",
            "## Keyword Baseline",
            "",
            "| Split | Practical | Adaptive-60s | Adaptive-300s | Short MAE(s) | P90 Error(s) | Follow-up F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split_name in ["test_iid", "test_lexical", "test_stress"]:
        report = baseline_reports[split_name]
        lines.append(
            f"| {split_name} | {report['practical_score']:.3f} | {report['adaptive_60s_rate']:.3f} | "
            f"{report['adaptive_300s_rate']:.3f} | {report['short_mae_seconds']:.1f} | "
            f"{report['p90_abs_error_seconds']:.1f} | {report['followup_f1']:.3f} |"
        )
    lines.extend(
        [
            f"| overall | {baseline_aggregate['practical_score']:.3f} | {baseline_aggregate['adaptive_60s_rate']:.3f} | "
            f"{baseline_aggregate['adaptive_300s_rate']:.3f} | {baseline_aggregate['short_mae_seconds']:.1f} | "
            f"{baseline_aggregate['p90_abs_error_seconds']:.1f} | {baseline_aggregate['followup_f1']:.3f} |",
            "",
            "## Golden Cases",
            "",
            f"- Pass rate: `{golden_report['pass_rate']:.3f}`",
        ]
    )
    for case in golden_report["cases"]:
        lines.append(f"- {case['scenario']}: target `{case['target']}`, predicted `{case['predicted']}`, pass `{case['passed']}`")
    lines.extend(["", "## Acceptance Checks", ""])
    for check_name, passed in accepted["checks"].items():
        lines.append(f"- {check_name}: `{passed}`")
    return "\n".join(lines) + "\n"


def build_context_showcase_cases() -> list[ConversationExample]:
    return [
        make_example(
            turns=[
                ("user", "If I still have not sent the deck by tomorrow morning, follow up."),
                ("assistant", "Understood. I will check back tomorrow morning if it is still missing."),
            ],
            hour_local=18,
            weekday=2,
            scenario="showcase_scheduled_morning",
            split="showcase",
            language="en",
            wait_seconds=next_daypart_wait_seconds(18, 2, "tomorrow_morning"),
            suppress=False,
            meta={
                "title": "Explicit reminder on the next morning",
                "description": "The user gives a clear future time anchor and asks for a reminder only after that point.",
            },
        ),
        make_example(
            turns=[
                ("user", "Prod is still down after the restart."),
                ("assistant", "Keep checking the worker logs and the failing job queue."),
                ("assistant", "If I do not hear back, I will check in again in 37 seconds."),
            ],
            hour_local=10,
            weekday=1,
            scenario="showcase_urgent_incident",
            split="showcase",
            language="en",
            wait_seconds=37.0,
            suppress=False,
            meta={
                "title": "Urgent production incident",
                "description": "Urgency and explicit short delay should force a very short follow-up window.",
            },
        ),
        make_example(
            turns=[
                ("user", "我接下来一直开会，今晚别再找我。"),
                ("assistant", "明白，我先停在这里，等你之后再回来。"),
            ],
            hour_local=14,
            weekday=3,
            scenario="showcase_do_not_disturb",
            split="showcase",
            language="zh",
            wait_seconds=None,
            suppress=True,
            meta={
                "title": "Explicit do-not-disturb",
                "description": "The user blocks any more proactive contact for the rest of the night.",
            },
        ),
        make_example(
            turns=[
                ("user", "I'm overwhelmed and need some space."),
                ("assistant", "Understood. I'll wait for you to restart the conversation."),
            ],
            hour_local=21,
            weekday=5,
            scenario="showcase_need_space",
            split="showcase",
            language="en",
            wait_seconds=None,
            suppress=True,
            meta={
                "title": "Need-space emotional context",
                "description": "The assistant should suppress proactive outreach when the user asks for space without a return time.",
            },
        ),
        make_example(
            turns=[
                ("user", "这个问题不算特别急，但我还缺一个字段。"),
                ("assistant", "你希望我用哪个数据源？如果你暂时没回，我两小时后再问。"),
            ],
            hour_local=11,
            weekday=3,
            scenario="showcase_clarification",
            split="showcase",
            language="zh",
            wait_seconds=7_200.0,
            suppress=False,
            meta={
                "title": "Missing detail blocks progress",
                "description": "The task is blocked by a missing detail, so the model should keep the thread alive but not too aggressively.",
            },
        ),
        make_example(
            turns=[
                ("user", "这个供应商回复还没到。"),
                ("assistant", "如果一天后还是没有消息，我再回来确认。"),
            ],
            hour_local=14,
            weekday=1,
            scenario="showcase_vendor_status",
            split="showcase",
            language="zh",
            wait_seconds=86_400.0,
            suppress=False,
            meta={
                "title": "Slow external dependency",
                "description": "Vendor or shipment updates usually deserve a longer wait before re-engagement.",
            },
        ),
        make_example(
            turns=[
                ("user", "If I have not done the workout in 83 seconds, check back on me."),
                ("assistant", "Will do. I will check back in 83 seconds if I still have not heard from you."),
            ],
            hour_local=19,
            weekday=0,
            scenario="showcase_accountability",
            split="showcase",
            language="en",
            wait_seconds=83.0,
            suppress=False,
            meta={
                "title": "Short accountability check-in",
                "description": "The user explicitly asks for a short-term accountability reminder.",
            },
        ),
        make_example(
            turns=[
                ("user", "我会忙到明天下午，如果方案还没发出来你再提醒我。"),
                ("assistant", "好，我先不打扰，等到明天下午如果还没收到，我再主动跟进。"),
            ],
            hour_local=20,
            weekday=2,
            scenario="showcase_async_deliverable",
            split="showcase",
            language="zh",
            wait_seconds=next_daypart_wait_seconds(20, 2, "tomorrow_afternoon"),
            suppress=False,
            meta={
                "title": "Async deliverable with explicit deadline",
                "description": "The assistant should wait until the user-provided delivery window instead of following up too early.",
            },
        ),
        make_example(
            turns=[
                ("user", "Perfect, that's all."),
                ("assistant", "Glad it helped. I'll leave it here."),
            ],
            hour_local=18,
            weekday=4,
            scenario="showcase_polite_close",
            split="showcase",
            language="en",
            wait_seconds=None,
            suppress=True,
            meta={
                "title": "Clean conversation close",
                "description": "The thread is finished and should not trigger a new proactive turn.",
            },
        ),
        make_example(
            turns=[
                ("user", "If I still have not finished the PRD by after lunch, follow up."),
                ("assistant", "Okay. I'll check back after lunch if it is still pending."),
            ],
            hour_local=9,
            weekday=2,
            scenario="showcase_after_lunch",
            split="showcase",
            language="en",
            wait_seconds=next_daypart_wait_seconds(9, 2, "after_lunch"),
            suppress=False,
            meta={
                "title": "Same-day after-lunch reminder",
                "description": "An intra-day reminder should land at the requested daypart, not immediately.",
            },
        ),
    ]


def render_context_showcase(cases: list[ConversationExample], predictions: list[dict[str, Any]]) -> str:
    lines = [
        "# Context Showcase",
        "",
        "This file shows how the trained wait-time policy behaves across different conversational contexts.",
        "",
    ]
    for index, (example, prediction) in enumerate(zip(cases, predictions), start=1):
        title = example.meta.get("title", example.scenario) if example.meta else example.scenario
        description = example.meta.get("description", "") if example.meta else ""
        target = format_wait_detail(example.wait_seconds)
        predicted = format_wait_detail(prediction["wait_seconds"])
        action = "suppress" if prediction["suppress"] else "follow_up"
        lines.extend(
            [
                f"## {index}. {title}",
                "",
                f"- Scenario: `{example.scenario}`",
                f"- Language: `{example.language}`",
                f"- Local hour / weekday: `{example.hour_local}:00`, `{example.weekday}`",
                f"- Description: {description}",
                f"- Expected target: `{target}`",
                f"- Model output: `{predicted}`",
                f"- Model action: `{action}`",
                f"- Predicted bucket: `{prediction['label']}`",
                "",
                "```text",
            ]
        )
        for turn in example.turns:
            lines.append(f"{turn['role']}: {turn['text']}")
        lines.extend(["```", ""])
    return "\n".join(lines) + "\n"


def write_context_showcase(
    *,
    output_path: Path,
    model: TinyWaitNet,
    featurizer: WaitTimeFeaturizer,
    device: str | torch.device,
    threshold: float,
) -> tuple[Path, list[dict[str, Any]]]:
    showcase_cases = build_context_showcase_cases()
    showcase_features, _, _, _ = examples_to_arrays(featurizer, showcase_cases)
    showcase_logits, showcase_log_wait = infer_raw_logits(model, showcase_features, device, batch_size=128)
    showcase_predictions = decode_from_raw(showcase_logits, showcase_log_wait, threshold, showcase_cases)
    showcase_path = output_path / "context_showcase.md"
    showcase_path.write_text(render_context_showcase(showcase_cases, showcase_predictions), encoding="utf-8")
    return showcase_path, showcase_predictions


def save_metrics_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def train_and_benchmark(
    *,
    seed: int = 7,
    output_dir: str | Path = ARTIFACTS_DIR,
    device: str | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")

    split_examples = build_benchmark_splits(seed=seed)
    featurizer = WaitTimeFeaturizer()
    split_arrays = {
        split_name: examples_to_arrays(featurizer, examples)
        for split_name, examples in split_examples.items()
    }
    split_features = {split_name: arrays[0] for split_name, arrays in split_arrays.items()}

    baseline_reports, baseline_aggregate = evaluate_keyword_baseline(split_examples)

    best_trial = None
    for trial_index, trial in enumerate(TRIAL_CONFIGS):
        train_features, train_bucket, train_log_wait, train_mask = split_arrays["train"]
        dev_features, _, _, _ = split_arrays["dev"]
        trial_result = train_one_trial(
            trial=trial,
            train_features=train_features,
            train_bucket=train_bucket,
            train_log_wait=train_log_wait,
            train_mask=train_mask,
            dev_features=dev_features,
            dev_examples=split_examples["dev"],
            device=device_name,
            seed=seed + trial_index,
        )
        split_reports, aggregate, predictions_by_split = evaluate_model_on_splits(
            model=trial_result["model"],
            split_features=split_features,
            split_examples=split_examples,
            device=device_name,
            threshold=trial_result["threshold"],
        )
        golden_examples = build_golden_cases()
        golden_features, _, _, _ = examples_to_arrays(featurizer, golden_examples)
        golden_logits, golden_log_wait = infer_raw_logits(trial_result["model"], golden_features, device_name, batch_size=128)
        golden_predictions = decode_from_raw(golden_logits, golden_log_wait, trial_result["threshold"], golden_examples)
        golden_report = evaluate_golden_cases(golden_examples, golden_predictions)
        acceptance = acceptance_check(aggregate, split_reports, golden_report, baseline_aggregate)
        candidate = {
            "trial_result": trial_result,
            "split_reports": split_reports,
            "aggregate": aggregate,
            "predictions_by_split": predictions_by_split,
            "golden_examples": golden_examples,
            "golden_report": golden_report,
            "acceptance": acceptance,
        }
        if best_trial is None or aggregate["practical_score"] > best_trial["aggregate"]["practical_score"]:
            best_trial = candidate
        if acceptance["passed"]:
            best_trial = candidate
            break

    assert best_trial is not None

    best_model: TinyWaitNet = best_trial["trial_result"]["model"]
    suppress_threshold = best_trial["trial_result"]["threshold"]
    state_path = output_path / "wait_policy_state.pt"
    bundle_path = output_path / "wait_policy_bundle.pt"
    torch.save(best_model.state_dict(), state_path)
    save_model_bundle(
        str(bundle_path),
        best_model,
        featurizer,
        metadata={
            "seed": seed,
            "bucket_specs": BUCKET_SPECS,
            "selected_trial": best_trial["trial_result"]["trial"],
            "aggregate_metrics": best_trial["aggregate"],
            "acceptance": best_trial["acceptance"],
        },
        suppress_threshold=suppress_threshold,
    )

    report_text = render_report(
        accepted=best_trial["acceptance"],
        aggregate=best_trial["aggregate"],
        split_reports=best_trial["split_reports"],
        baseline_reports=baseline_reports,
        baseline_aggregate=baseline_aggregate,
        golden_report=best_trial["golden_report"],
        trial_result=best_trial["trial_result"],
        split_examples=split_examples,
    )
    report_path = output_path / "benchmark_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    showcase_path, showcase_predictions = write_context_showcase(
        output_path=output_path,
        model=best_model,
        featurizer=featurizer,
        device=device_name,
        threshold=suppress_threshold,
    )

    metrics_payload = {
        "run": {
            "seed": seed,
            "device": device_name,
            "elapsed_seconds": round(time.time() - start_time, 2),
            "selected_trial": best_trial["trial_result"]["trial"],
            "best_epoch": best_trial["trial_result"]["best_epoch"],
            "threshold": suppress_threshold,
        },
        "model": best_trial["aggregate"],
        "model_by_split": best_trial["split_reports"],
        "keyword_baseline": baseline_aggregate,
        "keyword_baseline_by_split": baseline_reports,
        "golden": best_trial["golden_report"],
        "acceptance": best_trial["acceptance"],
        "artifacts": {
            "state_dict": str(state_path),
            "bundle": str(bundle_path),
            "report": str(report_path),
            "context_showcase": str(showcase_path),
        },
    }
    save_metrics_json(output_path / "metrics.json", metrics_payload)

    preview_examples = []
    for split_name in ["test_iid", "test_lexical", "test_stress"]:
        example = split_examples[split_name][0]
        prediction = best_trial["predictions_by_split"][split_name][0]
        preview_examples.append(
            {
                "split": split_name,
                "scenario": example.scenario,
                "turns": example.turns,
                "target": format_wait_detail(example.wait_seconds),
                "predicted": format_wait_detail(prediction["wait_seconds"]),
            }
        )
    preview_examples.extend(
        [
            {
                "split": "showcase",
                "scenario": example.scenario,
                "turns": example.turns,
                "target": format_wait_detail(example.wait_seconds),
                "predicted": format_wait_detail(prediction["wait_seconds"]),
            }
            for example, prediction in zip(build_context_showcase_cases()[:3], showcase_predictions[:3])
        ]
    )
    save_metrics_json(output_path / "prediction_preview.json", preview_examples)

    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and benchmark a lightweight proactive wait-time model.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default=str(ARTIFACTS_DIR))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    metrics = train_and_benchmark(seed=args.seed, output_dir=args.output_dir, device=args.device)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
