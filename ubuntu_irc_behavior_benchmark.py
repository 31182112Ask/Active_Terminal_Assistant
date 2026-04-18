from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from wait_model import ConversationExample, humanize_wait, load_model_bundle, predict_wait_time, resolve_device


DEFAULT_BUNDLE = Path("artifacts") / "wait_policy_bundle.pt"
DEFAULT_OUTPUT_DIR = Path("artifacts")
DEFAULT_DATES = ["2026-02-04", "2026-02-05", "2026-02-06"]
DEFAULT_CHANNEL = "ubuntu"

ROW_PATTERN = re.compile(
    r'<tr id="t(?P<clock>\d{2}:\d{2})"><th class="nick"[^>]*>(?P<nick>.*?)</th>'
    r'<td class="text"[^>]*>(?P<text>.*?)</td><td class="time"><a[^>]*>(?P=clock)</a></td></tr>',
    flags=re.IGNORECASE | re.DOTALL,
)
TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
WORD_PATTERN = re.compile(r"[A-Za-z0-9_']+")
ADDRESS_PATTERN = re.compile(r"^(?P<nick>[A-Za-z0-9_\-\[\]\\`^{}|]+)\s*[:,]\s+")
BOT_NICKS = {"ubottu"}
SVG_WIDTH = 1280
SVG_HEIGHT = 900


@dataclass
class IrcTurn:
    timestamp: dt.datetime
    nick: str
    text: str
    source_url: str


def fetch_text(url: str, timeout_seconds: float = 30.0) -> str:
    with urlopen(url, timeout=timeout_seconds) as response:
        payload = response.read()
    return payload.decode("utf-8", errors="replace")


def log_url(date_text: str, channel: str) -> str:
    year, month, day = date_text.split("-")
    return f"https://irclogs.ubuntu.com/{year}/{month}/{day}/%23{channel}.html"


def clean_html_text(raw_text: str) -> str:
    text = raw_text.replace("<br />", "\n").replace("<br/>", "\n").replace("<br>", "\n")
    text = TAG_PATTERN.sub("", text)
    text = html.unescape(text).replace("\xa0", " ")
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def parse_log_html(html_text: str, *, date_text: str, source_url: str) -> list[IrcTurn]:
    base_date = dt.date.fromisoformat(date_text)
    turns: list[IrcTurn] = []
    for match in ROW_PATTERN.finditer(html_text):
        nick = clean_html_text(match.group("nick"))
        text = clean_html_text(match.group("text"))
        if not nick or not text:
            continue
        hour_text, minute_text = match.group("clock").split(":")
        timestamp = dt.datetime.combine(base_date, dt.time(hour=int(hour_text), minute=int(minute_text)))
        turns.append(IrcTurn(timestamp=timestamp, nick=nick, text=text, source_url=source_url))
    return turns


def is_meaningful_turn(turn: IrcTurn) -> bool:
    if turn.nick.lower() in BOT_NICKS:
        return False
    if len(turn.text) < 12:
        return False
    if turn.text.startswith("!"):
        return False
    words = WORD_PATTERN.findall(turn.text)
    if len(words) < 3 and "?" not in turn.text:
        return False
    return True


def merge_consecutive_bursts(turns: list[IrcTurn], *, max_gap_seconds: int) -> list[IrcTurn]:
    if not turns:
        return []
    merged: list[IrcTurn] = [turns[0]]
    for turn in turns[1:]:
        previous = merged[-1]
        gap_seconds = int((turn.timestamp - previous.timestamp).total_seconds())
        if turn.nick == previous.nick and 0 <= gap_seconds <= max_gap_seconds:
            merged[-1] = IrcTurn(
                timestamp=previous.timestamp,
                nick=previous.nick,
                text=f"{previous.text} {turn.text}".strip(),
                source_url=previous.source_url,
            )
            continue
        merged.append(turn)
    return merged


def adaptive_tolerance(wait_seconds: float) -> float:
    return max(120.0, min(7_200.0, wait_seconds * 0.50))


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def evenly_spaced_sample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    if limit <= 1:
        return rows[:1]
    indices = sorted({round(index * (len(rows) - 1) / (limit - 1)) for index in range(limit)})
    return [rows[index] for index in indices]


def rankdata(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = (cursor + end - 1) / 2.0 + 1.0
        for original_index, _ in ordered[cursor:end]:
            ranks[original_index] = average_rank
        cursor = end
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    rx = rankdata(xs)
    ry = rankdata(ys)
    mx = mean_or_none(rx)
    my = mean_or_none(ry)
    if mx is None or my is None:
        return None
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denom_x = sum((x - mx) ** 2 for x in rx)
    denom_y = sum((y - my) ** 2 for y in ry)
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return float(numerator / math.sqrt(denom_x * denom_y))


def format_wait(wait_seconds: float | None, suppress: bool) -> str:
    if suppress or wait_seconds is None:
        return "suppress"
    return humanize_wait(float(wait_seconds))


def build_example(context_turns: list[IrcTurn], focal_nick: str) -> ConversationExample:
    rendered_turns = []
    for turn in context_turns:
        role = "assistant" if turn.nick == focal_nick else "user"
        rendered_turns.append({"role": role, "text": f"{turn.nick}: {turn.text}"})
    anchor = context_turns[-1].timestamp
    return ConversationExample(
        turns=rendered_turns,
        hour_local=anchor.hour,
        weekday=anchor.weekday(),
        scenario="ubuntu_irc_behavior",
        split="real_behavior",
        language="en",
        meta={"focal_nick": focal_nick},
    )


def extract_addressee(text: str) -> str | None:
    match = ADDRESS_PATTERN.match(text)
    if not match:
        return None
    return match.group("nick")


def infer_partner(turns: list[IrcTurn], index: int) -> str | None:
    turn = turns[index]
    explicit = extract_addressee(turn.text)
    if explicit and explicit != turn.nick:
        return explicit
    for candidate in reversed(turns[max(0, index - 6) : index]):
        if candidate.nick != turn.nick and (turn.timestamp - candidate.timestamp).total_seconds() <= 20 * 60:
            return candidate.nick
    return None


def qualifies_as_thread_followup(
    turns: list[IrcTurn],
    *,
    anchor_index: int,
    candidate_index: int,
    focal_nick: str,
    partner_nick: str,
) -> bool:
    anchor_turn = turns[anchor_index]
    candidate_turn = turns[candidate_index]
    gap_seconds = int((candidate_turn.timestamp - anchor_turn.timestamp).total_seconds())
    between = turns[anchor_index + 1 : candidate_index]
    if not between and gap_seconds <= 10 * 60:
        return True
    if extract_addressee(candidate_turn.text) == partner_nick:
        return True
    return any(turn.nick == partner_nick for turn in between)


def extract_samples(
    turns: list[IrcTurn],
    *,
    context_turn_limit: int,
    observation_window_seconds: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not turns:
        return rows
    latest_anchor = turns[-1].timestamp - dt.timedelta(seconds=observation_window_seconds)
    for index, turn in enumerate(turns):
        if turn.timestamp > latest_anchor:
            continue
        if not is_meaningful_turn(turn):
            continue
        partner_nick = infer_partner(turns, index)
        if partner_nick is None or partner_nick == turn.nick:
            continue

        window_context = turns[max(0, index - context_turn_limit * 3) : index + 1]
        context = [item for item in window_context if item.nick in {turn.nick, partner_nick}]
        if len(context) < 2:
            continue
        if not any(item.nick == partner_nick for item in context[:-1]):
            continue
        previous_partner_turn = next((item for item in reversed(context[:-1]) if item.nick == partner_nick), None)
        if previous_partner_turn is None:
            continue
        if (turn.timestamp - previous_partner_turn.timestamp).total_seconds() > 20 * 60:
            continue

        next_same: IrcTurn | None = None
        for candidate_index, candidate in enumerate(turns[index + 1 :], start=index + 1):
            gap_seconds = int((candidate.timestamp - turn.timestamp).total_seconds())
            if gap_seconds > observation_window_seconds:
                break
            if candidate.nick == turn.nick and qualifies_as_thread_followup(
                turns,
                anchor_index=index,
                candidate_index=candidate_index,
                focal_nick=turn.nick,
                partner_nick=partner_nick,
            ):
                next_same = candidate
                break

        human_suppress = next_same is None
        human_wait_seconds = None if next_same is None else float((next_same.timestamp - turn.timestamp).total_seconds())
        rows.append(
            {
                "sample_id": f"{turn.timestamp.isoformat()}::{turn.nick}::{index}",
                "anchor_time": turn.timestamp.isoformat(timespec="minutes"),
                "focal_nick": turn.nick,
                "partner_nick": partner_nick,
                "context_turns": [
                    {"timestamp": item.timestamp.isoformat(timespec="minutes"), "nick": item.nick, "text": item.text}
                    for item in context
                ],
                "source_url": turn.source_url,
                "human_suppress": human_suppress,
                "human_wait_seconds": human_wait_seconds,
                "tolerance_seconds": None if human_wait_seconds is None else adaptive_tolerance(human_wait_seconds),
                "example": build_example(context, turn.nick),
            }
        )
    return rows


def chart_domain(rows: list[dict[str, Any]]) -> tuple[float, float]:
    values = [
        float(row["human_wait_seconds"])
        for row in rows
        if not row["human_suppress"] and row["human_wait_seconds"] is not None
    ]
    values.extend(
        float(row["pred_wait_seconds"])
        for row in rows
        if not row["pred_suppress"] and row["pred_wait_seconds"] is not None
    )
    lower = max(60.0, min(values) * 0.75) if values else 60.0
    upper = max(values) * 1.25 if values else 21_600.0
    lower = min(lower, 120.0)
    upper = max(upper, 86_400.0)
    return lower, upper


def log_position(value: float, lower: float, upper: float, pixel_min: float, pixel_max: float) -> float:
    log_lower = math.log10(lower)
    log_upper = math.log10(upper)
    log_value = math.log10(max(lower, min(value, upper)))
    ratio = (log_value - log_lower) / (log_upper - log_lower)
    return pixel_min + ratio * (pixel_max - pixel_min)


def tick_values(lower: float, upper: float) -> list[float]:
    candidates = [60.0, 300.0, 900.0, 1800.0, 3600.0, 7200.0, 21_600.0, 43_200.0, 86_400.0]
    return [candidate for candidate in candidates if lower <= candidate <= upper]


def compare_predictions(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compared: list[dict[str, Any]] = []
    for row, prediction in zip(rows, predictions):
        human_wait_seconds = row["human_wait_seconds"]
        pred_wait_seconds = prediction.get("wait_seconds")
        abs_error_seconds = None
        if human_wait_seconds is not None and pred_wait_seconds is not None and not prediction["suppress"]:
            abs_error_seconds = abs(float(pred_wait_seconds) - float(human_wait_seconds))
        within_tolerance = row["human_suppress"] == bool(prediction["suppress"])
        if within_tolerance and not row["human_suppress"]:
            within_tolerance = abs_error_seconds is not None and abs_error_seconds <= float(row["tolerance_seconds"])
        compared.append(
            {
                "sample_id": row["sample_id"],
                "anchor_time": row["anchor_time"],
                "focal_nick": row["focal_nick"],
                "partner_nick": row["partner_nick"],
                "source_url": row["source_url"],
                "context_turns": row["context_turns"],
                "human_suppress": row["human_suppress"],
                "human_wait_seconds": human_wait_seconds,
                "human_wait_label": format_wait(human_wait_seconds, row["human_suppress"]),
                "tolerance_seconds": row["tolerance_seconds"],
                "pred_suppress": bool(prediction["suppress"]),
                "pred_wait_seconds": pred_wait_seconds,
                "pred_wait_label": format_wait(pred_wait_seconds, bool(prediction["suppress"])),
                "pred_bucket": prediction.get("label"),
                "abs_error_seconds": abs_error_seconds,
                "within_tolerance": bool(within_tolerance),
            }
        )
    return compared


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    followup_rows = [row for row in rows if not row["human_suppress"]]
    suppress_rows = [row for row in rows if row["human_suppress"]]
    comparable_followups = [row for row in followup_rows if not row["pred_suppress"] and row["abs_error_seconds"] is not None]
    errors = [float(row["abs_error_seconds"]) for row in comparable_followups]
    human_waits = [float(row["human_wait_seconds"]) for row in comparable_followups]
    pred_waits = [float(row["pred_wait_seconds"]) for row in comparable_followups]
    overall_within = sum(1 for row in rows if row["within_tolerance"])
    followup_within = sum(1 for row in followup_rows if row["within_tolerance"])
    suppress_within = sum(1 for row in suppress_rows if row["within_tolerance"])
    return {
        "total_cases": len(rows),
        "followup_cases": len(followup_rows),
        "suppress_cases": len(suppress_rows),
        "overall_within_tolerance_rate": overall_within / len(rows) if rows else 0.0,
        "followup_within_tolerance_rate": followup_within / len(followup_rows) if followup_rows else 0.0,
        "suppress_agreement_rate": suppress_within / len(suppress_rows) if suppress_rows else 0.0,
        "followup_detection_rate": sum(1 for row in followup_rows if not row["pred_suppress"]) / len(followup_rows) if followup_rows else 0.0,
        "followup_mae_seconds": mean_or_none(errors),
        "followup_median_abs_error_seconds": median_or_none(errors),
        "followup_spearman": spearman(human_waits, pred_waits),
        "confusion": {
            "human_followup_model_followup": sum(1 for row in rows if not row["human_suppress"] and not row["pred_suppress"]),
            "human_followup_model_suppress": sum(1 for row in rows if not row["human_suppress"] and row["pred_suppress"]),
            "human_suppress_model_followup": sum(1 for row in rows if row["human_suppress"] and not row["pred_suppress"]),
            "human_suppress_model_suppress": sum(1 for row in rows if row["human_suppress"] and row["pred_suppress"]),
        },
    }


def metric_card(x: int, y: int, width: int, height: int, title: str, value: str, subtitle: str) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="14" fill="#ffffff" stroke="#d9e2ec" stroke-width="1.2"/>',
            f'<text x="{x + 16}" y="{y + 28}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#52606d">{html.escape(title)}</text>',
            f'<text x="{x + 16}" y="{y + 62}" font-family="Segoe UI, Arial, sans-serif" font-size="25" font-weight="700" fill="#102a43">{html.escape(value)}</text>',
            f'<text x="{x + 16}" y="{y + 87}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#7b8794">{html.escape(subtitle)}</text>',
        ]
    )


def render_svg(rows: list[dict[str, Any]], summary: dict[str, Any], *, title_suffix: str) -> str:
    sampled_followups = evenly_spaced_sample(
        sorted([row for row in rows if not row["human_suppress"]], key=lambda item: float(item["human_wait_seconds"])),
        limit=72,
    )
    lower, upper = chart_domain(sampled_followups)
    plot_left = 90
    plot_top = 140
    plot_right = 760
    plot_bottom = 730
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        '<rect width="100%" height="100%" fill="#fcfcf8"/>',
        '<text x="60" y="58" font-family="Segoe UI, Arial, sans-serif" font-size="29" font-weight="700" fill="#1f2933">Ubuntu IRC Human Behavior Benchmark</text>',
        f'<text x="60" y="87" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#52606d">Real multi-party dialogue delays from official Ubuntu IRC logs. {html.escape(title_suffix)}</text>',
        f'<rect x="{plot_left - 20}" y="{plot_top - 26}" width="{plot_width + 40}" height="{plot_height + 58}" fill="#ffffff" stroke="#d9e2ec" stroke-width="1.4" rx="16"/>',
    ]

    for tick in tick_values(lower, upper):
        x = log_position(tick, lower, upper, plot_left, plot_right)
        y = log_position(tick, lower, upper, plot_bottom, plot_top)
        label = humanize_wait(tick)
        svg.append(f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_bottom}" stroke="#e9eef2" stroke-width="1"/>')
        svg.append(f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" stroke="#e9eef2" stroke-width="1"/>')
        svg.append(f'<text x="{x:.1f}" y="{plot_bottom + 30}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#7b8794">{html.escape(label)}</text>')
        svg.append(f'<text x="{plot_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#7b8794">{html.escape(label)}</text>')

    svg.append(f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_top}" stroke="#bcccdc" stroke-width="2" stroke-dasharray="8 6"/>')
    svg.append(f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#486581" stroke-width="1.4"/>')
    svg.append(f'<text x="{plot_left + plot_width / 2:.1f}" y="{plot_bottom + 54}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#334e68">Actual human next-turn delay</text>')
    svg.append(
        f'<text transform="translate({plot_left - 52},{plot_top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle" '
        'font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#334e68">Model predicted delay</text>'
    )

    for row in sampled_followups:
        x = log_position(float(row["human_wait_seconds"]), lower, upper, plot_left, plot_right)
        if row["pred_suppress"] or row["pred_wait_seconds"] is None:
            y = plot_top + 18
            marker = f'<path d="M {x - 6:.1f},{y - 6:.1f} L {x + 6:.1f},{y + 6:.1f} M {x + 6:.1f},{y - 6:.1f} L {x - 6:.1f},{y + 6:.1f}" stroke="#d64545" stroke-width="2.2"/>'
        else:
            y = log_position(float(row["pred_wait_seconds"]), lower, upper, plot_bottom, plot_top)
            fill = "#2f855a" if row["within_tolerance"] else "#d97706"
            marker = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.8" fill="{fill}" stroke="#ffffff" stroke-width="1.2"/>'
        tooltip = (
            f"{row['anchor_time']} | {row['focal_nick']} | human={row['human_wait_label']} | "
            f"model={row['pred_wait_label']} | within_tol={row['within_tolerance']}"
        )
        svg.append(f"<g>{marker}<title>{html.escape(tooltip)}</title></g>")

    cards_x = 820
    cards_y = 140
    card_w = 180
    card_h = 105
    svg.append(metric_card(cards_x, cards_y, card_w, card_h, "Overall agreement", f"{summary['overall_within_tolerance_rate'] * 100:.1f}%", f"{summary['total_cases']} real samples"))
    svg.append(metric_card(cards_x + 200, cards_y, card_w, card_h, "Follow-up within tol", f"{summary['followup_within_tolerance_rate'] * 100:.1f}%", f"{summary['followup_cases']} follow-up cases"))
    svg.append(metric_card(cards_x, cards_y + 126, card_w, card_h, "Suppress agreement", f"{summary['suppress_agreement_rate'] * 100:.1f}%", f"{summary['suppress_cases']} suppress cases"))
    mae_label = humanize_wait(summary["followup_mae_seconds"]) if summary["followup_mae_seconds"] is not None else "n/a"
    svg.append(metric_card(cards_x + 200, cards_y + 126, card_w, card_h, "Follow-up MAE", mae_label, "real human delay vs model"))
    spearman_label = f"{summary['followup_spearman']:.2f}" if summary["followup_spearman"] is not None else "n/a"
    svg.append(metric_card(cards_x, cards_y + 252, card_w, card_h, "Rank correlation", spearman_label, "Spearman on follow-up cases"))
    svg.append(metric_card(cards_x + 200, cards_y + 252, card_w, card_h, "Follow-up detected", f"{summary['followup_detection_rate'] * 100:.1f}%", "human follow-up cases not suppressed"))

    matrix_x = cards_x
    matrix_y = cards_y + 410
    cell_w = 132
    cell_h = 86
    svg.append(f'<text x="{matrix_x}" y="{matrix_y - 18}" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="600" fill="#102a43">Suppress confusion matrix</text>')
    svg.append(f'<text x="{matrix_x + 120}" y="{matrix_y - 42}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#52606d">Model</text>')
    svg.append(f'<text x="{matrix_x - 42}" y="{matrix_y + 86}" transform="rotate(-90 {matrix_x - 42},{matrix_y + 86})" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#52606d">Human</text>')
    matrix_cells = [
        ("follow-up", "follow-up", summary["confusion"]["human_followup_model_followup"], "#e8f5ee"),
        ("follow-up", "suppress", summary["confusion"]["human_followup_model_suppress"], "#fff4db"),
        ("suppress", "follow-up", summary["confusion"]["human_suppress_model_followup"], "#fde8e8"),
        ("suppress", "suppress", summary["confusion"]["human_suppress_model_suppress"], "#e6fffa"),
    ]
    for index, (row_label, col_label, value, fill) in enumerate(matrix_cells):
        row_index = index // 2
        col_index = index % 2
        x = matrix_x + col_index * cell_w
        y = matrix_y + row_index * cell_h
        if row_index == 0:
            svg.append(f'<text x="{x + (cell_w - 10) / 2:.1f}" y="{matrix_y - 10}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#52606d">{html.escape(col_label)}</text>')
        if col_index == 0:
            svg.append(f'<text x="{matrix_x - 12}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#52606d">{html.escape(row_label)}</text>')
        svg.append(f'<rect x="{x}" y="{y}" width="{cell_w - 10}" height="{cell_h - 10}" rx="12" fill="{fill}" stroke="#d9e2ec" stroke-width="1"/>')
        svg.append(f'<text x="{x + (cell_w - 10) / 2:.1f}" y="{y + 48:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#102a43">{value}</text>')

    hardest_rows = sorted(
        [row for row in rows if not row["within_tolerance"]],
        key=lambda item: float(item["abs_error_seconds"] or 0.0),
        reverse=True,
    )[:5]
    legend_y = 810
    svg.append(f'<text x="60" y="{legend_y - 20}" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="600" fill="#102a43">Largest misses</text>')
    for index, row in enumerate(hardest_rows, start=1):
        error_label = "suppress mismatch" if row["abs_error_seconds"] is None else humanize_wait(float(row["abs_error_seconds"]))
        svg.append(
            f'<text x="60" y="{legend_y + index * 18:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#334e68">'
            f'{index}. {html.escape(row["anchor_time"])} | {html.escape(row["focal_nick"])} | human={html.escape(row["human_wait_label"])} | '
            f'model={html.escape(row["pred_wait_label"])} | error={html.escape(error_label)}</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def render_report(rows: list[dict[str, Any]], summary: dict[str, Any], *, dates: list[str], channel: str, chart_name: str) -> str:
    hard_rows = sorted(
        [row for row in rows if not row["within_tolerance"]],
        key=lambda item: float(item["abs_error_seconds"] or 0.0),
        reverse=True,
    )[:12]
    lines = [
        "# Ubuntu IRC Human Behavior Benchmark",
        "",
        "This report compares the current wait-policy model against real timestamped human dialogue from the official Ubuntu IRC logs.",
        "",
        "## Label Definition",
        "",
        "- Data source: official Ubuntu IRC HTML logs",
        f"- Channel: `#{channel}`",
        f"- Dates: `{', '.join(dates)}`",
        "- Timestamp resolution: minute-level, as provided by the public logs",
        "- Focal speaker: the nick who produced the anchor turn",
        "- Positive label: delay until that same speaker speaks again within the observation window",
        "- Suppress label: no same-speaker next turn within the observation window",
        "- Role mapping for the current model: focal speaker -> `assistant`, everyone else -> `user`",
        "",
        "This is a behavior-comparison benchmark. It measures how well the model matches actual human next-turn timing, not an explicit optimal-policy label.",
        "",
        "## Aggregate Metrics",
        "",
        f"- Total cases: `{summary['total_cases']}`",
        f"- Follow-up cases: `{summary['followup_cases']}`",
        f"- Suppress cases: `{summary['suppress_cases']}`",
        f"- Overall within-tolerance rate: `{summary['overall_within_tolerance_rate']:.4f}`",
        f"- Follow-up within-tolerance rate: `{summary['followup_within_tolerance_rate']:.4f}`",
        f"- Suppress agreement rate: `{summary['suppress_agreement_rate']:.4f}`",
        f"- Follow-up detection rate: `{summary['followup_detection_rate']:.4f}`",
        f"- Follow-up MAE: `{summary['followup_mae_seconds']:.1f}s`" if summary["followup_mae_seconds"] is not None else "- Follow-up MAE: `n/a`",
        f"- Follow-up median absolute error: `{summary['followup_median_abs_error_seconds']:.1f}s`" if summary["followup_median_abs_error_seconds"] is not None else "- Follow-up median absolute error: `n/a`",
        f"- Follow-up Spearman rank correlation: `{summary['followup_spearman']:.4f}`" if summary["followup_spearman"] is not None else "- Follow-up Spearman rank correlation: `n/a`",
        "",
        "## Visualization",
        "",
        f"![Ubuntu IRC human behavior comparison]({chart_name})",
        "",
        "## Largest Misses",
        "",
        "| Anchor time | Nick | Human | Model | Error | Source |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in hard_rows:
        error_label = "suppress mismatch" if row["abs_error_seconds"] is None else humanize_wait(float(row["abs_error_seconds"]))
        lines.append(
            f"| `{row['anchor_time']}` | `{row['focal_nick']} -> {row['partner_nick']}` | `{row['human_wait_label']}` | "
            f"`{row['pred_wait_label']}` | `{error_label}` | `{row['source_url']}` |"
        )
    lines.extend(["", "## Sample Contexts", ""])
    for row in hard_rows[:6]:
        lines.extend(
            [
                f"### {row['anchor_time']} - {row['focal_nick']} to {row['partner_nick']}",
                "",
                f"- Human next-turn delay: `{row['human_wait_label']}`",
                f"- Model prediction: `{row['pred_wait_label']}`",
                f"- Source: `{row['source_url']}`",
                "",
                "```text",
            ]
        )
        for turn in row["context_turns"]:
            lines.append(f"{turn['timestamp']} {turn['nick']}: {turn['text']}")
        lines.extend(["```", ""])
    return "\n".join(lines) + "\n"


def run_benchmark(
    *,
    bundle_path: str | Path,
    output_dir: str | Path,
    dates: list[str],
    channel: str,
    device: str | None,
    context_turn_limit: int,
    observation_window_minutes: int,
    burst_merge_minutes: int,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fetched_logs: list[dict[str, str]] = []
    raw_turns: list[IrcTurn] = []
    for date_text in dates:
        source_url = log_url(date_text, channel)
        try:
            html_text = fetch_text(source_url)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            fetched_logs.append({"date": date_text, "url": source_url, "status": f"error: {exc}"})
            continue
        parsed_turns = parse_log_html(html_text, date_text=date_text, source_url=source_url)
        fetched_logs.append({"date": date_text, "url": source_url, "status": "ok", "rows": str(len(parsed_turns))})
        raw_turns.extend(parsed_turns)

    raw_turns.sort(key=lambda item: item.timestamp)
    merged_turns = merge_consecutive_bursts(raw_turns, max_gap_seconds=burst_merge_minutes * 60)
    sample_rows = extract_samples(
        merged_turns,
        context_turn_limit=context_turn_limit,
        observation_window_seconds=observation_window_minutes * 60,
    )
    examples = [row["example"] for row in sample_rows]

    resolved_device = resolve_device(device)
    model, featurizer, metadata = load_model_bundle(bundle_path, device=resolved_device)
    suppress_threshold = float(metadata.get("suppress_threshold", 0.42))
    predictions = [
        predict_wait_time(model, featurizer, example, device=resolved_device, suppress_threshold=suppress_threshold)
        for example in examples
    ]
    compared_rows = compare_predictions(sample_rows, predictions)
    summary = summarize(compared_rows)

    chart_name = "ubuntu_irc_behavior_comparison.svg"
    report_name = "ubuntu_irc_behavior_benchmark.md"
    json_name = "ubuntu_irc_behavior_benchmark.json"
    samples_name = "ubuntu_irc_behavior_samples.json"

    title_suffix = (
        f"Dates: {', '.join(dates)}. Observation window: {observation_window_minutes} minutes. "
        f"Minute-resolution public timestamps."
    )
    (output_path / chart_name).write_text(render_svg(compared_rows, summary, title_suffix=title_suffix), encoding="utf-8")
    (output_path / report_name).write_text(
        render_report(compared_rows, summary, dates=dates, channel=channel, chart_name=chart_name),
        encoding="utf-8",
    )
    payload = {
        "source": {
            "channel": channel,
            "dates": dates,
            "logs": fetched_logs,
            "observation_window_minutes": observation_window_minutes,
            "burst_merge_minutes": burst_merge_minutes,
            "context_turn_limit": context_turn_limit,
        },
        "summary": summary,
        "device": str(resolved_device),
    }
    (output_path / json_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_path / samples_name).write_text(json.dumps(compared_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the wait-policy model against real human timing from Ubuntu IRC logs.")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)
    parser.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--context-turn-limit", type=int, default=8)
    parser.add_argument("--observation-window-minutes", type=int, default=360)
    parser.add_argument("--burst-merge-minutes", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_benchmark(
        bundle_path=args.bundle,
        output_dir=args.output_dir,
        dates=args.dates,
        channel=args.channel,
        device=args.device,
        context_turn_limit=args.context_turn_limit,
        observation_window_minutes=args.observation_window_minutes,
        burst_merge_minutes=args.burst_merge_minutes,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
