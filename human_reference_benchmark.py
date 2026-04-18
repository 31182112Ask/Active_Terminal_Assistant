from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any

from wait_model import ConversationExample, humanize_wait, load_model_bundle, predict_wait_time, resolve_device


DEFAULT_BUNDLE = Path("artifacts") / "wait_policy_bundle.pt"
DEFAULT_CASES = Path("human_reference_cases.json")
DEFAULT_OUTPUT_DIR = Path("artifacts")
SVG_WIDTH = 1200
SVG_HEIGHT = 760


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_example(case: dict[str, Any]) -> ConversationExample:
    return ConversationExample(
        turns=[{"role": turn["role"], "text": turn["text"]} for turn in case["turns"]],
        hour_local=int(case["hour_local"]),
        weekday=int(case["weekday"]),
        scenario=str(case["id"]),
        split="human_reference",
        language=str(case.get("language", "en")),
        wait_seconds=case.get("human_wait_seconds"),
        suppress=bool(case["human_suppress"]),
        meta={
            "title": case["title"],
            "category": case["category"],
            "rationale": case["rationale"],
            "tolerance_seconds": float(case.get("tolerance_seconds", 0.0)),
        },
    )


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        avg_rank = (cursor + end - 1) / 2.0 + 1.0
        for original_index, _ in indexed[cursor:end]:
            ranks[original_index] = avg_rank
        cursor = end
    return ranks


def spearman_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    rx = rankdata(xs)
    ry = rankdata(ys)
    mx = safe_mean(rx)
    my = safe_mean(ry)
    if mx is None or my is None:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    var_x = sum((x - mx) ** 2 for x in rx)
    var_y = sum((y - my) ** 2 for y in ry)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return float(cov / math.sqrt(var_x * var_y))


def format_wait(wait_seconds: float | None, suppress: bool) -> str:
    if suppress or wait_seconds is None:
        return "suppress"
    return humanize_wait(float(wait_seconds))


def compare_predictions(
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (case, prediction) in enumerate(zip(cases, predictions), start=1):
        human_suppress = bool(case["human_suppress"])
        pred_suppress = bool(prediction["suppress"])
        human_wait = case.get("human_wait_seconds")
        pred_wait = prediction.get("wait_seconds")
        tolerance = float(case.get("tolerance_seconds", 0.0))
        abs_error = None
        if not human_suppress and not pred_suppress and human_wait is not None and pred_wait is not None:
            abs_error = abs(float(pred_wait) - float(human_wait))
        within_tolerance = human_suppress == pred_suppress
        if within_tolerance and not human_suppress:
            within_tolerance = abs_error is not None and abs_error <= tolerance
        rows.append(
            {
                "index": index,
                "id": case["id"],
                "title": case["title"],
                "category": case["category"],
                "hour_local": int(case["hour_local"]),
                "weekday": int(case["weekday"]),
                "human_suppress": human_suppress,
                "pred_suppress": pred_suppress,
                "human_wait_seconds": human_wait,
                "pred_wait_seconds": pred_wait,
                "human_wait_label": format_wait(human_wait, human_suppress),
                "pred_wait_label": format_wait(pred_wait, pred_suppress),
                "tolerance_seconds": tolerance,
                "abs_error_seconds": abs_error,
                "within_tolerance": bool(within_tolerance),
                "rationale": case["rationale"],
                "turns": case["turns"],
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    followup_rows = [row for row in rows if not row["human_suppress"]]
    suppress_rows = [row for row in rows if row["human_suppress"]]
    comparable_followups = [row for row in followup_rows if not row["pred_suppress"] and row["abs_error_seconds"] is not None]
    followup_errors = [float(row["abs_error_seconds"]) for row in comparable_followups]
    followup_within = sum(1 for row in followup_rows if row["within_tolerance"])
    suppress_within = sum(1 for row in suppress_rows if row["within_tolerance"])
    followup_detection = sum(1 for row in followup_rows if not row["pred_suppress"])
    overall_within = sum(1 for row in rows if row["within_tolerance"])

    human_waits = [float(row["human_wait_seconds"]) for row in comparable_followups]
    pred_waits = [float(row["pred_wait_seconds"]) for row in comparable_followups]

    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in followup_rows:
        category_rows[str(row["category"])].append(row)

    category_metrics = []
    for category, items in sorted(category_rows.items()):
        model_waits = [
            float(item["pred_wait_seconds"])
            for item in items
            if not item["pred_suppress"] and item["pred_wait_seconds"] is not None
        ]
        human_wait_values = [float(item["human_wait_seconds"]) for item in items if item["human_wait_seconds"] is not None]
        penalized_errors = []
        for item in items:
            if item["abs_error_seconds"] is not None:
                penalized_errors.append(float(item["abs_error_seconds"]))
            elif item["pred_suppress"] and item["human_wait_seconds"] is not None:
                penalized_errors.append(float(item["human_wait_seconds"]))
        category_metrics.append(
            {
                "category": category,
                "count": len(items),
                "human_mean_seconds": safe_mean(human_wait_values),
                "model_mean_seconds": safe_mean(model_waits),
                "mae_seconds": safe_mean(penalized_errors),
                "within_tolerance_rate": sum(1 for item in items if item["within_tolerance"]) / len(items),
            }
        )

    summary = {
        "total_cases": len(rows),
        "followup_cases": len(followup_rows),
        "suppress_cases": len(suppress_rows),
        "overall_within_tolerance_rate": overall_within / len(rows) if rows else 0.0,
        "followup_within_tolerance_rate": followup_within / len(followup_rows) if followup_rows else 0.0,
        "suppress_agreement_rate": suppress_within / len(suppress_rows) if suppress_rows else 0.0,
        "followup_detection_rate": followup_detection / len(followup_rows) if followup_rows else 0.0,
        "followup_mae_seconds": safe_mean(followup_errors),
        "followup_median_abs_error_seconds": safe_median(followup_errors),
        "followup_spearman": spearman_correlation(human_waits, pred_waits),
        "confusion": {
            "human_followup_model_followup": sum(1 for row in rows if not row["human_suppress"] and not row["pred_suppress"]),
            "human_followup_model_suppress": sum(1 for row in rows if not row["human_suppress"] and row["pred_suppress"]),
            "human_suppress_model_followup": sum(1 for row in rows if row["human_suppress"] and not row["pred_suppress"]),
            "human_suppress_model_suppress": sum(1 for row in rows if row["human_suppress"] and row["pred_suppress"]),
        },
        "category_metrics": category_metrics,
    }
    return summary


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds:.1f}s"


def chart_domain(rows: list[dict[str, Any]]) -> tuple[float, float]:
    waits = [
        float(row["human_wait_seconds"])
        for row in rows
        if not row["human_suppress"] and row["human_wait_seconds"] is not None
    ]
    waits.extend(
        float(row["pred_wait_seconds"])
        for row in rows
        if not row["pred_suppress"] and row["pred_wait_seconds"] is not None
    )
    lower = max(10.0, min(waits) * 0.75) if waits else 10.0
    upper = max(waits) * 1.25 if waits else 100000.0
    lower = min(lower, 20.0)
    upper = max(upper, 172800.0)
    return lower, upper


def log_scale(value: float, lower: float, upper: float, pixel_min: float, pixel_max: float) -> float:
    log_lower = math.log10(lower)
    log_upper = math.log10(upper)
    log_value = math.log10(max(lower, min(value, upper)))
    ratio = (log_value - log_lower) / (log_upper - log_lower)
    return pixel_min + ratio * (pixel_max - pixel_min)


def tick_values(lower: float, upper: float) -> list[float]:
    candidates = [30.0, 60.0, 300.0, 1800.0, 7200.0, 14400.0, 54000.0, 86400.0, 172800.0]
    return [value for value in candidates if lower <= value <= upper]


def render_comparison_svg(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lower, upper = chart_domain(rows)
    scatter_left = 80
    scatter_top = 120
    scatter_width = 620
    scatter_height = 500
    summary_left = 760
    summary_top = 110
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        '<rect width="100%" height="100%" fill="#fcfcf8"/>',
        '<text x="60" y="58" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#1f2933">Human Reference vs Model Wait Time</text>',
        '<text x="60" y="86" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#52606d">Manual human-reference labels compared against the current wait-policy bundle.</text>',
        f'<rect x="{scatter_left}" y="{scatter_top}" width="{scatter_width}" height="{scatter_height}" fill="#ffffff" stroke="#d9e2ec" stroke-width="1.5" rx="16"/>',
        '<text x="102" y="152" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="600" fill="#102a43">Follow-up cases: human wait vs model wait</text>',
    ]

    plot_left = scatter_left + 58
    plot_top = scatter_top + 46
    plot_right = scatter_left + scatter_width - 24
    plot_bottom = scatter_top + scatter_height - 58
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    ticks = tick_values(lower, upper)
    for tick in ticks:
        x = log_scale(tick, lower, upper, plot_left, plot_right)
        y = log_scale(tick, lower, upper, plot_bottom, plot_top)
        label = humanize_wait(tick)
        svg.append(f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_bottom}" stroke="#e9eef2" stroke-width="1"/>')
        svg.append(f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" stroke="#e9eef2" stroke-width="1"/>')
        svg.append(f'<text x="{x:.1f}" y="{plot_bottom + 26}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#7b8794">{escape(label)}</text>')
        svg.append(f'<text x="{plot_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#7b8794">{escape(label)}</text>')

    svg.append(f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_top}" stroke="#bcccdc" stroke-width="2" stroke-dasharray="8 6"/>')
    svg.append(f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#486581" stroke-width="1.4"/>')
    svg.append(f'<text x="{plot_left + plot_width / 2:.1f}" y="{plot_bottom + 48}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#334e68">Human reference wait</text>')
    svg.append(
        f'<text transform="translate({plot_left - 50},{plot_top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle" '
        'font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#334e68">Model wait</text>'
    )

    followup_rows = [row for row in rows if not row["human_suppress"]]
    for row in followup_rows:
        x = log_scale(float(row["human_wait_seconds"]), lower, upper, plot_left, plot_right)
        if row["pred_suppress"] or row["pred_wait_seconds"] is None:
            y = plot_top + 16
            color = "#d64545"
            marker = f'<path d="M {x - 6:.1f},{y - 6:.1f} L {x + 6:.1f},{y + 6:.1f} M {x + 6:.1f},{y - 6:.1f} L {x - 6:.1f},{y + 6:.1f}" stroke="{color}" stroke-width="2.2"/>'
        else:
            y = log_scale(float(row["pred_wait_seconds"]), lower, upper, plot_bottom, plot_top)
            color = "#2f855a" if row["within_tolerance"] else "#d97706"
            marker = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{color}" stroke="#ffffff" stroke-width="1.4"/>'
        tooltip = (
            f"#{row['index']} {row['title']} | human={row['human_wait_label']} | "
            f"model={row['pred_wait_label']} | within_tol={row['within_tolerance']}"
        )
        svg.append(f'<g>{marker}<title>{escape(tooltip)}</title></g>')
        svg.append(
            f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="11" fill="#102a43">{row["index"]}</text>'
        )

    def metric_card(x: int, y: int, w: int, h: int, title: str, value: str, subtitle: str) -> str:
        return "\n".join(
            [
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#ffffff" stroke="#d9e2ec" stroke-width="1.2"/>',
                f'<text x="{x + 18}" y="{y + 28}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#52606d">{escape(title)}</text>',
                f'<text x="{x + 18}" y="{y + 62}" font-family="Segoe UI, Arial, sans-serif" font-size="26" font-weight="700" fill="#102a43">{escape(value)}</text>',
                f'<text x="{x + 18}" y="{y + 88}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#7b8794">{escape(subtitle)}</text>',
            ]
        )

    overall = f"{summary['overall_within_tolerance_rate'] * 100:.1f}%"
    followup = f"{summary['followup_within_tolerance_rate'] * 100:.1f}%"
    suppress = f"{summary['suppress_agreement_rate'] * 100:.1f}%"
    mae = humanize_wait(summary["followup_mae_seconds"]) if summary["followup_mae_seconds"] is not None else "n/a"
    spearman = f"{summary['followup_spearman']:.2f}" if summary["followup_spearman"] is not None else "n/a"

    svg.append(metric_card(summary_left, summary_top, 180, 104, "Overall agreement", overall, f"{summary['total_cases']} manual cases"))
    svg.append(metric_card(summary_left + 200, summary_top, 180, 104, "Follow-up within tol", followup, f"{summary['followup_cases']} follow-up cases"))
    svg.append(metric_card(summary_left, summary_top + 124, 180, 104, "Suppress agreement", suppress, f"{summary['suppress_cases']} suppress cases"))
    svg.append(metric_card(summary_left + 200, summary_top + 124, 180, 104, "Follow-up MAE", mae, "only cases with non-suppress predictions"))
    svg.append(metric_card(summary_left, summary_top + 248, 180, 104, "Rank correlation", spearman, "Spearman on follow-up cases"))
    svg.append(metric_card(summary_left + 200, summary_top + 248, 180, 104, "Follow-up detected", f"{summary['followup_detection_rate'] * 100:.1f}%", "human follow-up cases kept alive"))

    matrix_x = summary_left
    matrix_y = summary_top + 392
    cell_w = 110
    cell_h = 76
    svg.append(f'<text x="{matrix_x}" y="{matrix_y - 18}" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="600" fill="#102a43">Suppress confusion matrix</text>')
    svg.append(f'<text x="{matrix_x + 144}" y="{matrix_y - 44}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#52606d">Model</text>')
    svg.append(f'<text x="{matrix_x - 40}" y="{matrix_y + 72}" transform="rotate(-90 {matrix_x - 40},{matrix_y + 72})" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#52606d">Human</text>')
    labels = [
        ("follow-up", "follow-up", summary["confusion"]["human_followup_model_followup"], "#e8f5ee"),
        ("follow-up", "suppress", summary["confusion"]["human_followup_model_suppress"], "#fff4db"),
        ("suppress", "follow-up", summary["confusion"]["human_suppress_model_followup"], "#fde8e8"),
        ("suppress", "suppress", summary["confusion"]["human_suppress_model_suppress"], "#e6fffa"),
    ]
    for idx, (row_label, col_label, count, fill) in enumerate(labels):
        row_index = idx // 2
        col_index = idx % 2
        x = matrix_x + col_index * cell_w
        y = matrix_y + row_index * cell_h
        if row_index == 0:
            svg.append(f'<text x="{x + cell_w / 2:.1f}" y="{matrix_y - 8}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#52606d">{escape(col_label)}</text>')
        if col_index == 0:
            svg.append(f'<text x="{matrix_x - 12}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#52606d">{escape(row_label)}</text>')
        svg.append(f'<rect x="{x}" y="{y}" width="{cell_w - 10}" height="{cell_h - 10}" rx="12" fill="{fill}" stroke="#d9e2ec" stroke-width="1"/>')
        svg.append(f'<text x="{x + (cell_w - 10) / 2:.1f}" y="{y + 42:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#102a43">{count}</text>')

    bar_x = 60
    bar_y = 650
    bar_w = 1080
    svg.append(f'<text x="{bar_x}" y="{bar_y - 18}" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="600" fill="#102a43">Category mean absolute error</text>')
    category_metrics = [item for item in summary["category_metrics"] if item["mae_seconds"] is not None]
    if category_metrics:
        max_error = max(float(item["mae_seconds"]) for item in category_metrics) or 1.0
        label_w = 200
        gap = 14
        bar_height = 20
        for index, item in enumerate(category_metrics):
            row_y = bar_y + index * 28
            width = (float(item["mae_seconds"]) / max_error) * (bar_w - label_w - 160)
            svg.append(f'<text x="{bar_x}" y="{row_y + 15}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#334e68">{escape(item["category"])}</text>')
            svg.append(f'<rect x="{bar_x + label_w}" y="{row_y}" width="{bar_w - label_w - 160}" height="{bar_height}" rx="10" fill="#f0f4f8"/>')
            svg.append(f'<rect x="{bar_x + label_w}" y="{row_y}" width="{width:.1f}" height="{bar_height}" rx="10" fill="#2b6cb0"/>')
            value_label = humanize_wait(float(item["mae_seconds"])) if item["mae_seconds"] is not None else "n/a"
            svg.append(f'<text x="{bar_x + label_w + width + gap:.1f}" y="{row_y + 15}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#102a43">{escape(value_label)}</text>')

    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def render_report(rows: list[dict[str, Any]], summary: dict[str, Any], chart_name: str) -> str:
    lines = [
        "# Human Reference Benchmark",
        "",
        "This report compares the current wait-policy model against a manually labeled human-reference set.",
        "",
        "Important: this is a single-rater human-reference benchmark, not production user telemetry.",
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
        f"- Follow-up MAE: `{format_seconds(summary['followup_mae_seconds'])}`",
        f"- Follow-up median absolute error: `{format_seconds(summary['followup_median_abs_error_seconds'])}`",
        f"- Follow-up Spearman rank correlation: `{summary['followup_spearman']:.4f}`" if summary["followup_spearman"] is not None else "- Follow-up Spearman rank correlation: `n/a`",
        "",
        "## Visualization",
        "",
        f"![Human reference comparison]({chart_name})",
        "",
        "## Case Table",
        "",
        "| # | Case | Category | Human | Model | Tolerance | Within tolerance |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        tolerance_label = "n/a" if row["human_suppress"] else humanize_wait(float(row["tolerance_seconds"]))
        lines.append(
            f"| {row['index']} | {row['title']} | `{row['category']}` | `{row['human_wait_label']}` | "
            f"`{row['pred_wait_label']}` | `{tolerance_label}` | `{row['within_tolerance']}` |"
        )
    lines.extend(["", "## Dialogue Contexts", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['index']}. {row['title']}",
                "",
                f"- Category: `{row['category']}`",
                f"- Human reference: `{row['human_wait_label']}`",
                f"- Model prediction: `{row['pred_wait_label']}`",
                f"- Rationale: {row['rationale']}",
                "",
                "```text",
            ]
        )
        for turn in row["turns"]:
            lines.append(f"{turn['role']}: {turn['text']}")
        lines.extend(["```", ""])
    return "\n".join(lines) + "\n"


def run_benchmark(
    *,
    bundle_path: str | Path,
    cases_path: str | Path,
    output_dir: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cases = load_cases(cases_path)
    examples = [build_example(case) for case in cases]
    resolved_device = resolve_device(device)
    model, featurizer, metadata = load_model_bundle(bundle_path, device=resolved_device)
    suppress_threshold = float(metadata.get("suppress_threshold", 0.42))

    predictions = [
        predict_wait_time(model, featurizer, example, device=resolved_device, suppress_threshold=suppress_threshold)
        for example in examples
    ]
    rows = compare_predictions(cases, predictions)
    summary = summarize_rows(rows)

    chart_name = "human_reference_comparison.svg"
    report_name = "human_reference_benchmark.md"
    json_name = "human_reference_benchmark.json"
    svg_text = render_comparison_svg(rows, summary)
    report_text = render_report(rows, summary, chart_name)
    payload = {
        "summary": summary,
        "rows": rows,
        "bundle_path": str(bundle_path),
        "cases_path": str(cases_path),
        "device": str(resolved_device),
    }

    (output_path / chart_name).write_text(svg_text, encoding="utf-8")
    (output_path / report_name).write_text(report_text, encoding="utf-8")
    (output_path / json_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare wait-policy predictions against a manual human-reference benchmark.")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_benchmark(
        bundle_path=args.bundle,
        cases_path=args.cases,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
