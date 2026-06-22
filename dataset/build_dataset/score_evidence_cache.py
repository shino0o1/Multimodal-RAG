#!/usr/bin/env python3
"""Binary quality filter for evidence cache rows.

The policy is intentionally permissive: rows are marked unusable only when they
are clearly unsuitable for QA generation, such as search failure, no fetched
evidence snippets, missing target mention, or severe mojibake.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "dataset/build_dataset/evidence_cache/search_results.jsonl"
DEFAULT_OUTPUT_DIR = "dataset/build_dataset/evidence_cache/scored"

TARGET_ALIASES = {
    "小菜蛾": ["小菜蛾", "吊丝虫", "吊死虫", "菜蛾", "小青虫", "两头尖"],
    "黄曲条跳甲": ["黄曲条跳甲", "黄条跳甲", "跳甲"],
    "菜青虫": ["菜青虫", "青菜虫", "菜粉蝶", "菜白蝶", "白粉蝶"],
    "蚜虫": ["蚜虫", "菜蚜", "甘蓝蚜", "萝卜蚜", "桃蚜"],
    "菜螟": ["菜螟", "钻心虫", "钻蛀害虫"],
    "地蛆": ["地蛆", "根蛆", "种蝇", "萝卜蝇"],
    "根肿病": ["根肿病", "十字花科根肿病"],
    "霜霉病": ["霜霉病", "霜霉"],
    "软腐病": ["软腐病", "细菌性软腐病"],
    "病毒病": ["病毒病", "花叶病毒病"],
    "炭疽病": ["炭疽病", "炭疽"],
    "灰霉病": ["灰霉病", "灰霉"],
    "白锈病": ["白锈病", "白锈"],
    "菌核病": ["菌核病", "核盘菌"],
    "黑腐病": ["黑腐病", "细菌性黑腐病"],
    "黑斑病": ["黑斑病", "链格孢"],
}

CROP_ALIASES = {
    "大白菜": ["大白菜", "白菜"],
    "小白菜": ["小白菜", "青菜", "上海青"],
    "甘蓝": ["甘蓝", "包菜", "卷心菜", "圆白菜"],
    "花椰菜": ["花椰菜", "菜花", "花菜"],
    "青花菜": ["青花菜", "西兰花", "西蓝花"],
    "萝卜": ["萝卜", "白萝卜"],
    "油菜": ["油菜", "油菜薹"],
    "芥菜": ["芥菜", "雪里蕻", "儿菜"],
}

TASK_HINTS = {
    "symptom_diagnosis": ["症状", "为害", "危害", "病斑", "叶片", "根部", "识别", "发病"],
    "occurrence_condition": ["发生", "流行", "条件", "温度", "湿度", "雨", "原因", "规律"],
    "control_timing": ["时期", "适期", "防治", "发生期", "盛期", "前", "后"],
    "agronomic_control": ["农业防治", "轮作", "清园", "排水", "田间", "管理", "抗病"],
    "biological_control": ["生物", "天敌", "绿僵菌", "苏云金杆菌", "菌剂", "绿色防控"],
    "chemical_control": ["药剂", "用药", "喷雾", "防治", "农药", "施药"],
    "safety_correction": ["安全", "禁用", "注意", "间隔期", "药害", "蜜蜂", "残留"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-snippet-chars", type=int, default=80)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--require-target",
        dest="require_target",
        action="store_true",
        help="Require target or target alias to appear in a fetched result.",
    )
    target_group.add_argument(
        "--no-require-target",
        dest="require_target",
        action="store_false",
        help="Do not require target or target alias to appear in a fetched result.",
    )
    parser.set_defaults(require_target=True)

    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--require-task-hint",
        dest="require_task_hint",
        action="store_true",
        help="Require task-specific keywords in a fetched result.",
    )
    task_group.add_argument(
        "--no-require-task-hint",
        dest="require_task_hint",
        action="store_false",
        help="Do not require task-specific keywords in a fetched result.",
    )
    parser.set_defaults(require_task_hint=False)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def terms_for(value: str, aliases: dict[str, list[str]]) -> list[str]:
    value = str(value or "").strip()
    terms = aliases.get(value, [value])
    return [term for term in terms if term]


def text_of_result(result: dict[str, Any]) -> str:
    pieces = [
        result.get("title", ""),
        result.get("search_snippet", ""),
        result.get("page_title", ""),
        result.get("meta_description", ""),
    ]
    pieces.extend(result.get("evidence_snippets") or [])
    return "\n".join(str(piece or "") for piece in pieces)


def search_text_of_result(result: dict[str, Any]) -> str:
    return "\n".join(
        str(piece or "")
        for piece in [
            result.get("title", ""),
            result.get("search_snippet", ""),
        ]
    )


def contains_any(text: str, terms: list[str]) -> bool:
    return any(term and term in text for term in terms)


def mojibake_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = text.count("�") + text.count("����") * 2
    return bad / max(len(text), 1)


def snippet_chars(result: dict[str, Any]) -> int:
    return sum(len(str(snippet or "")) for snippet in result.get("evidence_snippets") or [])


def has_search_snippet_fallback(row: dict[str, Any], result: dict[str, Any]) -> bool:
    text = search_text_of_result(result)
    if not text.strip() or mojibake_ratio(text) > 0.01:
        return False
    crop_terms = terms_for(str(row.get("crop", "")), CROP_ALIASES)
    target_terms = terms_for(str(row.get("target", "")), TARGET_ALIASES)
    task_terms = TASK_HINTS.get(str(row.get("task_type", "")), [])
    has_crop = contains_any(text, crop_terms) or "十字花科" in text
    has_target = contains_any(text, target_terms)
    has_task = not task_terms or contains_any(text, task_terms)
    return has_crop and has_target and has_task


def score_result(row: dict[str, Any], result: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if result.get("fetch_status") != "ok":
        if has_search_snippet_fallback(row, result):
            return True, ["search_snippet_only", "fetch_failed"]
        return False, ["fetch_failed"]
    snippets = result.get("evidence_snippets") or []
    if not snippets:
        if has_search_snippet_fallback(row, result):
            return True, ["search_snippet_only", "missing_snippets"]
        return False, ["missing_snippets"]
    if snippet_chars(result) < args.min_snippet_chars:
        reasons.append("short_snippets")

    text = text_of_result(result)
    if mojibake_ratio(text) > 0.01 or text.count("�") >= 5:
        return False, ["mojibake"]

    crop_terms = terms_for(str(row.get("crop", "")), CROP_ALIASES)
    target_terms = terms_for(str(row.get("target", "")), TARGET_ALIASES)
    has_crop = contains_any(text, crop_terms) or "十字花科" in text
    has_target = contains_any(text, target_terms)
    if args.require_target and not has_target:
        return False, ["target_not_found"]
    if not has_crop:
        reasons.append("crop_not_found")

    task_type = str(row.get("task_type", ""))
    task_terms = TASK_HINTS.get(task_type, [])
    if task_terms and not contains_any(text, task_terms):
        if args.require_task_hint:
            return False, ["task_hint_not_found"]
        reasons.append("task_hint_not_found")

    return True, reasons


def classify_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    usable_results: list[dict[str, Any]] = []
    rejected_results: list[dict[str, Any]] = []
    result_reasons = Counter()

    if row.get("status") not in {"ok", "no_results"}:
        return {
            **row,
            "evidence_quality": "unusable",
            "quality_reasons": [row.get("status") or "search_error"],
            "usable_results": [],
        }

    for result in row.get("results") or []:
        ok, reasons = score_result(row, result, args)
        result_reasons.update(reasons or ["usable"])
        result_with_quality = dict(result)
        result_with_quality["result_quality_reasons"] = reasons
        if ok and "search_snippet_only" in reasons and not result_with_quality.get("evidence_snippets"):
            snippet = str(result_with_quality.get("search_snippet") or "").strip()
            result_with_quality["evidence_snippets"] = [snippet] if snippet else []
        if ok:
            usable_results.append(result_with_quality)
        else:
            rejected_results.append(result_with_quality)

    quality_reasons: list[str] = []
    if not row.get("results"):
        quality_reasons.append("no_search_results")
    if not usable_results:
        quality_reasons.append("no_usable_fetched_evidence")
    if result_reasons:
        quality_reasons.extend(
            reason for reason, _ in result_reasons.most_common() if reason != "usable"
        )

    output = dict(row)
    output["usable_results"] = usable_results
    output["rejected_result_count"] = len(rejected_results)
    output["evidence_quality"] = "usable" if usable_results else "unusable"
    output["quality_reasons"] = quality_reasons
    return output


def summarize(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    quality_counts = Counter(row.get("evidence_quality") for row in scored_rows)
    reason_counts = Counter(
        reason for row in scored_rows for reason in row.get("quality_reasons", [])
    )
    task_counts = Counter(
        (row.get("task_type"), row.get("evidence_quality")) for row in scored_rows
    )
    target_counts = Counter(
        (row.get("target"), row.get("evidence_quality")) for row in scored_rows
    )
    usable_result_counts = Counter(len(row.get("usable_results") or []) for row in scored_rows)
    return {
        "total": len(scored_rows),
        "quality_counts": dict(quality_counts),
        "usable_rate": round(quality_counts.get("usable", 0) / max(len(scored_rows), 1), 4),
        "reason_counts": dict(reason_counts.most_common()),
        "usable_result_counts": dict(sorted(usable_result_counts.items())),
        "task_quality_counts": {
            f"{task}|{quality}": count for (task, quality), count in task_counts.most_common()
        },
        "target_quality_counts": {
            f"{target}|{quality}": count for (target, quality), count in target_counts.most_common(50)
        },
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    rows = read_jsonl(input_path)
    scored = [classify_row(row, args) for row in rows]
    usable = [row for row in scored if row.get("evidence_quality") == "usable"]
    unusable = [row for row in scored if row.get("evidence_quality") == "unusable"]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "usable_evidence.jsonl", usable)
    write_jsonl(output_dir / "unusable_evidence.jsonl", unusable)
    write_jsonl(output_dir / "all_scored_evidence.jsonl", scored)

    report = summarize(scored)
    report["input"] = str(input_path)
    report["output_dir"] = str(output_dir)
    report["policy"] = {
        "binary": True,
        "bias": "usable unless clearly unusable",
        "min_snippet_chars": args.min_snippet_chars,
        "require_target": args.require_target,
        "require_task_hint": args.require_task_hint,
        "search_snippet_fallback": True,
        "short_snippets_hard_reject": False,
    }
    (output_dir / "evidence_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
