#!/usr/bin/env python3
"""Remove exact and near-duplicate questions from train-format QA JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from qa_dedup import QuestionDeduplicator, training_row_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="输入的 messages 格式 QA JSONL。")
    parser.add_argument("--output", required=True, help="去重后的 QA JSONL。")
    parser.add_argument("--rejected-output", default=None, help="可选：保存被判定为重复的样本及原因。")
    parser.add_argument("--threshold", type=float, default=0.88, help="同主题近似重复阈值；设为 0 仅做完全去重。")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            clean_row = {key: value for key, value in row.items() if key != "_line_number"}
            handle.write(json.dumps(clean_row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    rejected_output_path = Path(args.rejected_output) if args.rejected_output else None

    deduplicator = QuestionDeduplicator(args.threshold)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for row in read_jsonl(input_path):
        question, crop, target, category = training_row_fields(row)
        reason = deduplicator.add(question, crop, target, category)
        if reason:
            reason_name = reason.split(":", 1)[0]
            reason_counts[reason_name] += 1
            rejected.append(
                {
                    **row,
                    "dedup_reject_reason": reason,
                    "dedup_source_line": row.get("_line_number"),
                }
            )
        else:
            kept.append(row)

    write_jsonl(output_path, kept)
    if rejected_output_path:
        write_jsonl(rejected_output_path, rejected)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "threshold": args.threshold,
        "total": len(kept) + len(rejected),
        "kept": len(kept),
        "rejected": len(rejected),
        "reason_counts": dict(reason_counts),
        "rejected_output": str(rejected_output_path) if rejected_output_path else None,
    }
    report_path = output_path.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
