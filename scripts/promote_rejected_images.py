#!/usr/bin/env python3
"""Promote manually reviewed image eval samples from rejected.jsonl to accepted.jsonl."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raganything.eval_dataset.reports import write_jsonl, write_report, write_review_sheet
from raganything.eval_dataset.validators import mark_rejected


DEFAULT_REASONS = {"judge_rejected"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move reviewed image QA rows from rejected.jsonl to accepted.jsonl."
    )
    parser.add_argument("--dataset-dir", default="eval_dataset_200")
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument(
        "--auto-image-judge-rejected",
        action="store_true",
        help="Promote all image rows rejected by the judge if question and answer are non-empty.",
    )
    parser.add_argument(
        "--include-over-target",
        action="store_true",
        help="Also promote image rows rejected only because they exceeded target size.",
    )
    parser.add_argument(
        "--keep-target-size",
        action="store_true",
        help="Demote accepted text rows so accepted.jsonl keeps its original length.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def backup(path: Path) -> None:
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(path, path.with_suffix(path.suffix + f".bak.{stamp}"))


def promotable(row: dict[str, Any], args: argparse.Namespace) -> bool:
    quality = row.get("quality", {}) or {}
    reason = str(quality.get("reason", ""))
    allowed_reasons = set(DEFAULT_REASONS)
    if args.include_over_target:
        allowed_reasons.add("over_target_size")
    return (
        row.get("modality") == "image"
        and reason in allowed_reasons
        and bool(str(row.get("question", "")).strip())
        and bool(str(row.get("gold_answer", "")).strip())
    )


def as_accepted(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    quality = dict(row.get("quality", {}) or {})
    quality["status"] = "accepted"
    quality["reason"] = ""
    quality.pop("reasons", None)
    row["quality"] = quality
    return row


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    accepted_path = dataset_dir / "accepted.jsonl"
    rejected_path = dataset_dir / "rejected.jsonl"
    candidates_path = dataset_dir / "candidates.jsonl"

    accepted = read_jsonl(accepted_path)
    rejected = read_jsonl(rejected_path)
    candidates = read_jsonl(candidates_path)
    original_target = len(accepted)

    explicit_ids = set(args.ids)
    promoted: list[dict[str, Any]] = []
    remaining_rejected: list[dict[str, Any]] = []
    for row in rejected:
        should_promote = row.get("id") in explicit_ids
        should_promote = should_promote or (args.auto_image_judge_rejected and promotable(row, args))
        if should_promote:
            if not str(row.get("question", "")).strip() or not str(row.get("gold_answer", "")).strip():
                raise ValueError(f"Cannot promote empty generated fields: {row.get('id')}")
            promoted.append(as_accepted(row))
        else:
            remaining_rejected.append(row)

    demoted: list[dict[str, Any]] = []
    if args.keep_target_size and promoted:
        protected_ids = {row.get("id") for row in promoted}
        sortable = [
            (idx, row)
            for idx, row in enumerate(accepted)
            if row.get("id") not in protected_ids and row.get("modality") != "image"
        ]
        sortable.sort(
            key=lambda item: (
                int((item[1].get("quality", {}) or {}).get("judge_score", 0)),
                -item[0],
            )
        )
        drop_indices = {idx for idx, _ in sortable[: len(promoted)]}
        if len(drop_indices) < len(promoted):
            raise ValueError("Not enough accepted text rows to demote while keeping target size.")
        kept_accepted = []
        for idx, row in enumerate(accepted):
            if idx in drop_indices:
                demoted.append(mark_rejected(row, ["over_target_size"]))
            else:
                kept_accepted.append(row)
        accepted = kept_accepted

    new_accepted = accepted + promoted
    new_rejected = remaining_rejected + demoted

    print(
        json.dumps(
            {
                "promoted": [row.get("id") for row in promoted],
                "demoted": [row.get("id") for row in demoted],
                "accepted_before": original_target,
                "accepted_after": len(new_accepted),
                "rejected_before": len(rejected),
                "rejected_after": len(new_rejected),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)
    for path in [accepted_path, rejected_path, dataset_dir / "review_sheet.csv", dataset_dir / "report.json"]:
        backup(path)
    write_jsonl(accepted_path, new_accepted)
    write_jsonl(rejected_path, new_rejected)
    write_review_sheet(dataset_dir / "review_sheet.csv", new_accepted + new_rejected)
    write_report(dataset_dir / "report.json", new_accepted, new_rejected, candidates)


if __name__ == "__main__":
    main()
