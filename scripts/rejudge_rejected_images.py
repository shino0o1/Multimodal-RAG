#!/usr/bin/env python3
"""Rejudge non-empty rejected image QA rows and promote passing rows."""

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

from raganything.eval_dataset.generators import EvalLLMClient, SampleGenerator
from raganything.eval_dataset.reports import write_jsonl, write_report, write_review_sheet
from raganything.eval_dataset.validators import apply_judge, judge_passed, mark_rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rejudge rejected image QA samples with the original image and promote passing rows."
    )
    parser.add_argument("--dataset-dir", default="eval_dataset_200")
    parser.add_argument("--image-manifest", default="raganything/eval_dataset/manifest.jsonl")
    parser.add_argument("--ids", nargs="*", default=[], help="Optional explicit rejected row ids to rejudge.")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--vision-model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def backup(path: Path) -> None:
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(path, path.with_suffix(path.suffix + f".bak.{stamp}"))


def load_manifest_by_path(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = [row for row in payload.get("items", []) if isinstance(row, dict)]
        elif isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, dict)]
    else:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return {str(row.get("image_path", "")).strip(): row for row in rows}


def image_match_metadata(match: dict[str, Any]) -> dict[str, Any]:
    return {
        "match": bool(match.get("match", False)),
        "confidence": float(match.get("confidence", 0.0) or 0.0),
        "visible_clues": list(match.get("visible_clues") or []),
        "reason": str(match.get("reason", "")),
    }


def is_rejudge_target(row: dict[str, Any], explicit_ids: set[str]) -> bool:
    if explicit_ids and row.get("id") not in explicit_ids:
        return False
    return (
        row.get("modality") == "image"
        and bool(str(row.get("question", "")).strip())
        and bool(str(row.get("gold_answer", "")).strip())
        and bool(str(row.get("image_path", "")).strip())
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
    manifest_by_path = load_manifest_by_path(Path(args.image_manifest))
    existing_accepted_ids = {row.get("id") for row in accepted}
    explicit_ids = set(args.ids)

    targets = [row for row in rejected if is_rejudge_target(row, explicit_ids)]
    if not targets:
        print(json.dumps({"rejudged": 0, "promoted": [], "still_rejected": []}, ensure_ascii=False, indent=2))
        return

    llm_client = EvalLLMClient(
        judge_model=args.judge_model,
        vision_model=args.vision_model,
        api_key=args.api_key,
        base_url=args.base_url,
        enabled=True,
    )
    if not llm_client.enabled:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY/API_KEY or pass --api-key.")
    generator = SampleGenerator(llm_client)

    rejudged_by_id: dict[str, dict[str, Any]] = {}
    promoted_ids: list[str] = []
    still_rejected_ids: list[str] = []
    errored: dict[str, str] = {}

    for row in targets:
        row_id = str(row.get("id", ""))
        try:
            row_for_judge = dict(row)
            manifest_row = manifest_by_path.get(str(row.get("image_path", "")).strip(), {})
            labels = manifest_row.get("labels") if isinstance(manifest_row.get("labels"), dict) else {}
            notes = str(manifest_row.get("notes", "") or "")
            match = llm_client.verify_image_match(str(row.get("image_path")), labels, notes)
            row_for_judge.setdefault("metadata", {})["image_match"] = image_match_metadata(match)
            if not match.get("match"):
                rejudged_by_id[row_id] = mark_rejected(
                    row_for_judge, ["image_target_mismatch_after_image_rejudge"]
                )
                still_rejected_ids.append(row_id)
                continue

            judged = apply_judge(row_for_judge, generator.judge(row_for_judge))
            if judge_passed(judged):
                rejudged_by_id[row_id] = as_accepted(judged)
                promoted_ids.append(row_id)
            else:
                rejudged_by_id[row_id] = mark_rejected(judged, ["judge_rejected_after_image_rejudge"])
                still_rejected_ids.append(row_id)
        except Exception as exc:
            errored[row_id] = str(exc)[:300]

    new_accepted = list(accepted)
    remaining_rejected: list[dict[str, Any]] = []
    for row in rejected:
        row_id = str(row.get("id", ""))
        updated = rejudged_by_id.get(row_id)
        if not updated:
            remaining_rejected.append(row)
            continue
        if row_id in promoted_ids:
            if row_id not in existing_accepted_ids:
                new_accepted.append(updated)
        else:
            remaining_rejected.append(updated)

    result = {
        "rejudged": len(targets),
        "promoted": promoted_ids,
        "still_rejected": still_rejected_ids,
        "errored": errored,
        "accepted_before": len(accepted),
        "accepted_after": len(new_accepted),
        "rejected_before": len(rejected),
        "rejected_after": len(remaining_rejected),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    for path in [accepted_path, rejected_path, dataset_dir / "review_sheet.csv", dataset_dir / "report.json"]:
        backup(path)
    write_jsonl(accepted_path, new_accepted)
    write_jsonl(rejected_path, remaining_rejected)
    write_review_sheet(dataset_dir / "review_sheet.csv", new_accepted + remaining_rejected)
    write_report(dataset_dir / "report.json", new_accepted, remaining_rejected, candidates)


if __name__ == "__main__":
    main()
