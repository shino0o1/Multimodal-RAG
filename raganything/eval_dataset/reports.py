"""Output helpers and quality reports for eval dataset builds."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(
    path: str | Path,
    accepted: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    report = build_report(accepted, rejected, candidates)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def write_review_sheet(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "status",
        "task_type",
        "modality",
        "difficulty",
        "core_entity",
        "entity_type",
        "question",
        "gold_answer",
        "evidence_chunks",
        "judge_score",
        "reasons",
        "manual_decision",
        "manual_notes",
    ]
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            quality = row.get("quality", {}) or {}
            metadata = row.get("metadata", {}) or {}
            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "status": quality.get("status", ""),
                    "task_type": row.get("task_type", ""),
                    "modality": row.get("modality", ""),
                    "difficulty": row.get("difficulty", ""),
                    "core_entity": metadata.get("core_entity", ""),
                    "entity_type": metadata.get("entity_type", ""),
                    "question": row.get("question", ""),
                    "gold_answer": row.get("gold_answer", ""),
                    "evidence_chunks": "|".join(
                        str(item.get("chunk_id", "")) for item in row.get("evidence", [])
                    ),
                    "judge_score": quality.get("judge_score", ""),
                    "reasons": "|".join(str(item) for item in quality.get("reasons", [])),
                    "manual_decision": "",
                    "manual_notes": "",
                }
            )


def build_report(
    accepted: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "counts": {
            "candidates": len(candidates),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "acceptance_rate": round(len(accepted) / max(1, len(candidates)), 4),
        },
        "accepted_distribution": {
            "task_type": _counter(accepted, "task_type"),
            "modality": _counter(accepted, "modality"),
            "difficulty": _counter(accepted, "difficulty"),
            "entity_type": _counter_nested(accepted, "metadata", "entity_type"),
            "core_entity_top20": _counter_nested(accepted, "metadata", "core_entity", 20),
            "evidence_chunk_top20": _evidence_chunk_counter(accepted, 20),
        },
        "rejection_reasons": _rejection_reasons(rejected),
        "quality": {
            "avg_rule_score": _avg(accepted, "rule_score"),
            "avg_evidence_score": _avg(accepted, "evidence_score"),
            "avg_judge_score": _avg(accepted, "judge_score"),
        },
    }


def _counter(rows: List[Dict[str, Any]], field: str, limit: int | None = None) -> Dict[str, int]:
    counter = Counter(str(row.get(field, "")) for row in rows if row.get(field))
    return dict(counter.most_common(limit))


def _counter_nested(
    rows: List[Dict[str, Any]], parent: str, field: str, limit: int | None = None
) -> Dict[str, int]:
    counter = Counter(
        str((row.get(parent, {}) or {}).get(field, ""))
        for row in rows
        if (row.get(parent, {}) or {}).get(field)
    )
    return dict(counter.most_common(limit))


def _evidence_chunk_counter(rows: List[Dict[str, Any]], limit: int) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for item in row.get("evidence", []):
            chunk_id = str(item.get("chunk_id", ""))
            if chunk_id:
                counter[chunk_id] += 1
    return dict(counter.most_common(limit))


def _rejection_reasons(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for reason in (row.get("quality", {}) or {}).get("reasons", []):
            counter[str(reason)] += 1
    return dict(counter.most_common())


def _avg(rows: List[Dict[str, Any]], key: str) -> float:
    values: List[float] = []
    for row in rows:
        quality = row.get("quality", {}) or {}
        if key in quality:
            try:
                values.append(float(quality[key]))
            except Exception:
                pass
    return round(sum(values) / len(values), 4) if values else 0.0
