"""Minimal validation and selection for eval samples."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from .loaders import KnowledgeBase


def validate_candidate(sample: Dict[str, Any], kb: KnowledgeBase) -> Tuple[bool, List[str], Dict[str, float]]:
    reasons: List[str] = []
    rule_score = 1.0
    evidence_score = 1.0

    for field in ["id", "task_type", "question", "gold_answer", "evidence"]:
        if not sample.get(field):
            reasons.append(f"missing_{field}")
            rule_score -= 0.2

    evidence = sample.get("evidence") if isinstance(sample.get("evidence"), list) else []
    if not evidence:
        reasons.append("empty_evidence")
        evidence_score = 0.0
    else:
        for item in evidence:
            chunk_id = str(item.get("chunk_id", "")).strip()
            if not chunk_id or chunk_id not in kb.chunks:
                reasons.append(f"invalid_chunk:{chunk_id}")
                evidence_score -= 0.3

    rule_score = max(0.0, min(1.0, rule_score))
    evidence_score = max(0.0, min(1.0, evidence_score))
    accepted = not any(
        reason.startswith("missing_")
        or reason.startswith("invalid_chunk")
        or reason == "empty_evidence"
        for reason in reasons
    )
    return accepted, reasons, {"rule_score": rule_score, "evidence_score": evidence_score}


def select_accepted_samples(
    candidates: List[Dict[str, Any]],
    target_size: int,
    min_image_ratio: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    def sort_key(item: Dict[str, Any]) -> tuple[int, float, str]:
        return (
            -int(item.get("quality", {}).get("judge_score", 0)),
            -float(item.get("quality", {}).get("evidence_score", 0)),
            item.get("id", ""),
        )

    ordered = sorted(
        candidates,
        key=sort_key,
    )
    image_quota = min(
        target_size,
        math.ceil(target_size * max(0.0, min_image_ratio)),
    )
    image_candidates = [item for item in ordered if item.get("modality") == "image"]
    required_images = min(image_quota, len(image_candidates))
    selected_ids = {id(item) for item in image_candidates[:required_images]}
    for sample in ordered:
        if len(selected_ids) >= target_size:
            break
        selected_ids.add(id(sample))

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for sample in ordered:
        if id(sample) not in selected_ids:
            if target_size >= 0:
                rejected.append(mark_rejected(sample, ["over_target_size"]))
            continue
        sample["quality"]["status"] = "accepted"
        accepted.append(sample)
    return accepted, rejected


def mark_rejected(sample: Dict[str, Any], reasons: List[str]) -> Dict[str, Any]:
    sample = dict(sample)
    quality = dict(sample.get("quality", {}))
    quality["status"] = "rejected"
    quality["reasons"] = list(quality.get("reasons", [])) + reasons
    quality["reason"] = "|".join(str(r) for r in quality["reasons"])
    sample["quality"] = quality
    return sample


def apply_judge(sample: Dict[str, Any], judge: Dict[str, Any]) -> Dict[str, Any]:
    quality = dict(sample.get("quality", {}))
    total = (
        int(judge.get("answer_correctness", 0))
        + int(judge.get("evidence_consistency", 0))
        + int(judge.get("safety", 0))
        + int(judge.get("clarity", 0))
    )
    quality["judge"] = judge
    quality["judge_score"] = total
    sample["quality"] = quality
    return sample


def judge_passed(sample: Dict[str, Any]) -> bool:
    judge = sample.get("quality", {}).get("judge", {})
    return int(judge.get("evidence_consistency", 0)) >= 3
