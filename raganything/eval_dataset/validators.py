"""Minimal validation and selection for eval samples."""

from __future__ import annotations

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
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            -int(item.get("quality", {}).get("judge_score", 0)),
            -float(item.get("quality", {}).get("evidence_score", 0)),
            item.get("id", ""),
        ),
    )
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for idx, sample in enumerate(ordered):
        if idx >= target_size:
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

