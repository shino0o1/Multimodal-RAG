"""Validation and final selection for eval samples."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from .loaders import KnowledgeBase


def validate_candidate(sample: Dict[str, Any], kb: KnowledgeBase) -> Tuple[bool, List[str], Dict[str, float]]:
    reasons: List[str] = []
    rule_score = 1.0
    evidence_score = 1.0

    for field in ["id", "task_type", "modality", "question", "gold_answer", "evidence"]:
        if not sample.get(field):
            reasons.append(f"missing_{field}")
            rule_score -= 0.15

    evidence = sample.get("evidence") if isinstance(sample.get("evidence"), list) else []
    if not evidence:
        reasons.append("empty_evidence")
        evidence_score = 0.0

    for item in evidence:
        chunk_id = str(item.get("chunk_id", ""))
        quote = str(item.get("quote", ""))
        chunk = kb.chunks.get(chunk_id)
        if not chunk:
            reasons.append(f"invalid_chunk:{chunk_id}")
            evidence_score -= 0.25
            continue
        content = str(chunk.get("content", ""))
        if quote and not fuzzy_contains(content, quote):
            reasons.append(f"quote_not_found:{chunk_id}")
            evidence_score -= 0.15

    entities = _string_list(sample.get("expected_entities"))
    if not any(entity in kb.entity_names for entity in entities):
        reasons.append("no_expected_entity_in_kg")
        rule_score -= 0.25

    answer = str(sample.get("gold_answer", ""))
    must_include = _string_list(sample.get("must_include"))
    missing_must = [term for term in must_include if term and term not in answer]
    if missing_must:
        reasons.append("missing_must_include:" + ",".join(missing_must[:3]))
        rule_score -= min(0.2, 0.05 * len(missing_must))

    must_not = _string_list(sample.get("must_not_include"))
    violated = [term for term in must_not if term and term in answer]
    if violated:
        reasons.append("violates_must_not:" + ",".join(violated[:3]))
        rule_score -= 0.25

    if sample.get("task_type") == "证据不足" and "无法确定" not in answer and "不能" not in answer:
        reasons.append("unknown_answer_not_uncertain")
        rule_score -= 0.25

    rule_score = max(0.0, min(1.0, rule_score))
    evidence_score = max(0.0, min(1.0, evidence_score))
    accepted = not reasons or (
        rule_score >= 0.65
        and evidence_score >= 0.65
        and not any(reason.startswith("invalid_chunk") for reason in reasons)
        and "empty_evidence" not in reasons
    )
    return accepted, reasons, {"rule_score": rule_score, "evidence_score": evidence_score}


def select_accepted_samples(
    candidates: List[Dict[str, Any]],
    target_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            -float(item.get("quality", {}).get("evidence_score", 0)),
            -float(item.get("quality", {}).get("rule_score", 0)),
            -int(item.get("quality", {}).get("judge_score", 0)),
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
    return (
        int(judge.get("evidence_consistency", 0)) >= 4
        and int(judge.get("safety", 0)) >= 4
        and int(sample.get("quality", {}).get("judge_score", 0)) >= 16
    )


def fuzzy_contains(content: str, quote: str, threshold: float = 0.75) -> bool:
    content_norm = _normalize_text(_clean_quote_like(content))
    quote_norm = _normalize_text(_clean_quote_like(quote))
    if not quote_norm:
        return True
    if quote_norm in content_norm:
        return True
    if len(quote_norm) > 80:
        head = quote_norm[:40]
        if head in content_norm:
            return True
        quote_norm = quote_norm[:80]
    return SequenceMatcher(None, content_norm, quote_norm).quick_ratio() >= threshold


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _clean_quote_like(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = re.sub(
        r"(?:图片路径|Image Path)\s*[:：]\s*\S+\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"图片内容分析\s*[:：]\s*", "", cleaned)
    cleaned = re.sub(r"标注\s*[:：]\s*None\s*", "", cleaned)
    cleaned = re.sub(r"脚注\s*[:：]\s*None\s*", "", cleaned)
    return cleaned


def _string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
