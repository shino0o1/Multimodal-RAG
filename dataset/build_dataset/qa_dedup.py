#!/usr/bin/env python3
"""Shared question deduplication helpers for synthetic QA generation."""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any


QUESTION_NOISE_RE = re.compile(r"[\s，。！？；：、,.!?;:\-—_（）()【】\[\]“”‘’\"']+")


def normalize_question(text: str) -> str:
    return QUESTION_NOISE_RE.sub("", str(text or "").lower()).strip()


def training_row_fields(row: dict[str, Any]) -> tuple[str, str, str, str]:
    metadata = row.get("metadata") or {}
    question = ""
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            question = str(message.get("content") or "").strip()
            break
    return (
        question,
        str(metadata.get("crop") or "").strip(),
        str(metadata.get("disease_or_pest") or "").strip(),
        str(metadata.get("category") or "").strip(),
    )


class QuestionDeduplicator:
    """Reject exact duplicates globally and near duplicates within one topic."""

    def __init__(self, near_threshold: float = 0.88) -> None:
        self.near_threshold = near_threshold
        self.exact_questions: set[str] = set()
        self.group_questions: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    def add(
        self,
        question: str,
        crop: str,
        target: str,
        category: str,
    ) -> str | None:
        normalized = normalize_question(question)
        if not normalized:
            return "empty_question"
        if normalized in self.exact_questions:
            return "exact_duplicate"

        group = (crop.strip(), target.strip(), category.strip())
        if self.near_threshold > 0:
            for previous in self.group_questions[group]:
                similarity = SequenceMatcher(None, normalized, previous).ratio()
                if similarity >= self.near_threshold:
                    return f"near_duplicate:{similarity:.3f}"

        self.exact_questions.add(normalized)
        self.group_questions[group].append(normalized)
        return None

    def add_training_row(self, row: dict[str, Any]) -> str | None:
        return self.add(*training_row_fields(row))
