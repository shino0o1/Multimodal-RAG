"""Pipeline orchestration for local KB-backed eval dataset generation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .generators import EvalLLMClient, SampleGenerator, TASK_PREFIX
from .loaders import EvidencePack, build_evidence_packs, load_image_manifest, load_knowledge_base
from .reports import write_jsonl, write_report, write_review_sheet
from .validators import (
    apply_judge,
    judge_passed,
    mark_rejected,
    select_accepted_samples,
    validate_candidate,
)


DEFAULT_QUOTAS = {
    "病虫害诊断": 45,
    "防治建议": 45,
    "药剂推荐": 35,
    "症状识别": 35,
    "图像问答": 30,
    "证据不足": 10,
}


@dataclass
class EvalDatasetBuildConfig:
    rag_dir: str = "rag_storage_whole_book_gemini"
    output_dir: str = "eval_dataset"
    image_manifest: Optional[str] = None
    target_size: int = 200
    seed: int = 42
    candidate_multiplier: float = 1.5
    generator_model: Optional[str] = None
    judge_model: Optional[str] = None
    vision_model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class EvalDatasetBuilder:
    def __init__(self, config: EvalDatasetBuildConfig) -> None:
        self.config = config
        self.random = random.Random(config.seed)
        self.llm_client = EvalLLMClient(
            generator_model=config.generator_model,
            judge_model=config.judge_model,
            vision_model=config.vision_model,
            api_key=config.api_key,
            base_url=config.base_url,
            enabled=True,
        )
        self.generator = SampleGenerator(self.llm_client)

    def build(self) -> Dict[str, Any]:
        if not self.llm_client.enabled:
            raise RuntimeError(
                "Evaluation dataset generation requires an API key. "
                "Set OPENAI_API_KEY, pass --api-key, or configure api_key in pdf_rag_pipeline.py."
            )
        kb = load_knowledge_base(self.config.rag_dir)
        image_rows = load_image_manifest(self.config.image_manifest)
        packs = build_evidence_packs(kb, image_rows)
        quotas = self._scaled_quotas(bool(image_rows))

        candidates: List[Dict[str, Any]] = []
        preselection_rejected: List[Dict[str, Any]] = []
        counters = {task: 0 for task in quotas}

        for task_type, quota in quotas.items():
            if quota <= 0:
                continue
            task_packs = self._packs_for_task(packs, task_type)
            if not task_packs:
                continue
            needed = max(quota, math.ceil(quota * self.config.candidate_multiplier))
            for pack in self._sample_packs(task_packs, needed):
                counters[task_type] += 1
                sample_id = f"{TASK_PREFIX.get(task_type, 'qa')}_{counters[task_type]:04d}"
                if pack.modality == "image":
                    try:
                        match = self.generator.verify_image_match(pack)
                    except Exception as exc:
                        preselection_rejected.append(
                            _generation_error_sample(
                                sample_id,
                                task_type,
                                pack,
                                RuntimeError("image_verification_error:" + str(exc)[:180]),
                            )
                        )
                        continue
                    if (not match.get("match")) or float(match.get("confidence", 0.0)) < 0.5:
                        rejected = {
                            "id": sample_id,
                            "task_type": task_type,
                            "modality": "image",
                            "question": "",
                            "image_path": pack.image_path,
                            "gold_answer": "",
                            "evidence": [item.__dict__ for item in pack.evidence],
                            "metadata": {"core_entity": pack.core_entity},
                            "quality": {
                                "status": "rejected",
                                "reasons": ["image_target_mismatch"],
                                "reason": "image_target_mismatch",
                            },
                        }
                        preselection_rejected.append(rejected)
                        continue
                try:
                    sample = self.generator.generate(sample_id, task_type, pack)
                except Exception as exc:
                    preselection_rejected.append(
                        _generation_error_sample(sample_id, task_type, pack, exc)
                    )
                    continue
                ok, reasons, scores = validate_candidate(sample, kb)
                sample["quality"].update(scores)
                if not ok:
                    preselection_rejected.append(mark_rejected(sample, reasons))
                    continue

                try:
                    judged = apply_judge(sample, self.generator.judge(sample))
                except Exception as exc:
                    preselection_rejected.append(
                        mark_rejected(sample, ["judge_error:" + str(exc)[:200]])
                    )
                    continue
                if not judge_passed(judged):
                    preselection_rejected.append(mark_rejected(judged, ["judge_rejected"]))
                else:
                    candidates.append(_minimize_sample(judged))

        valid_candidates = [
            row
            for row in candidates
            if (row.get("quality", {}) or {}).get("judge")
            and judge_passed(row)
        ]
        accepted, overflow_rejected = select_accepted_samples(
            valid_candidates,
            target_size=self.config.target_size,
        )
        rejected = preselection_rejected + overflow_rejected

        out = Path(self.config.output_dir)
        write_jsonl(out / "candidates.jsonl", candidates)
        write_jsonl(out / "accepted.jsonl", accepted)
        write_jsonl(out / "rejected.jsonl", rejected)
        write_review_sheet(out / "review_sheet.csv", accepted + rejected)
        report = write_report(out / "report.json", accepted, rejected, candidates)
        return report

    def _scaled_quotas(self, has_images: bool) -> Dict[str, int]:
        base = dict(DEFAULT_QUOTAS)
        if not has_images:
            base["图像问答"] = 0

        total = sum(base.values())
        if total <= 0:
            return {}

        quotas = {
            task: int(round(self.config.target_size * count / total))
            for task, count in base.items()
        }
        diff = self.config.target_size - sum(quotas.values())
        ordered = [task for task, count in sorted(base.items(), key=lambda kv: -kv[1]) if quotas[task] > 0]
        idx = 0
        while diff != 0 and ordered:
            task = ordered[idx % len(ordered)]
            if diff > 0:
                quotas[task] += 1
                diff -= 1
            elif quotas[task] > 1:
                quotas[task] -= 1
                diff += 1
            idx += 1
        return quotas

    def _packs_for_task(self, packs: List[EvidencePack], task_type: str) -> List[EvidencePack]:
        if task_type == "图像问答":
            return [pack for pack in packs if pack.modality == "image"]
        text_packs = [pack for pack in packs if pack.modality == "text"]
        if task_type == "病虫害诊断":
            return [
                pack
                for pack in text_packs
                if pack.entity_type in {"病害", "虫害"}
                and _context_has(pack, ["症状", "危害", "发生"])
            ]
        if task_type == "防治建议":
            return [
                pack
                for pack in text_packs
                if pack.entity_type in {"病害", "虫害"}
                and _context_has(pack, ["防治", "清除", "轮作", "药剂"])
            ]
        if task_type == "药剂推荐":
            return [
                pack
                for pack in text_packs
                if pack.entity_type == "药剂"
                or _context_has(pack, ["药剂", "倍液", "乳油", "悬浮剂", "可湿性粉剂"])
            ]
        if task_type == "症状识别":
            return [
                pack
                for pack in text_packs
                if pack.entity_type in {"病害", "虫害"}
                and _context_has(pack, ["症状", "病斑", "叶片", "危害"])
            ]
        if task_type == "证据不足":
            return [pack for pack in text_packs if pack.entity_type in {"病害", "虫害", "药剂"}]
        return text_packs

    def _sample_packs(self, packs: List[EvidencePack], needed: int) -> List[EvidencePack]:
        if not packs:
            return []
        shuffled = list(packs)
        self.random.shuffle(shuffled)
        if needed <= len(shuffled):
            return shuffled[:needed]
        out: List[EvidencePack] = []
        while len(out) < needed:
            cycle = list(shuffled)
            self.random.shuffle(cycle)
            out.extend(cycle)
        return out[:needed]

    def _pack_with_visual_notes(
        self, pack: EvidencePack, visual_clues: str, image_match: Dict[str, Any]
    ) -> EvidencePack:
        return EvidencePack(
            pack_id=pack.pack_id,
            task_seed=pack.task_seed,
            core_entity=pack.core_entity,
            entity_type=pack.entity_type,
            expected_entities=pack.expected_entities,
            expected_relations=pack.expected_relations,
            evidence=pack.evidence,
            context=pack.context,
            modality=pack.modality,
            image_path=pack.image_path,
            image_labels=pack.image_labels,
            notes=(
                pack.notes
                + "\n视觉线索："
                + visual_clues
                + "\n图片匹配校验："
                + str(image_match)
            ).strip(),
        )


def _context_has(pack: EvidencePack, terms: List[str]) -> bool:
    text = pack.context + " " + " ".join(item.quote for item in pack.evidence)
    return any(term in text for term in terms)


def _generation_error_sample(
    sample_id: str, task_type: str, pack: EvidencePack, exc: Exception
) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "task_type": task_type,
        "modality": pack.modality,
        "question": "",
        "image_path": pack.image_path,
        "gold_answer": "",
        "evidence": [item.__dict__ for item in pack.evidence],
        "metadata": {
            "core_entity": pack.core_entity,
        },
        "quality": {
            "status": "rejected",
            "reasons": ["generation_error:" + str(exc)[:200]],
            "reason": "generation_error",
        },
    }


def _minimize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    quality = sample.get("quality", {}) or {}
    metadata = sample.get("metadata", {}) or {}
    minimized = {
        "id": sample.get("id", ""),
        "task_type": sample.get("task_type", ""),
        "modality": sample.get("modality", ""),
        "question": sample.get("question", ""),
        "image_path": sample.get("image_path"),
        "gold_answer": sample.get("gold_answer", ""),
        "evidence": [
            {
                "chunk_id": item.get("chunk_id", ""),
                "quote": item.get("quote", ""),
                "file_path": item.get("file_path", ""),
            }
            for item in sample.get("evidence", [])
        ],
        "metadata": {"core_entity": metadata.get("core_entity", "")},
        "quality": {
            "status": quality.get("status", "candidate"),
            "reason": quality.get("reason", ""),
            "rule_score": quality.get("rule_score", 0.0),
            "evidence_score": quality.get("evidence_score", 0.0),
            "judge_score": quality.get("judge_score", 0),
            "judge": quality.get("judge", {}),
        },
    }
    reasons = quality.get("reasons")
    if reasons:
        minimized["quality"]["reasons"] = list(reasons)
    return minimized
