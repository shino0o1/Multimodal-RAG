"""LLM generators for local KB-backed eval samples."""

from __future__ import annotations

import base64
import ast
import asyncio
import inspect
import json
import mimetypes
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .loaders import EvidencePack


TASK_PREFIX = {
    "病虫害诊断": "diag",
    "防治建议": "control",
    "药剂推荐": "pesticide",
    "症状识别": "symptom",
    "图像问答": "image",
    "证据不足": "unknown",
}


class EvalLLMClient:
    """Small wrapper around LightRAG's OpenAI-compatible helper."""

    def __init__(
        self,
        generator_model: Optional[str] = None,
        judge_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        cfg = _load_rag_config_defaults()
        pipeline_secrets = _load_pipeline_api_defaults()
        self.generator_model = (
            generator_model
            or cfg.get("model_answer")
            or "gemini-2.5-flash"
        )
        self.judge_model = (
            judge_model
            or cfg.get("model_judge")
            or "gemini-3.1-pro-preview"
        )
        self.vision_model = (
            vision_model
            or cfg.get("model_vision")
            or self.generator_model
        )
        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("API_KEY")
            or pipeline_secrets.get("api_key")
        )
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or pipeline_secrets.get("base_url")
        )
        self.enabled = enabled and bool(self.api_key)

    def complete(self, prompt: str, system_prompt: str, model: Optional[str] = None) -> str:
        if not self.enabled:
            raise RuntimeError("LLM client disabled or missing API key")
        from lightrag.llm.openai import openai_complete_if_cache

        return _resolve_maybe_awaitable(
            openai_complete_if_cache(
                model or self.generator_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=[],
                api_key=self.api_key,
                base_url=self.base_url,
            )
        )

    def complete_json(
        self, prompt: str, system_prompt: str, model: Optional[str] = None
    ) -> Dict[str, Any]:
        raw = self.complete(prompt, system_prompt, model=model)
        return extract_json_object(raw)

    def describe_image(self, image_path: str, labels: Dict[str, Any], notes: str) -> str:
        if not self.enabled:
            raise RuntimeError("LLM client disabled or missing API key")
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        from lightrag.llm.openai import openai_complete_if_cache

        media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        prompt = (
            "请根据图片和给定标签提取农业病虫害问答所需的视觉线索。"
            "只描述可见作物、病虫害对象、症状、危害部位，不要给出本地证据中没有的防治知识。\n"
            f"标签: {json.dumps(labels, ensure_ascii=False)}\n备注: {notes}"
        )
        messages = [
            {
                "role": "system",
                "content": "你是农业病虫害图像线索提取助手，只输出简短中文描述。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{encoded}",
                        },
                    },
                ],
            },
        ]
        return _resolve_maybe_awaitable(
            openai_complete_if_cache(
                self.vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
            )
        )

    def verify_image_match(self, image_path: str, labels: Dict[str, Any], notes: str) -> Dict[str, Any]:
        path = Path(image_path)
        if not path.exists():
            return {"match": False, "confidence": 0.0, "reason": "image_not_found"}
        from lightrag.llm.openai import openai_complete_if_cache

        media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        prompt = f"""
请判断图片是否与给定农业病虫害标签匹配。
只根据图片可见内容和标签判断，不要使用外部资料。

标签:
{json.dumps(labels, ensure_ascii=False)}
备注:
{notes}

输出JSON:
{{
  "match": true,
  "confidence": 0.0,
  "visible_clues": ["可见线索"],
  "reason": "简短原因"
}}
""".strip()
        messages = [
            {
                "role": "system",
                "content": "你是农业病虫害图像标注质检员，只输出合法JSON。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                    },
                ],
            },
        ]
        raw = _resolve_maybe_awaitable(
            openai_complete_if_cache(
                self.vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
            )
        )
        payload = extract_json_object(raw)
        return {
            "match": bool(payload.get("match", False)),
            "confidence": float(payload.get("confidence", 0.0) or 0.0),
            "visible_clues": _string_list(payload.get("visible_clues")),
            "reason": str(payload.get("reason", "")).strip(),
        }


class SampleGenerator:
    def __init__(self, llm_client: EvalLLMClient) -> None:
        self.llm_client = llm_client

    def generate(
        self,
        sample_id: str,
        task_type: str,
        pack: EvidencePack,
    ) -> Dict[str, Any]:
        payload = self._generate_with_llm(sample_id, task_type, pack)
        if not payload:
            raise RuntimeError("LLM generation returned an empty payload.")
        return payload

    def judge(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = _judge_prompt(sample)
        payload = self.llm_client.complete_json(
            prompt,
            system_prompt="你是评测集质量裁判，只能依据给定证据评分并输出JSON。",
            model=self.llm_client.judge_model,
        )
        return normalize_judge(payload)

    def describe_image(self, pack: EvidencePack) -> str:
        if not pack.image_path:
            raise RuntimeError("Image pack is missing image_path")
        return self.llm_client.describe_image(
            pack.image_path, pack.image_labels, pack.notes
        )

    def verify_image_match(self, pack: EvidencePack) -> Dict[str, Any]:
        if not pack.image_path:
            return {"match": False, "confidence": 0.0, "reason": "missing_image_path"}
        return self.llm_client.verify_image_match(
            pack.image_path, pack.image_labels, pack.notes
        )

    def _generate_with_llm(
        self, sample_id: str, task_type: str, pack: EvidencePack
    ) -> Dict[str, Any]:
        prompt = _generation_prompt(sample_id, task_type, pack)
        payload = self.llm_client.complete_json(
            prompt,
            system_prompt="你是农业病虫害评测集构建专家，只输出合法JSON。",
            model=self.llm_client.generator_model,
        )
        return normalize_generated_sample(sample_id, task_type, pack, payload)

def normalize_generated_sample(
    sample_id: str,
    task_type: str,
    pack: EvidencePack,
    payload: Dict[str, Any],
    generation_method: str = "llm",
) -> Dict[str, Any]:
    question = str(payload.get("question", "")).strip()
    answer = str(payload.get("gold_answer", "")).strip()
    must_include = _string_list(payload.get("must_include")) or [
        pack.core_entity,
        *(pack.expected_entities[:2]),
    ]
    must_not_include = _string_list(payload.get("must_not_include")) or [
        "无证据药剂",
        "无证据剂量",
        "确定为非证据病虫害",
    ]
    difficulty = str(payload.get("difficulty") or "medium").strip().lower()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"

    return {
        "id": sample_id,
        "task_type": task_type,
        "modality": pack.modality,
        "question": question,
        "image_path": pack.image_path,
        "gold_answer": answer,
        "expected_entities": _string_list(payload.get("expected_entities"))
        or pack.expected_entities[:6],
        "expected_relations": _string_list(payload.get("expected_relations"))
        or pack.expected_relations[:5],
        "evidence": [asdict(item) for item in pack.evidence],
        "must_include": must_include,
        "must_not_include": must_not_include,
        "difficulty": difficulty,
        "metadata": {
            "core_entity": pack.core_entity,
            "entity_type": pack.entity_type,
            "evidence_pack_id": pack.pack_id,
            "image_labels": pack.image_labels,
            "notes": pack.notes,
            "generation_method": generation_method,
        },
        "quality": {
            "rule_score": 0.0,
            "evidence_score": 0.0,
            "judge_score": 0,
            "status": "candidate",
        },
    }


def normalize_judge(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "answer_correctness": _clamp_int(payload.get("answer_correctness"), 1, 5, 3),
        "evidence_consistency": _clamp_int(payload.get("evidence_consistency"), 1, 5, 3),
        "safety": _clamp_int(payload.get("safety"), 1, 5, 3),
        "clarity": _clamp_int(payload.get("clarity"), 1, 5, 3),
        "reason": str(payload.get("reason", "")).strip(),
    }


def extract_json_object(text: str) -> Dict[str, Any]:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(stripped)):
        if stripped[i] == "{":
            depth += 1
        elif stripped[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(stripped[start : i + 1])
                    return payload if isinstance(payload, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


def _generation_prompt(sample_id: str, task_type: str, pack: EvidencePack) -> str:
    evidence_json = json.dumps([asdict(item) for item in pack.evidence], ensure_ascii=False)
    return f"""
请根据给定本地知识库证据，生成1条农业病虫害问答评测样本。

硬约束：
1. gold_answer 只能使用 evidence/context 中明确支持的信息，不要使用外部知识。
2. evidence/context 已被截取为围绕当前实体/关系的局部证据；不要使用同一大chunk中与当前实体无关的内容。
3. 如果证据不足类问题，应明确说明“仅凭现有证据无法确定”。
4. 药剂推荐必须严格复述证据中的药剂、剂型、倍液、剂量、适用对象和时期；证据没有的用法不要补充。
5. 输出一个合法JSON对象，不要Markdown。

样本ID：{sample_id}
任务类型：{task_type}
核心实体：{pack.core_entity}
实体类型：{pack.entity_type}
候选实体：{json.dumps(pack.expected_entities, ensure_ascii=False)}
候选关系：{json.dumps(pack.expected_relations, ensure_ascii=False)}
图像路径：{pack.image_path or ""}
图像标签：{json.dumps(pack.image_labels, ensure_ascii=False)}
图像备注：{pack.notes}
证据：
{evidence_json}
上下文：
{pack.context[:2400]}

输出字段：
{{
  "question": "真实用户风格问题",
  "gold_answer": "基于证据的标准答案",
  "expected_entities": ["实体1"],
  "expected_relations": ["关系1"],
  "must_include": ["必须出现的要点"],
  "must_not_include": ["不能出现的错误"],
  "difficulty": "easy|medium|hard"
}}
""".strip()


def _judge_prompt(sample: Dict[str, Any]) -> str:
    payload = {
        "question": sample.get("question"),
        "gold_answer": sample.get("gold_answer"),
        "expected_entities": sample.get("expected_entities"),
        "expected_relations": sample.get("expected_relations"),
        "evidence": sample.get("evidence"),
    }
    return f"""
请只依据给定证据判断评测样本质量，不能使用外部知识。
重点检查 gold_answer 是否使用了同一大chunk中不属于 expected_entities / expected_relations 的内容。
评分范围均为1-5分。

样本：
{json.dumps(payload, ensure_ascii=False)}

输出JSON：
{{
  "answer_correctness": 1,
  "evidence_consistency": 1,
  "safety": 1,
  "clarity": 1,
  "reason": "简短原因"
}}
""".strip()


def _string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return max(low, min(high, parsed))


def _resolve_maybe_awaitable(value: Any) -> str:
    if inspect.isawaitable(value):
        return str(asyncio.run(value))
    return str(value)


def _load_rag_config_defaults() -> Dict[str, str]:
    try:
        from raganything.config import RAGAnythingConfig

        cfg = RAGAnythingConfig()
        return {
            "model_answer": cfg.model_answer,
            "model_judge": getattr(cfg, "model_judge", "gemini-3.1-pro-preview"),
            "model_vision": cfg.model_vision,
        }
    except Exception:
        return {
            "model_answer": os.getenv("RAG_MODEL_ANSWER", "gemini-2.5-flash"),
            "model_judge": os.getenv("RAG_MODEL_JUDGE", "gemini-3.1-pro-preview"),
            "model_vision": os.getenv("RAG_MODEL_VISION", "gemini-2.5-flash"),
        }


def _load_pipeline_api_defaults() -> Dict[str, str]:
    path = Path("pdf_rag_pipeline.py")
    if not path.exists():
        return {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    defaults: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "main":
            continue
        defaults.update(_extract_string_assignments(node.body))
        break
    if not defaults:
        defaults.update(_extract_string_assignments(tree.body))
    return defaults


def _extract_string_assignments(statements: List[ast.stmt]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for stmt in statements:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            continue
        name = stmt.targets[0].id
        if name not in {"api_key", "base_url"}:
            continue
        if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            values[name] = stmt.value.value
    return values
