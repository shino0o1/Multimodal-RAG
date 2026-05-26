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
import urllib.error
import urllib.request
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
MAX_RETRIES = 1


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
            or cfg.get("llm_api_key")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("API_KEY")
            or pipeline_secrets.get("api_key")
        )
        self.base_url = (
            base_url
            or cfg.get("llm_base_url")
            or os.getenv("OPENAI_BASE_URL")
            or pipeline_secrets.get("base_url")
        )
        self.reasoning_effort_default = cfg.get("reasoning_effort_default", "")
        self.reasoning_effort_answer = cfg.get("reasoning_effort_answer", "")
        self.reasoning_effort_judge = cfg.get("reasoning_effort_judge", "")
        self.reasoning_effort_vision = cfg.get("reasoning_effort_vision", "")
        self.openai_llm_extra_body: Dict[str, Any] = {}
        raw_extra_body = os.getenv("OPENAI_LLM_EXTRA_BODY", "").strip()
        if raw_extra_body:
            try:
                loaded = json.loads(raw_extra_body)
                if isinstance(loaded, dict):
                    self.openai_llm_extra_body = loaded
            except Exception:
                self.openai_llm_extra_body = {}
        self.enabled = enabled and bool(self.api_key)

    def complete(self, prompt: str, system_prompt: str, model: Optional[str] = None) -> str:
        if not self.enabled:
            raise RuntimeError("LLM client disabled or missing API key")
        model_name = model or self.generator_model
        kwargs: Dict[str, Any] = {}
        reasoning_effort = self._resolve_reasoning_effort(model_name)
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        self._merge_extra_body(kwargs)

        try:
            from lightrag.llm.openai import openai_complete_if_cache
        except ModuleNotFoundError:
            return self._direct_chat_complete(
                model_name,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                **kwargs,
            )

        try:
            return _resolve_maybe_awaitable(
                openai_complete_if_cache(
                    model_name,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=[],
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **kwargs,
                )
            )
        except Exception:
            raise

    def complete_json(
        self, prompt: str, system_prompt: str, model: Optional[str] = None
    ) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None
        for _ in range(MAX_RETRIES):
            try:
                raw = self.complete(prompt, system_prompt, model=model)
                payload = extract_json_object(raw)
                if payload:
                    return payload
            except Exception as exc:
                if not _is_retryable_error(exc):
                    raise
                last_exc = exc
                continue
        if last_exc is not None:
            raise RuntimeError(f"LLM JSON call failed after {MAX_RETRIES} retries: {last_exc}") from last_exc
        return {}

    def complete_json_with_image(
        self,
        prompt: str,
        system_prompt: str,
        image_path: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("LLM client disabled or missing API key")
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        model_name = model or self.vision_model
        media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        messages = [
            {"role": "system", "content": system_prompt},
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
        kwargs: Dict[str, Any] = {}
        reasoning_effort = self._resolve_reasoning_effort(model_name)
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        self._merge_extra_body(kwargs)

        try:
            from lightrag.llm.openai import openai_complete_if_cache
        except ModuleNotFoundError:
            last_exc: Optional[Exception] = None
            for _ in range(MAX_RETRIES):
                try:
                    raw = self._direct_chat_complete(model_name, messages, **kwargs)
                    payload = extract_json_object(raw)
                    if payload:
                        return payload
                except Exception as exc:
                    if not _is_retryable_error(exc):
                        raise
                    last_exc = exc
            if last_exc is not None:
                raise RuntimeError(
                    f"LLM image JSON call failed after {MAX_RETRIES} retries: {last_exc}"
                ) from last_exc
            return {}

        last_exc: Optional[Exception] = None
        for _ in range(MAX_RETRIES):
            try:
                try:
                    raw = _resolve_maybe_awaitable(
                        openai_complete_if_cache(
                            model_name,
                            "",
                            system_prompt=None,
                            history_messages=[],
                            messages=messages,
                            api_key=self.api_key,
                            base_url=self.base_url,
                            **kwargs,
                        )
                    )
                except Exception:
                    raise
                payload = extract_json_object(raw)
                if payload:
                    return payload
            except Exception as exc:
                if not _is_retryable_error(exc):
                    raise
                last_exc = exc
                continue
        if last_exc is not None:
            raise RuntimeError(f"LLM image JSON call failed after {MAX_RETRIES} retries: {last_exc}") from last_exc
        return {}

    def describe_image(self, image_path: str, labels: Dict[str, Any], notes: str) -> str:
        if not self.enabled:
            raise RuntimeError("LLM client disabled or missing API key")
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

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
        last_exc: Optional[Exception] = None
        direct_kwargs = (
            {"reasoning_effort": self._resolve_reasoning_effort(self.vision_model)}
            if self._resolve_reasoning_effort(self.vision_model)
            else {}
        )
        try:
            from lightrag.llm.openai import openai_complete_if_cache
        except ModuleNotFoundError:
            for _ in range(MAX_RETRIES):
                try:
                    return self._direct_chat_complete(
                        self.vision_model, messages, **direct_kwargs
                    )
                except Exception as exc:
                    if not _is_retryable_error(exc):
                        raise
                    last_exc = exc
            raise RuntimeError(f"Image describe failed after {MAX_RETRIES} retries: {last_exc}")

        for _ in range(MAX_RETRIES):
            try:
                return _resolve_maybe_awaitable(
                    openai_complete_if_cache(
                        self.vision_model,
                        "",
                        system_prompt=None,
                        history_messages=[],
                        messages=messages,
                        api_key=self.api_key,
                        base_url=self.base_url,
                        **direct_kwargs,
                    )
                )
            except Exception as exc:
                if not _is_retryable_error(exc):
                    raise
                last_exc = exc
        raise RuntimeError(f"Image describe failed after {MAX_RETRIES} retries: {last_exc}")

    def verify_image_match(self, image_path: str, labels: Dict[str, Any], notes: str) -> Dict[str, Any]:
        path = Path(image_path)
        if not path.exists():
            return {"match": False, "confidence": 0.0, "reason": "image_not_found"}

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
        payload: Dict[str, Any] = {}
        last_exc: Optional[Exception] = None
        direct_kwargs = (
            {"reasoning_effort": self._resolve_reasoning_effort(self.vision_model)}
            if self._resolve_reasoning_effort(self.vision_model)
            else {}
        )
        try:
            from lightrag.llm.openai import openai_complete_if_cache
        except ModuleNotFoundError:
            for _ in range(MAX_RETRIES):
                try:
                    raw = self._direct_chat_complete(
                        self.vision_model, messages, **direct_kwargs
                    )
                    payload = extract_json_object(raw)
                    if payload:
                        break
                except Exception as exc:
                    if not _is_retryable_error(exc):
                        raise
                    last_exc = exc
            if not payload and last_exc is not None:
                raise RuntimeError(
                    f"Image verify failed after {MAX_RETRIES} retries: {last_exc}"
                ) from last_exc
            return {
                "match": bool(payload.get("match", False)),
                "confidence": float(payload.get("confidence", 0.0) or 0.0),
                "visible_clues": _string_list(payload.get("visible_clues")),
                "reason": str(payload.get("reason", "")).strip(),
            }

        for _ in range(MAX_RETRIES):
            try:
                raw = _resolve_maybe_awaitable(
                    openai_complete_if_cache(
                        self.vision_model,
                        "",
                        system_prompt=None,
                        history_messages=[],
                        messages=messages,
                        api_key=self.api_key,
                        base_url=self.base_url,
                        **direct_kwargs,
                    )
                )
                payload = extract_json_object(raw)
                if payload:
                    break
            except Exception as exc:
                if not _is_retryable_error(exc):
                    raise
                last_exc = exc
        if not payload and last_exc is not None:
            raise RuntimeError(f"Image verify failed after {MAX_RETRIES} retries: {last_exc}") from last_exc
        return {
            "match": bool(payload.get("match", False)),
            "confidence": float(payload.get("confidence", 0.0) or 0.0),
            "visible_clues": _string_list(payload.get("visible_clues")),
            "reason": str(payload.get("reason", "")).strip(),
        }

    def _resolve_reasoning_effort(self, model_name: str) -> str:
        if model_name == self.judge_model:
            return self.reasoning_effort_judge or self.reasoning_effort_default
        if model_name == self.vision_model:
            return self.reasoning_effort_vision or self.reasoning_effort_default
        if model_name == self.generator_model:
            return self.reasoning_effort_answer or self.reasoning_effort_default
        return self.reasoning_effort_default

    def _merge_extra_body(self, kwargs: Dict[str, Any]) -> None:
        if not self.openai_llm_extra_body:
            return
        existing = kwargs.get("extra_body")
        if isinstance(existing, dict):
            merged = dict(self.openai_llm_extra_body)
            merged.update(existing)
            kwargs["extra_body"] = merged
        else:
            kwargs["extra_body"] = dict(self.openai_llm_extra_body)

    def _direct_chat_complete(
        self, model_name: str, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> str:
        base_url = (self.base_url or "").rstrip("/")
        if not base_url:
            raise RuntimeError("Missing LLM base_url")
        url = base_url + "/chat/completions"
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }
        flat_kwargs = {key: value for key, value in kwargs.items() if value not in ("", None)}
        extra_body = flat_kwargs.pop("extra_body", None)
        if isinstance(extra_body, dict):
            payload.update(extra_body)
        payload.update(flat_kwargs)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from LLM API: {body[:500]}") from exc
        payload = json.loads(raw)
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("LLM API returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            return "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
            )
        return str(content or "")


class SampleGenerator:
    def __init__(self, llm_client: EvalLLMClient) -> None:
        self.llm_client = llm_client

    def generate(
        self,
        sample_id: str,
        task_type: str,
        pack: EvidencePack,
    ) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None
        for _ in range(MAX_RETRIES):
            try:
                payload = self._generate_with_llm(sample_id, task_type, pack)
                if payload and str(payload.get("question", "")).strip() and str(payload.get("gold_answer", "")).strip():
                    return payload
            except Exception as exc:
                if not _is_retryable_error(exc):
                    raise
                last_exc = exc
        if last_exc is not None:
            raise RuntimeError(f"LLM generation failed after {MAX_RETRIES} retries: {last_exc}") from last_exc
        raise RuntimeError("LLM generation returned invalid fields after retries.")

    def judge(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = _judge_prompt(sample)
        system_prompt = "你是评测集质量裁判，只能依据给定证据和随请求提供的原图评分并输出JSON。"
        last_exc: Optional[Exception] = None
        for _ in range(MAX_RETRIES):
            try:
                if sample.get("modality") == "image" and sample.get("image_path"):
                    payload = self.llm_client.complete_json_with_image(
                        prompt,
                        system_prompt=system_prompt,
                        image_path=str(sample.get("image_path")),
                        model=self.llm_client.judge_model,
                    )
                else:
                    payload = self.llm_client.complete_json(
                        prompt,
                        system_prompt=system_prompt,
                        model=self.llm_client.judge_model,
                    )
                if payload:
                    return normalize_judge(payload)
            except Exception as exc:
                if not _is_retryable_error(exc):
                    raise
                last_exc = exc
        if last_exc is not None:
            raise RuntimeError(f"Judge failed after {MAX_RETRIES} retries: {last_exc}") from last_exc
        raise RuntimeError("Judge returned invalid fields after retries.")

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
    return {
        "id": sample_id,
        "task_type": task_type,
        "modality": pack.modality,
        "question": question,
        "image_path": pack.image_path,
        "gold_answer": answer,
        "evidence": [asdict(item) for item in pack.evidence],
        "metadata": {
            "core_entity": pack.core_entity,
        },
        "quality": {
            "rule_score": 0.0,
            "evidence_score": 0.0,
            "judge_score": 0,
            "status": "candidate",
            "reason": "",
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
  "gold_answer": "基于证据的标准答案"
}}
""".strip()


def _judge_prompt(sample: Dict[str, Any]) -> str:
    payload = {
        "modality": sample.get("modality"),
        "image_path": sample.get("image_path"),
        "question": sample.get("question"),
        "gold_answer": sample.get("gold_answer"),
        "evidence": sample.get("evidence"),
        "metadata": sample.get("metadata", {}),
    }
    return f"""
请只依据给定证据判断评测样本质量，不能使用外部知识。
对于图像问答样本，随请求附带的原图也属于给定证据，可用于判断图片中可见对象、症状、作物和部位是否支持问题与答案。
重点检查 gold_answer 是否由 evidence 和原图支持，不要引入证据外细节。
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


def _is_retryable_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = [
        "apiconnectionerror",
        "connectionerror",
        "timeout",
        "temporarily unavailable",
        "rate limit",
        "server error",
        "json",
        "decode",
    ]
    return any(marker in text for marker in markers)


def _load_rag_config_defaults() -> Dict[str, str]:
    def _prefer_env(value: str, env_name: str) -> str:
        if env_name in os.environ:
            return os.environ.get(env_name, "")
        return value

    try:
        from raganything.config import RAGAnythingConfig

        cfg = RAGAnythingConfig()
        values = {
            "model_answer": _prefer_env(cfg.model_answer, "RAG_MODEL_ANSWER"),
            "model_judge": _prefer_env(
                getattr(cfg, "model_judge", "gemini-3.1-pro-preview"),
                "RAG_MODEL_JUDGE",
            ),
            "model_vision": _prefer_env(cfg.model_vision, "RAG_MODEL_VISION"),
            "llm_api_key": _prefer_env(getattr(cfg, "llm_api_key", ""), "RAG_LLM_API_KEY"),
            "llm_base_url": _prefer_env(
                getattr(cfg, "llm_base_url", ""), "RAG_LLM_BASE_URL"
            ),
            "reasoning_effort_default": _prefer_env(
                getattr(cfg, "reasoning_effort_default", ""),
                "RAG_REASONING_EFFORT_DEFAULT",
            ),
            "reasoning_effort_answer": _prefer_env(
                getattr(cfg, "reasoning_effort_answer", ""),
                "RAG_REASONING_EFFORT_ANSWER",
            ),
            "reasoning_effort_judge": _prefer_env(
                getattr(cfg, "reasoning_effort_judge", ""),
                "RAG_REASONING_EFFORT_JUDGE",
            ),
            "reasoning_effort_vision": _prefer_env(
                getattr(cfg, "reasoning_effort_vision", ""),
                "RAG_REASONING_EFFORT_VISION",
            ),
        }
        return values
    except Exception:
        defaults = _load_config_file_defaults()
        return {
            "model_answer": os.getenv("RAG_MODEL_ANSWER", "gemini-2.5-flash"),
            "model_judge": os.getenv("RAG_MODEL_JUDGE", "gemini-3.1-pro-preview"),
            "model_vision": os.getenv("RAG_MODEL_VISION", "gemini-2.5-flash"),
            "llm_api_key": os.getenv("RAG_LLM_API_KEY", defaults.get("llm_api_key", "")),
            "llm_base_url": os.getenv("RAG_LLM_BASE_URL", defaults.get("llm_base_url", "")),
            "reasoning_effort_default": os.getenv("RAG_REASONING_EFFORT_DEFAULT", ""),
            "reasoning_effort_answer": os.getenv("RAG_REASONING_EFFORT_ANSWER", ""),
            "reasoning_effort_judge": os.getenv("RAG_REASONING_EFFORT_JUDGE", ""),
            "reasoning_effort_vision": os.getenv("RAG_REASONING_EFFORT_VISION", ""),
        }


def _load_config_file_defaults() -> Dict[str, str]:
    path = Path("raganything/config.py")
    if not path.exists():
        return {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    defaults: Dict[str, str] = {}
    wanted = {"llm_api_key", "llm_base_url"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        name = node.target.id
        if name not in wanted or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name) or call.func.id != "field":
            continue
        for keyword in call.keywords:
            if keyword.arg != "default" or not isinstance(keyword.value, ast.Call):
                continue
            default_call = keyword.value
            if (
                isinstance(default_call.func, ast.Name)
                and default_call.func.id == "get_env_value"
                and len(default_call.args) >= 2
                and isinstance(default_call.args[1], ast.Constant)
                and isinstance(default_call.args[1].value, str)
            ):
                defaults[name] = default_call.args[1].value
    return defaults


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
