#!/usr/bin/env python3
"""Run end-to-end evaluation for RAGAnything with a judge model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raganything.eval_dataset.generators import EvalLLMClient
from ui.service import RAGUIService


JUDGE_SYSTEM_PROMPT = (
    "你是农业病虫害问答评测裁判。"
    "只根据输入的问题、gold答案、系统答案、证据信息打分。"
    "禁止使用外部知识。"
    "只输出JSON，不要输出额外文本。"
)

JUDGE_USER_PROMPT = """
请按下面规则打分（1-5分，整数）：
1) correctness：系统答案是否回答了问题且事实基本正确
2) evidence_consistency：系统答案是否与给定证据一致，是否出现证据外硬性结论
3) completeness：关键要点覆盖是否充分
4) clarity：表达是否清晰、结构是否可读
5) safety：是否存在明显误导或高风险建议

输出JSON格式：
{{
  "correctness": 1,
  "evidence_consistency": 1,
  "completeness": 1,
  "clarity": 1,
  "safety": 1,
  "reason": "简短中文解释"
}}

输入：
- question: {question}
- gold_answer: {gold_answer}
- system_answer: {system_answer}
- gold_evidence: {gold_evidence}
- retrieved_evidence: {retrieved_evidence}
""".strip()


@dataclass
class EvalConfig:
    eval_file: Path
    rag_dir: Path
    output_dir: Path
    concurrency: int = 6
    limit: int = 0
    mode: str = "hybrid"
    system_model: str = "qwen3-8b"
    judge_model: str = "qwen3.5-plus"
    planner_enabled: bool = True
    ablate_no_kg: bool = False
    ablate_no_rag: bool = False
    openai_llm_extra_body: str = ""
    enable_judge: bool = False
    disable_llm_cache: bool = True


class _MemoryUpload:
    """Simple file-like wrapper for service.query_with_image."""

    def __init__(self, path: Path) -> None:
        self.name = path.name
        self._data = path.read_bytes()

    def getvalue(self) -> bytes:
        return self._data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate QA system using Eval.jsonl + judge model."
    )
    parser.add_argument("--eval-file", default="eval_dataset_200/Eval.jsonl")
    parser.add_argument("--rag-dir", default="rag_storage_whole_book_gemini")
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["local", "global", "hybrid", "naive", "mix", "bypass"],
        help="normal query mode when no ablation override is used",
    )
    parser.add_argument("--system-model", default="qwen3-8b")
    parser.add_argument("--judge-model", default="qwen3.5-plus")
    parser.add_argument(
        "--openai-llm-extra-body",
        default="",
        help="raw JSON string for OPENAI_LLM_EXTRA_BODY, provider-specific",
    )
    parser.add_argument(
        "--disable-planner",
        action="store_true",
        help="disable planner stage",
    )
    parser.add_argument(
        "--ablate-no-kg",
        action="store_true",
        help="ablation: disable KG path by forcing mode=naive",
    )
    parser.add_argument(
        "--ablate-no-rag",
        action="store_true",
        help="ablation: disable retrieval by forcing mode=bypass",
    )
    parser.add_argument(
        "--enable-judge",
        action="store_true",
        help="enable LLM judge step (disabled by default for speed)",
    )
    parser.add_argument(
        "--disable-llm-cache",
        action="store_true",
        default=True,
        help="disable query-time LLM cache during evaluation (default: true)",
    )
    parser.add_argument(
        "--enable-llm-cache",
        action="store_true",
        help="override and enable query-time LLM cache during evaluation",
    )
    return parser.parse_args()


def _force_system_model_env(
    model_name: str,
    openai_llm_extra_body: str = "",
    disable_llm_cache: bool = True,
) -> None:
    # Force all system-side model routes to one small model for evaluation.
    os.environ["RAG_MODEL_ANSWER"] = model_name
    os.environ["RAG_MODEL_PLANNER"] = model_name
    os.environ["RAG_MODEL_VISION"] = model_name
    os.environ["RAG_MODEL_IMAGE_DESCRIPTION"] = model_name
    os.environ["RAG_UI_LLM_MODEL"] = model_name
    os.environ["RAG_UI_VISION_MODEL"] = model_name
    # Avoid provider-side reasoning mode conflicts during eval.
    os.environ["RAG_REASONING_EFFORT_DEFAULT"] = ""
    os.environ["RAG_REASONING_EFFORT_ANSWER"] = ""
    os.environ["RAG_REASONING_EFFORT_PLANNER"] = ""
    os.environ["RAG_REASONING_EFFORT_VISION"] = ""
    os.environ["RAG_REASONING_EFFORT_IMAGE_DESCRIPTION"] = ""
    os.environ["RAG_REASONING_EFFORT_JUDGE"] = ""
    # Reduce duplicate retries when SDK/client retry is already active upstream.
    os.environ["OPENAI_MAX_RETRIES"] = "0"
    if disable_llm_cache:
        os.environ["RAG_DISABLE_LLM_CACHE"] = "true"
    else:
        os.environ.pop("RAG_DISABLE_LLM_CACHE", None)
    if openai_llm_extra_body.strip():
        os.environ["OPENAI_LLM_EXTRA_BODY"] = openai_llm_extra_body.strip()


def _collect_reasoning_runtime(judge_client: Optional[EvalLLMClient]) -> Dict[str, Any]:
    system_efforts = {
        "default": os.getenv("RAG_REASONING_EFFORT_DEFAULT", "").strip() or "none",
        "answer": os.getenv("RAG_REASONING_EFFORT_ANSWER", "").strip() or "none",
        "planner": os.getenv("RAG_REASONING_EFFORT_PLANNER", "").strip() or "none",
        "vision": os.getenv("RAG_REASONING_EFFORT_VISION", "").strip() or "none",
        "image_description": os.getenv("RAG_REASONING_EFFORT_IMAGE_DESCRIPTION", "").strip() or "none",
    }
    judge_effort = "disabled"
    if judge_client is not None:
        judge_effort = (
            (judge_client.reasoning_effort_judge or judge_client.reasoning_effort_default or "").strip()
            or "none"
        )
    return {
        "thinking_mode": "not_explicitly_set",
        "system_reasoning_efforts": system_efforts,
        "judge_reasoning_effort": judge_effort,
        "openai_llm_extra_body": os.getenv("OPENAI_LLM_EXTRA_BODY", "").strip(),
        "llm_cache_disabled": (
            os.getenv("RAG_DISABLE_LLM_CACHE", "").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
    }


def _print_reasoning_runtime(reasoning: Dict[str, Any]) -> None:
    thinking_mode = str(reasoning.get("thinking_mode", "not_explicitly_set"))
    print(f"[reasoning] thinking: {thinking_mode}")
    efforts = reasoning.get("system_reasoning_efforts", {}) or {}
    print(
        "[reasoning] system_effort:"
        f" answer={efforts.get('answer', 'none')},"
        f" planner={efforts.get('planner', 'none')},"
        f" vision={efforts.get('vision', 'none')},"
        f" image_desc={efforts.get('image_description', 'none')},"
        f" default={efforts.get('default', 'none')}"
    )
    print(f"[reasoning] judge_effort: {reasoning.get('judge_reasoning_effort', 'none')}")
    extra_body = str(reasoning.get("openai_llm_extra_body", "") or "").strip()
    print(f"[reasoning] OPENAI_LLM_EXTRA_BODY: {extra_body or '(empty)'}")
    print(
        f"[reasoning] llm_cache_disabled: {bool(reasoning.get('llm_cache_disabled', False))}"
    )
    if all(v == "none" for v in efforts.values()):
        print("[reasoning] notice: system reasoning not active in current run.")


def _resolve_query_mode(cfg: EvalConfig) -> str:
    if cfg.ablate_no_rag:
        return "bypass"
    if cfg.ablate_no_kg:
        return "naive"
    return cfg.mode


def _load_rows(eval_file: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in eval_file.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    if limit > 0:
        return rows[:limit]
    return rows


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _clamp_1_5(value: Any) -> int:
    try:
        n = int(value)
    except Exception:
        return 3
    return max(1, min(5, n))


def _get_gold_chunk_ids(row: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for item in row.get("evidence", []) or []:
        cid = str(item.get("chunk_id", "")).strip()
        if cid and cid not in out:
            out.append(cid)
    return out


def _get_pred_chunk_ids(resp: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for item in resp.get("citations", []) or []:
        cid = str(item.get("chunk_id", "")).strip()
        if cid and cid not in out:
            out.append(cid)
    return out


def _calc_retrieval(gold_ids: List[str], pred_ids: List[str]) -> Dict[str, float]:
    g = set(gold_ids)
    p = set(pred_ids)
    hit = len(g & p)
    recall = hit / len(g) if g else 0.0
    precision = hit / len(p) if p else 0.0
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return {
        "hit": hit,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }


def _build_judge_prompt(
    question: str,
    gold_answer: str,
    system_answer: str,
    gold_evidence: List[Dict[str, Any]],
    retrieved_evidence: List[Dict[str, Any]],
) -> str:
    try:
        return JUDGE_USER_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
            system_answer=system_answer,
            gold_evidence=json.dumps(gold_evidence, ensure_ascii=False),
            retrieved_evidence=json.dumps(retrieved_evidence, ensure_ascii=False),
        )
    except KeyError as exc:
        raise RuntimeError(
            f"Judge prompt template formatting failed, missing placeholder: {exc}"
        ) from exc


def _judge_one(
    judge_client: EvalLLMClient,
    judge_model: str,
    row: Dict[str, Any],
    system_answer: str,
    pred_chunk_ids: List[str],
) -> Dict[str, Any]:
    gold_evidence = row.get("evidence", []) if isinstance(row.get("evidence"), list) else []
    retrieved_evidence = [{"chunk_id": cid} for cid in pred_chunk_ids]
    prompt = _build_judge_prompt(
        question=str(row.get("question", "")),
        gold_answer=str(row.get("gold_answer", "")),
        system_answer=system_answer,
        gold_evidence=gold_evidence,
        retrieved_evidence=retrieved_evidence,
    )
    payload = judge_client.complete_json(
        prompt=prompt,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        model=judge_model,
    )
    return {
        "correctness": _clamp_1_5(payload.get("correctness")),
        "evidence_consistency": _clamp_1_5(payload.get("evidence_consistency")),
        "completeness": _clamp_1_5(payload.get("completeness")),
        "clarity": _clamp_1_5(payload.get("clarity")),
        "safety": _clamp_1_5(payload.get("safety")),
        "reason": str(payload.get("reason", "")).strip(),
    }


def _resolve_image_path(raw: str, eval_file: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    candidate = (eval_file.parent / p).resolve()
    if candidate.exists():
        return candidate
    return p.resolve()


def _evaluate_one(
    service: RAGUIService,
    kb_id: str,
    judge_client: Optional[EvalLLMClient],
    cfg: EvalConfig,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    mode = _resolve_query_mode(cfg)
    planner_enabled = cfg.planner_enabled and (not cfg.ablate_no_rag)
    sample_id = str(row.get("id", "")).strip()
    question = str(row.get("question", "")).strip()
    started = time.perf_counter()
    query_time = 0.0
    judge_time = 0.0

    try:
        modality = str(row.get("modality", "")).lower().strip()
        t_query_start = time.perf_counter()
        if modality == "image" and row.get("image_path"):
            image_path = _resolve_image_path(str(row.get("image_path")), cfg.eval_file)
            if not image_path.exists():
                raise FileNotFoundError(f"image not found: {image_path}")
            response = service.query_with_image(
                kb_id=kb_id,
                question=question,
                image_file=_MemoryUpload(image_path),
                mode=mode,
                planner_enabled=planner_enabled,
                debug=False,
                include_evidence=False,
            )
        else:
            response = service.query(
                kb_id=kb_id,
                question=question,
                mode=mode,
                planner_enabled=planner_enabled,
                debug=False,
                include_evidence=False,
            )
        query_time = round(time.perf_counter() - t_query_start, 4)

        latency = round(time.perf_counter() - started, 4)
        answer = str(response.get("answer", "")).strip()
        pred_chunk_ids = _get_pred_chunk_ids(response)
        gold_chunk_ids = _get_gold_chunk_ids(row)
        retrieval = _calc_retrieval(gold_chunk_ids, pred_chunk_ids)
        judge: Dict[str, Any] = {}
        total_score: Optional[int] = None
        pass_flag: Optional[bool] = None
        if cfg.enable_judge:
            if judge_client is None:
                raise RuntimeError("judge enabled but judge client is not initialized")
            t_judge_start = time.perf_counter()
            judge = _judge_one(judge_client, cfg.judge_model, row, answer, pred_chunk_ids)
            judge_time = round(time.perf_counter() - t_judge_start, 4)
            total_score = (
                judge["correctness"]
                + judge["evidence_consistency"]
                + judge["completeness"]
                + judge["clarity"]
                + judge["safety"]
            )
            pass_flag = (
                judge["correctness"] >= 4
                and judge["evidence_consistency"] >= 4
                and judge["safety"] >= 4
            )
        return {
            "id": sample_id,
            "task_type": row.get("task_type", ""),
            "modality": row.get("modality", ""),
            "question": question,
            "gold_answer": row.get("gold_answer", ""),
            "system_answer": answer,
            "mode": mode,
            "planner_enabled": planner_enabled,
            "latency_sec": latency,
            "query_time_sec": query_time,
            "judge_time_sec": judge_time,
            "retrieval": retrieval,
            "gold_chunk_ids": gold_chunk_ids,
            "pred_chunk_ids": pred_chunk_ids,
            "judge": judge,
            "total_score": total_score,
            "pass": pass_flag,
            "error": "",
        }
    except Exception as exc:
        return {
            "id": sample_id,
            "task_type": row.get("task_type", ""),
            "modality": row.get("modality", ""),
            "question": question,
            "gold_answer": row.get("gold_answer", ""),
            "system_answer": "",
            "mode": mode,
            "planner_enabled": planner_enabled,
            "latency_sec": round(time.perf_counter() - started, 4),
            "query_time_sec": query_time,
            "judge_time_sec": judge_time,
            "retrieval": {"hit": 0, "recall": 0.0, "precision": 0.0, "f1": 0.0},
            "gold_chunk_ids": _get_gold_chunk_ids(row),
            "pred_chunk_ids": [],
            "judge": {},
            "total_score": None,
            "pass": False,
            "error": str(exc)[:500],
        }


def _build_group_report(results: List[Dict[str, Any]], group_key: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row.get(group_key, ""))].append(row)

    output: List[Dict[str, Any]] = []
    for group_name in sorted(grouped):
        rows = grouped[group_name]
        ok = [r for r in rows if not r.get("error")]
        output.append(
            {
                group_key: group_name,
                "samples": len(rows),
                "success": len(ok),
                "error_count": len(rows) - len(ok),
                "pass_rate": round(sum(1 for r in ok if r.get("pass")) / max(1, len(ok)), 4),
                "avg_total_score": _safe_mean([_to_float(r.get("total_score", 0)) for r in ok]),
                "avg_correctness": _safe_mean(
                    [_to_float((r.get("judge") or {}).get("correctness", 0)) for r in ok]
                ),
                "avg_evidence_consistency": _safe_mean(
                    [_to_float((r.get("judge") or {}).get("evidence_consistency", 0)) for r in ok]
                ),
                "avg_completeness": _safe_mean(
                    [_to_float((r.get("judge") or {}).get("completeness", 0)) for r in ok]
                ),
                "avg_clarity": _safe_mean(
                    [_to_float((r.get("judge") or {}).get("clarity", 0)) for r in ok]
                ),
                "avg_safety": _safe_mean(
                    [_to_float((r.get("judge") or {}).get("safety", 0)) for r in ok]
                ),
                "avg_recall": _safe_mean(
                    [_to_float((r.get("retrieval") or {}).get("recall", 0)) for r in ok]
                ),
                "avg_precision": _safe_mean(
                    [_to_float((r.get("retrieval") or {}).get("precision", 0)) for r in ok]
                ),
                "avg_f1": _safe_mean([_to_float((r.get("retrieval") or {}).get("f1", 0)) for r in ok]),
                "avg_latency_sec": _safe_mean([_to_float(r.get("latency_sec", 0)) for r in ok]),
            }
        )
    return output


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = max(0, int(len(arr) * 0.95) - 1)
    return round(arr[idx], 4)


def main() -> None:
    args = parse_args()
    cfg = EvalConfig(
        eval_file=Path(args.eval_file),
        rag_dir=Path(args.rag_dir),
        output_dir=Path(args.output_dir),
        concurrency=max(1, int(args.concurrency)),
        limit=max(0, int(args.limit)),
        mode=str(args.mode),
        system_model=str(args.system_model),
        judge_model=str(args.judge_model),
        planner_enabled=not bool(args.disable_planner),
        ablate_no_kg=bool(args.ablate_no_kg),
        ablate_no_rag=bool(args.ablate_no_rag),
        openai_llm_extra_body=str(args.openai_llm_extra_body or ""),
        enable_judge=bool(args.enable_judge),
        disable_llm_cache=(not bool(args.enable_llm_cache)),
    )

    if not cfg.eval_file.exists():
        raise FileNotFoundError(f"Eval file not found: {cfg.eval_file}")
    if not cfg.rag_dir.exists():
        raise FileNotFoundError(f"RAG dir not found: {cfg.rag_dir}")

    _force_system_model_env(
        cfg.system_model,
        cfg.openai_llm_extra_body,
        disable_llm_cache=cfg.disable_llm_cache,
    )
    if cfg.system_model.lower().startswith("qwen3") and not cfg.openai_llm_extra_body.strip():
        print(
            "[warning] qwen3 model detected but --openai-llm-extra-body is empty. "
            "If provider requires non-thinking in non-stream mode, request may fail with 429."
        )

    service = RAGUIService()
    kb_meta = service.register_existing_kb(str(cfg.rag_dir))
    kb_id = str(kb_meta["kb_id"])

    judge_client: Optional[EvalLLMClient] = None
    if cfg.enable_judge:
        judge_client = EvalLLMClient(
            generator_model=cfg.judge_model,
            judge_model=cfg.judge_model,
            vision_model=cfg.judge_model,
            enabled=True,
        )
        if not judge_client.enabled:
            raise RuntimeError("Judge client disabled: missing API key or base URL.")
    reasoning_runtime = _collect_reasoning_runtime(judge_client)
    _print_reasoning_runtime(reasoning_runtime)

    rows = _load_rows(cfg.eval_file, cfg.limit)
    results: List[Dict[str, Any]] = []

    started = time.perf_counter()
    if rows:
        # Warm up once to avoid race-driven duplicate RAG initialization under concurrency.
        first = _evaluate_one(service, kb_id, judge_client, cfg, rows[0])
        results.append(first)
        print(f"[progress] 1/{len(rows)}")

    remaining = rows[1:] if len(rows) > 1 else []
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        futures = [
            pool.submit(_evaluate_one, service, kb_id, judge_client, cfg, row)
            for row in remaining
        ]
        done = 1 if rows else 0
        total = len(rows)
        for fut in as_completed(futures):
            done += 1
            results.append(fut.result())
            if done % 10 == 0 or done == total:
                print(f"[progress] {done}/{total}")
    total_wall = round(time.perf_counter() - started, 4)

    ok = [r for r in results if not r.get("error")]
    latency_values = [_to_float(r.get("latency_sec", 0)) for r in ok]

    summary = {
        "config": {
            "eval_file": str(cfg.eval_file),
            "rag_dir": str(cfg.rag_dir),
            "resolved_mode": _resolve_query_mode(cfg),
            "planner_enabled": cfg.planner_enabled and (not cfg.ablate_no_rag),
            "ablate_no_kg": cfg.ablate_no_kg,
            "ablate_no_rag": cfg.ablate_no_rag,
            "system_model": cfg.system_model,
            "judge_model": cfg.judge_model,
            "enable_judge": cfg.enable_judge,
            "disable_llm_cache": cfg.disable_llm_cache,
            "concurrency": cfg.concurrency,
            "sample_count": len(rows),
            "reasoning": reasoning_runtime,
        },
        "timing": {
            "total_wall_time_sec": total_wall,
            "avg_latency_sec": _safe_mean(latency_values),
            "p95_latency_sec": _p95(latency_values),
        },
        "counts": {
            "total": len(results),
            "success": len(ok),
            "errors": len(results) - len(ok),
            "pass": (
                sum(1 for r in ok if r.get("pass"))
                if cfg.enable_judge
                else None
            ),
            "pass_rate": (
                round(sum(1 for r in ok if r.get("pass")) / max(1, len(ok)), 4)
                if cfg.enable_judge
                else None
            ),
        },
        "scores": {
            "avg_total_score": _safe_mean([_to_float(r.get("total_score", 0)) for r in ok]),
            "avg_correctness": _safe_mean(
                [_to_float((r.get("judge") or {}).get("correctness", 0)) for r in ok]
            ),
            "avg_evidence_consistency": _safe_mean(
                [_to_float((r.get("judge") or {}).get("evidence_consistency", 0)) for r in ok]
            ),
            "avg_completeness": _safe_mean(
                [_to_float((r.get("judge") or {}).get("completeness", 0)) for r in ok]
            ),
            "avg_clarity": _safe_mean(
                [_to_float((r.get("judge") or {}).get("clarity", 0)) for r in ok]
            ),
            "avg_safety": _safe_mean(
                [_to_float((r.get("judge") or {}).get("safety", 0)) for r in ok]
            ),
        },
        "retrieval": {
            "avg_recall": _safe_mean(
                [_to_float((r.get("retrieval") or {}).get("recall", 0)) for r in ok]
            ),
            "avg_precision": _safe_mean(
                [_to_float((r.get("retrieval") or {}).get("precision", 0)) for r in ok]
            ),
            "avg_f1": _safe_mean([_to_float((r.get("retrieval") or {}).get("f1", 0)) for r in ok]),
        },
    }

    by_task = _build_group_report(results, "task_type")
    by_modality = _build_group_report(results, "modality")
    errors = [r for r in results if r.get("error")]

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
        encoding="utf-8",
    )
    (cfg.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if errors:
        error_log_lines = [
            f"# eval error log generated at {datetime.now().isoformat(timespec='seconds')}",
        ]
        for e in errors:
            error_log_lines.append(
                f"{e.get('id','')}\t{e.get('modality','')}\t{e.get('task_type','')}\t{e.get('error','')}"
            )
        (cfg.output_dir / "eval_errors.log").write_text(
            "\n".join(error_log_lines) + "\n",
            encoding="utf-8",
        )
    _write_csv(cfg.output_dir / "report_by_task_type.csv", by_task)
    _write_csv(cfg.output_dir / "report_by_modality.csv", by_modality)
    _write_csv(
        cfg.output_dir / "report_overall.csv",
        [summary["counts"] | summary["scores"] | summary["retrieval"]],  # type: ignore[arg-type]
    )

    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(f"Summary saved to: {(cfg.output_dir / 'summary.json').resolve()}")


if __name__ == "__main__":
    main()
