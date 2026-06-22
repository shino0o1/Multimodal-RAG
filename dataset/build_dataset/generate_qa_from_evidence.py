#!/usr/bin/env python3
"""Generate synthetic QA pairs from usable evidence with SiliconFlow.

Input rows come from score_evidence_cache.py. The model is constrained to use
only the provided evidence snippets, and the output is written as JSONL for
later judge/filtering.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from qa_dedup import QuestionDeduplicator


DEFAULT_INPUT = "dataset/build_dataset/evidence_cache/scored/usable_evidence.jsonl"
DEFAULT_OUTPUT = "dataset/build_dataset/synthetic_qa/generated_qa_raw.jsonl"
DEFAULT_TRAIN_OUTPUT = "dataset/build_dataset/synthetic_qa/generated_qa_train.jsonl"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_API_KEY_PLACEHOLDER = "YOUR_SILICONFLOW_API_KEY"

TASK_TYPE_MAP = {
    "symptom_diagnosis": "症状识别",
    "occurrence_condition": "发生条件",
    "control_timing": "防治时期",
    "agronomic_control": "防治方法",
    "biological_control": "防治方法",
    "chemical_control": "防治方法",
    "safety_correction": "防治方法",
}
ALLOWED_TASK_TYPES = ["病虫害识别", "症状识别", "发生条件", "防治时期", "防治方法"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="输入的可用证据 JSONL 文件路径。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出原始生成结果 JSONL，保留 query、证据和 QA。")
    parser.add_argument("--train-output", default=DEFAULT_TRAIN_OUTPUT, help="输出可直接训练的 messages 格式 JSONL。")
    parser.add_argument("--no-train-output", action="store_true", help="只输出原始生成结果，不额外写训练 JSONL。")
    parser.add_argument("--api-key", default=os.getenv("SILICONFLOW_API_KEY", DEFAULT_API_KEY_PLACEHOLDER), help="SiliconFlow API Key；默认读取环境变量 SILICONFLOW_API_KEY。")
    parser.add_argument("--base-url", default=os.getenv("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL), help="SiliconFlow API 基础地址。")
    parser.add_argument("--model", default=os.getenv("SILICONFLOW_MODEL", DEFAULT_MODEL), help="调用的 SiliconFlow 模型名称。")
    parser.add_argument("--qa-per-row", type=int, default=2, help="每条证据输入最多生成几条 QA。")
    parser.add_argument("--max-results", type=int, default=4, help="每条证据最多取多少个搜索结果参与生成。")
    parser.add_argument("--max-snippets-per-result", type=int, default=3, help="每个搜索结果最多取多少个证据片段。")
    parser.add_argument("--max-snippet-chars", type=int, default=500, help="每个证据片段最多保留多少字符。")
    parser.add_argument("--temperature", type=float, default=0.4, help="模型采样温度；默认兼顾事实稳定性和问题表达多样性。")
    parser.add_argument("--dedup-threshold", type=float, default=0.88, help="同主题问题的近似去重阈值；设为 0 可关闭近似去重。")
    parser.add_argument("--max-tokens", type=int, default=1800, help="模型单次回复的最大 token 数。")
    parser.add_argument("--timeout", type=float, default=90.0, help="单次 API 请求超时时间，单位秒。")
    parser.add_argument("--delay", type=float, default=1.0, help="并发任务提交之间的等待时间，单位秒，用于降低限流风险。")
    parser.add_argument("--workers", type=int, default=8, help="并发 API 请求数；默认 8。")
    parser.add_argument("--rate-limit-workers", type=int, default=2, help="遇到 HTTP 429 后使用的并发请求数；默认 2。")
    parser.add_argument("--retries", type=int, default=2, help="遇到限流、超时或 5xx 错误时的重试次数。")
    parser.add_argument("--retry-base-delay", type=float, default=3.0, help="指数退避重试的基础等待时间，单位秒。")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条输入，便于小批量测试。")
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过原始输出中已经 status=ok 的 query_id。")
    parser.add_argument("--dry-run", action="store_true", help="只写请求 prompt，不实际调用 API。")
    parser.add_argument("--seed", type=int, default=7, help="随机种子，预留给后续多样化采样。")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            query_id = str(row.get("query_id") or "").strip()
            if query_id and row.get("status") in {"ok", "duplicate_generation"}:
                done.add(query_id)
    return done


def sync_train_output_from_raw(raw_path: Path, train_path: Path) -> tuple[int, bool]:
    if not raw_path.exists():
        return 0, False

    successful_rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(raw_path):
        query_id = str(row.get("query_id") or "").strip()
        if query_id and row.get("status") == "ok":
            successful_rows[query_id] = row

    expected_rows: list[dict[str, Any]] = []
    for row in successful_rows.values():
        expected_rows.extend(list(row.get("training_rows") or []))

    existing_count = len(read_jsonl(train_path)) if train_path.exists() else 0
    if existing_count == len(expected_rows):
        return existing_count, False

    write_jsonl(train_path, expected_rows)
    return len(expected_rows), True


def task_type_cn(raw_task_type: str, index: int) -> str:
    mapped = TASK_TYPE_MAP.get(raw_task_type, "")
    if mapped:
        return mapped
    # Use pest identification occasionally for broad/general rows without a direct mapping.
    return "病虫害识别" if index % 5 == 0 else "防治方法"


def truncate(text: str, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def compact_evidence(row: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    usable_results = list(row.get("usable_results") or [])
    usable_results.sort(
        key=lambda item: (
            int(item.get("source_score") or 0),
            len("".join(item.get("evidence_snippets") or [])),
        ),
        reverse=True,
    )
    evidence: list[dict[str, Any]] = []
    for item in usable_results[: args.max_results]:
        snippets = [
            truncate(snippet, args.max_snippet_chars)
            for snippet in (item.get("evidence_snippets") or [])[: args.max_snippets_per_result]
            if str(snippet or "").strip()
        ]
        if not snippets:
            continue
        evidence.append(
            {
                "title": item.get("title") or item.get("page_title") or "",
                "source_type": item.get("source_type", "unknown"),
                "source_score": item.get("source_score", 0),
                "snippets": snippets,
            }
        )
    return evidence


def collect_source_urls(row: dict[str, Any], max_results: int) -> list[str]:
    urls: list[str] = []
    for item in list(row.get("usable_results") or [])[: max_results]:
        url = str(item.get("url") or "").strip()
        if url and url not in urls:
            urls.append(url)
    return urls


def build_prompt(row: dict[str, Any], evidence: list[dict[str, Any]], task_cn: str, qa_count: int) -> list[dict[str, str]]:
    crop = row.get("crop", "")
    target = row.get("target", "")
    system = (
        "你是农业病虫害问答数据构造助手。你需要严格基于给定证据生成训练用问答，"
        "不得编造证据外信息(包括但不限于病因、药剂、剂量、时期等)。输出必须是合法 JSON。"
    )
    user_payload = {
        "目标作物": crop,
        "目标病虫害": target,
        "材料检索方向": task_cn,
        "生成后可选标签": ALLOWED_TASK_TYPES,
        "生成条数": qa_count,
        "证据": evidence,
        "输出要求": {
            "只输出 JSON 对象": True,
            "顶层字段": ["qa_pairs"],
            "qa_pairs字段": [
                "crop",
                "target",
                "question",
                "answer",
                "task_type",
                "evidence_snippets",
                "confidence",
                "notes",
            ],
            "问题构造要求": [
                "根据证据自由构造真实种植者可能提出的问题，不强制按照材料检索方向生成",
                "问题总体属于十字花科蔬菜病虫害问答，可涉及诊断、相似病虫害区分、防治建议、药剂用法或相关种植管理",
                "不同 QA 必须在问题意图、使用场景或表达方式上有明显差异，不能只做同义改写",
                "表达可以简洁、口语化、场景化，也可以在一句话中提出多个相互关联的问题",
                "只有证据明确支持时，问题才能加入具体地点、时间、生育阶段、症状、药剂或剂量等事实前提",
                "不得为了多样性编造证据中没有的作物、病虫害、症状、药剂或管理措施",
                "如果证据不足以支持指定数量，可以少生成，不要用重复或虚构问题凑数",
            ],
            "标签要求": [
                "根据每条 QA 的主要意图，在生成后可选标签中选择一个 task_type",
                "复合问题按答案的主要内容选择 task_type",
                "crop 和 target 应填写该问题实际涉及且证据明确支持的作物与病虫害",
            ],
            "answer要求": [
                "必须谨慎、基于证据",
                "不要出现证据外药剂、剂量、用药时期等各种虚构信息",
                "不要使用'根据百科'、'根据资料'、'根据现有资料'、'证据显示'等来源口吻",
                "除非证据片段明确提供且来源可靠，否则不要输出精确药剂剂量、倍液、浸种时间或安全间隔期",
                "涉及用药时，应提醒结合当地植保建议、登记范围和产品标签执行",
                "如果证据不足，回答中要说明需结合具体情况",
                "不要在答案中提到这是合成数据",
            ],
        },
    }
    user = (
        "请基于下面 JSON 证据生成农业问答训练样本。"
        "问题要自然，答案要完整但不要扩展到证据之外。\n\n"
        + json.dumps(user_payload, ensure_ascii=False, indent=2)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_siliconflow(
    messages: list[dict[str, str]],
    args: argparse.Namespace,
    rate_limit_event: threading.Event | None = None,
) -> dict[str, Any]:
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    last_error: Exception | None = None
    for attempt in range(max(0, args.retries) + 1):
        request = Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {args.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=args.timeout) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except HTTPError as exc:
            last_error = exc
            if exc.code == 429 and rate_limit_event is not None:
                rate_limit_event.set()
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= args.retries:
                raise
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= args.retries:
                raise
        time.sleep(args.retry_base_delay * (2**attempt))
    assert last_error is not None
    raise last_error


def parse_model_json(response: dict[str, Any]) -> dict[str, Any]:
    content = response["choices"][0]["message"]["content"]
    if isinstance(content, dict):
        return content
    text = str(content).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


SOURCE_PHRASES = [
    "根据百度百科，",
    "根据现有资料，",
    "根据资料，",
    "根据现有证据，",
    "证据显示，",
    "资料显示，",
]


def clean_answer_text(answer: str) -> str:
    answer = answer.strip()
    changed = True
    while changed:
        changed = False
        for phrase in SOURCE_PHRASES:
            if answer.startswith(phrase):
                answer = answer[len(phrase) :].lstrip()
                changed = True
    return answer


def normalize_qa_pairs(
    payload: dict[str, Any],
    source_crop: str,
    source_target: str,
    expected_task_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = payload.get("qa_pairs")
    if not isinstance(pairs, list):
        return [], []
    clean_pairs: list[dict[str, Any]] = []
    rejected_pairs: list[dict[str, Any]] = []
    for item in pairs:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = clean_answer_text(str(item.get("answer", "")).strip())
        if not question or not answer:
            continue
        crop = str(item.get("crop") or source_crop).strip()
        target = str(item.get("target") or item.get("disease_or_pest") or source_target).strip()
        raw_task_type = str(item.get("task_type", "")).strip()
        task_type = raw_task_type or expected_task_type
        if task_type not in ALLOWED_TASK_TYPES:
            rejected = dict(item)
            rejected["reject_reason"] = f"invalid_task_type:{task_type}"
            rejected_pairs.append(rejected)
            continue
        clean_pairs.append(
            {
                "crop": crop,
                "target": target,
                "question": question,
                "answer": answer,
                "task_type": task_type,
                "evidence_urls": [],
                "evidence_snippets": list(item.get("evidence_snippets") or []),
                "confidence": item.get("confidence", 0.0),
                "notes": str(item.get("notes", "")).strip(),
            }
        )
    return clean_pairs, rejected_pairs


def filter_duplicate_pairs(
    qa_pairs: list[dict[str, Any]],
    deduplicator: QuestionDeduplicator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for qa in qa_pairs:
        reason = deduplicator.add(
            str(qa.get("question") or ""),
            str(qa.get("crop") or ""),
            str(qa.get("target") or ""),
            str(qa.get("task_type") or ""),
        )
        if reason:
            rejected.append({**qa, "reject_reason": reason})
        else:
            accepted.append(qa)
    return accepted, rejected


def as_training_rows(source_row: dict[str, Any], qa_pairs: list[dict[str, Any]], task_cn: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qa in qa_pairs:
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": "你是农业病虫害问答助手，回答必须谨慎、基于证据。"},
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]},
                ],
                "source": "synthetic_DuckDuckgo",
                "metadata": {
                    "crop": qa.get("crop") or source_row.get("crop"),
                    "category": qa.get("task_type") or task_cn,
                    "disease_or_pest": qa.get("target") or source_row.get("target"),
                    "images": [],
                },
            }
        )
    return rows


def generate_for_row(
    row: dict[str, Any],
    row_index: int,
    args: argparse.Namespace,
    rate_limit_event: threading.Event | None = None,
) -> dict[str, Any]:
    evidence = compact_evidence(row, args)
    task_cn = task_type_cn(str(row.get("task_type", "")), row_index)
    messages = build_prompt(row, evidence, task_cn, args.qa_per_row)
    base = {
        "query_id": row.get("query_id"),
        "crop": row.get("crop"),
        "target": row.get("target"),
        "task_type": task_cn,
        "source_task_type": row.get("task_type"),
        "source_urls": collect_source_urls(row, args.max_results),
        "model": args.model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if not evidence:
        return {**base, "status": "skipped", "error": "empty_compact_evidence", "qa_pairs": []}
    if args.dry_run:
        return {**base, "status": "dry_run", "request_messages": messages, "qa_pairs": []}

    response = call_siliconflow(messages, args, rate_limit_event)
    payload = parse_model_json(response)
    qa_pairs, rejected_pairs = normalize_qa_pairs(
        payload,
        source_crop=str(row.get("crop") or "").strip(),
        source_target=str(row.get("target") or "").strip(),
        expected_task_type=task_cn,
    )
    return {
        **base,
        "status": "ok" if qa_pairs else "empty_generation",
        "qa_pairs": qa_pairs,
        "rejected_qa_pairs": rejected_pairs,
        "training_rows": as_training_rows(row, qa_pairs, task_cn),
    }


def process_row_request(
    row: dict[str, Any],
    row_index: int,
    args: argparse.Namespace,
    rate_limit_event: threading.Event,
) -> dict[str, Any]:
    query_id = str(row.get("query_id") or "").strip()
    try:
        return generate_for_row(row, row_index, args, rate_limit_event)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError) as exc:
        return {
            "query_id": query_id,
            "crop": row.get("crop"),
            "target": row.get("target"),
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


def finalize_output(
    row: dict[str, Any],
    output: dict[str, Any],
    args: argparse.Namespace,
    deduplicator: QuestionDeduplicator,
) -> dict[str, Any]:
    if output.get("status") != "ok" or args.dry_run:
        return output

    accepted_pairs, duplicate_pairs = filter_duplicate_pairs(
        list(output.get("qa_pairs") or []), deduplicator
    )
    output["qa_pairs"] = accepted_pairs
    output["rejected_qa_pairs"] = list(output.get("rejected_qa_pairs") or []) + duplicate_pairs
    output["training_rows"] = as_training_rows(
        row, accepted_pairs, str(output.get("task_type") or "")
    )
    if not accepted_pairs:
        output["status"] = "duplicate_generation"
    return output


def main() -> None:
    args = parse_args()
    if not args.dry_run and str(args.api_key or "").strip() == DEFAULT_API_KEY_PLACEHOLDER:
        raise SystemExit(
            "Missing API key. Set SILICONFLOW_API_KEY or pass --api-key YOUR_KEY."
        )

    random.seed(args.seed)
    input_path = Path(args.input)
    output_path = Path(args.output)
    train_output_path = Path(args.train_output)
    rows = read_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    if output_path.exists() and not args.resume:
        output_path.unlink()
    if train_output_path.exists() and not args.resume and not args.no_train_output:
        train_output_path.unlink()

    if args.resume and not args.no_train_output:
        synced_count, rebuilt = sync_train_output_from_raw(output_path, train_output_path)
        if rebuilt:
            print(
                json.dumps(
                    {
                        "event": "resume_train_rebuilt",
                        "train_rows": synced_count,
                        "train_output": str(train_output_path),
                    },
                    ensure_ascii=False,
                )
            )

    done_ids = load_done_ids(output_path) if args.resume else set()

    deduplicator = QuestionDeduplicator(args.dedup_threshold)
    if args.resume and train_output_path.exists() and not args.no_train_output:
        for existing_train_row in read_jsonl(train_output_path):
            deduplicator.add_training_row(existing_train_row)

    processed = 0
    skipped = 0
    errors = 0
    train_rows_written = 0
    pending_rows: list[tuple[int, dict[str, Any]]] = []
    for row_index, row in enumerate(rows, start=1):
        query_id = str(row.get("query_id") or "").strip()
        if query_id in done_ids:
            skipped += 1
            continue
        pending_rows.append((row_index, row))

    requested_workers = max(1, args.workers)
    reduced_workers = min(requested_workers, max(1, args.rate_limit_workers))
    effective_workers = requested_workers
    rate_limit_event = threading.Event()
    rate_limit_reduced = False
    next_row = 0
    futures: dict[Future[dict[str, Any]], tuple[int, dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=requested_workers) as executor:
        while next_row < len(pending_rows) or futures:
            if rate_limit_event.is_set() and effective_workers > reduced_workers:
                effective_workers = reduced_workers
                rate_limit_reduced = True
                print(
                    json.dumps(
                        {
                            "event": "rate_limit_detected",
                            "workers_reduced_to": effective_workers,
                        },
                        ensure_ascii=False,
                    )
                )

            while next_row < len(pending_rows) and len(futures) < effective_workers:
                row_index, row = pending_rows[next_row]
                future = executor.submit(
                    process_row_request, row, row_index, args, rate_limit_event
                )
                futures[future] = (row_index, row)
                next_row += 1
                if args.delay > 0 and next_row < len(pending_rows):
                    time.sleep(args.delay)
                if rate_limit_event.is_set() and effective_workers > reduced_workers:
                    break

            if not futures:
                continue

            completed, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            completed_items = sorted(
                ((future, *futures.pop(future)) for future in completed),
                key=lambda item: item[1],
            )
            for future, row_index, row in completed_items:
                query_id = str(row.get("query_id") or "").strip()
                try:
                    output = future.result()
                except Exception as exc:
                    output = {
                        "query_id": query_id,
                        "crop": row.get("crop"),
                        "target": row.get("target"),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }

                output = finalize_output(row, output, args, deduplicator)
                if output.get("status") == "error":
                    errors += 1
                write_jsonl_row(output_path, output)
                training_rows = output.get("training_rows") or []
                if training_rows and not args.no_train_output and not args.dry_run:
                    for train_row in training_rows:
                        write_jsonl_row(train_output_path, train_row)
                    train_rows_written += len(training_rows)
                processed += 1
                print(
                    json.dumps(
                        {
                            "index": row_index,
                            "query_id": query_id,
                            "status": output.get("status"),
                            "qa_pairs": len(output.get("qa_pairs") or []),
                            "train_rows": len(training_rows),
                        },
                        ensure_ascii=False,
                    )
                )

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "train_output": None if args.no_train_output else str(train_output_path),
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "train_rows_written": train_rows_written,
        "total_requested": len(rows),
        "dry_run": args.dry_run,
        "model": args.model,
        "workers_requested": requested_workers,
        "workers_final": effective_workers,
        "rate_limit_reduced": rate_limit_reduced,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
