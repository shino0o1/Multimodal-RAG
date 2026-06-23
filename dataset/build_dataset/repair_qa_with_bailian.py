#!/usr/bin/env python3
"""Repair existing QA datasets with Bailian web-search-enabled LLM.

The script reads one or more existing QA JSONL files, asks a Bailian/OpenAI
compatible chat model to verify and repair each sample with web search enabled,
and writes exactly two artifacts:

1. a complete repaired QA JSONL dataset; and
2. a change log JSONL containing only modified/removed samples.

It intentionally does not write evidence caches or add evidence metadata to the
training rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OUTPUT = "dataset/build_dataset/synthetic_qa/bailian_repaired_full_qa.jsonl"
DEFAULT_CHANGE_LOG = "dataset/build_dataset/synthetic_qa/bailian_change_log.jsonl"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
DEFAULT_API_KEY_PLACEHOLDER = "YOUR_DASHSCOPE_API_KEY"
SYSTEM_ROLE_CONTENT = "你是农业病虫害问答助手，回答必须谨慎、基于证据。"


PRINT_LOCK = threading.Lock()


class BailianError(RuntimeError):
    """Raised when Bailian API call or response parsing fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Existing QA JSONL file. Can be passed multiple times.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Complete repaired QA JSONL output path.")
    parser.add_argument("--change-log", default=DEFAULT_CHANGE_LOG, help="JSONL change log for modified/removed rows only.")
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY", DEFAULT_API_KEY_PLACEHOLDER), help="Bailian/DashScope API key. Defaults to DASHSCOPE_API_KEY.")
    parser.add_argument("--base-url", default=os.getenv("BAILIAN_BASE_URL", DEFAULT_BASE_URL), help="OpenAI-compatible base URL.")
    parser.add_argument("--model", default=os.getenv("BAILIAN_MODEL", DEFAULT_MODEL), help="Bailian model name.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Low temperature is recommended for repair stability.")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request read timeout in seconds. Web search can be slow; increase this for live Bailian calls.")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay after each row in a worker, in seconds.")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent API requests. Default: 4.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Rows per checkpoint batch. Defaults to --workers.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume successful rows from the checkpoint file.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint JSON path. Defaults to <output>.checkpoint.json.",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-base-delay", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows across all inputs.")
    parser.add_argument("--start", type=int, default=0, help="Skip the first N rows across all inputs.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call API; copy input rows and emit no changes.")
    parser.add_argument(
        "--on-error",
        choices=["keep", "stop"],
        default="keep",
        help="What to do when a row still fails after retries. Default keeps the original row and continues.",
    )
    parser.add_argument("--keep-removed-in-output", action="store_true", help="Keep samples marked remove in the complete output instead of dropping them.")
    return parser.parse_args()


def read_jsonl_with_source(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_str in paths:
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append({"row": row, "source_file": str(path), "source_line": line_no})
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def selection_signature(selected: list[dict[str, Any]], args: argparse.Namespace) -> str:
    digest = hashlib.sha256()
    identity = {
        "start": args.start,
        "limit": args.limit,
        "inputs": [str(Path(path).resolve()) for path in args.input],
    }
    digest.update(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for item in selected:
        digest.update(str(item["source_line"]).encode("ascii"))
        digest.update(json.dumps(item["row"], ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def checkpoint_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "keep_removed_in_output": args.keep_removed_in_output,
    }


def extract_messages(row: dict[str, Any]) -> tuple[str, str, str]:
    system = ""
    question = ""
    answer = ""
    for message in row.get("messages") or []:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "system" and not system:
            system = content
        elif role == "user" and not question:
            question = content
        elif role == "assistant":
            answer = content
    return system, question, answer


def set_qa(row: dict[str, Any], question: str, answer: str, category: str | None = None) -> dict[str, Any]:
    repaired = deepcopy(row)
    messages = repaired.get("messages") or []
    has_system = any(message.get("role") == "system" for message in messages)
    if not has_system:
        messages.insert(0, {"role": "system", "content": SYSTEM_ROLE_CONTENT})

    user_set = False
    assistant_set = False
    for message in messages:
        if message.get("role") == "user" and not user_set:
            message["content"] = question
            user_set = True
        elif message.get("role") == "assistant":
            message["content"] = answer
            assistant_set = True
            break
    if not user_set:
        messages.append({"role": "user", "content": question})
    if not assistant_set:
        messages.append({"role": "assistant", "content": answer})
    repaired["messages"] = messages

    if category:
        metadata = repaired.get("metadata")
        if isinstance(metadata, dict):
            metadata["category"] = category
        elif "category" in repaired:
            repaired["category"] = category
    return repaired


def row_category(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("category") or "")
    return str(row.get("category") or "")


def build_repair_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    _, question, answer = extract_messages(row)
    payload = {
        "question": question,
        "answer": answer,
        "category": row_category(row),
        "crop": (row.get("metadata") or {}).get("crop", "") if isinstance(row.get("metadata"), dict) else "",
        "disease_or_pest": (row.get("metadata") or {}).get("disease_or_pest", "") if isinstance(row.get("metadata"), dict) else "",
    }
    system = (
        "你是农业病虫害问答数据集纠偏助手。你需要联网搜索相关农业资料，"
        "检查并必要时修正给定 QA。只输出合法 JSON，不要输出 Markdown。"
    )
    user = {
        "任务": "检查并纠偏已有农业病虫害 QA。",
        "输入QA": payload,
        "纠偏规则": [
            "如果原 QA 基本正确且信息充分，返回 action=keep，并保持原问题和原答案。",
            "如果原回答存在事实错误、遗漏关键信息、回答过于空泛、分类方向不匹配，请返回 action=modify，并给出修正后的 question、answer、category。",
            "如果原 QA 明显错误且无法通过联网搜索获得可靠依据修正，请返回 action=remove。",
            "如果搜索结果中明确提到地区，回答中要体现地区限定；如果没有提到地区，不要强行添加地区限定。",
            "防治时期类问题不能只写季节，应尽量写生育期、发病初期、病害发生前、卵孵化盛期、低龄幼虫期等更具体时期。",
            "在说明药剂使用和药剂效果时，要严格遵循已有证据；没有证据时不要编造药剂、剂量、倍数、防效或安全间隔期。",
            "不要输出证据 URL、证据摘要或额外 metadata。",
        ],
        "输出JSON格式": {
            "action": "keep | modify | remove",
            "question": "最终问题文本。keep 时应与原问题一致。",
            "answer": "最终答案文本。keep 时应与原答案一致；remove 时可为空字符串。",
            "category": "病虫害识别 | 症状识别 | 发生条件 | 防治时期 | 防治方法，或保持原分类",
            "reason": "modify/remove 时简要说明修改或删除原因；keep 时为空字符串",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def chat_completions_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def format_http_error(error: HTTPError) -> str:
    try:
        body = error.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    if body:
        return f"HTTP {error.code} {error.reason}: {body[:1000]}"
    return f"HTTP {error.code} {error.reason}"


def call_bailian(messages: list[dict[str, str]], args: argparse.Namespace) -> str:
    body = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "enable_search": True,
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(
        chat_completions_url(args.base_url),
        data=payload,
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        raise BailianError(format_http_error(exc)) from exc
    except socket.timeout as exc:
        raise BailianError(f"Request timed out after {args.timeout:g}s. Try increasing --timeout or lowering --limit.") from exc
    data = json.loads(raw)
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise BailianError(f"Unexpected API response: {raw[:1000]}") from exc


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def progress(message: str, *, error: bool = False) -> None:
    """Print one complete progress line without concurrent output interleaving."""
    with PRINT_LOCK:
        print(message, file=sys.stderr if error else sys.stdout, flush=True)


def repair_row(
    row: dict[str, Any],
    args: argparse.Namespace,
    *,
    retry_label: str = "",
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            content = call_bailian(build_repair_prompt(row), args)
            result = extract_json_object(content)
            action = str(result.get("action") or "").strip().lower()
            if action not in {"keep", "modify", "remove"}:
                raise BailianError(f"Invalid action: {action!r}")
            result["action"] = action
            return result
        except (URLError, TimeoutError, socket.timeout, json.JSONDecodeError, BailianError) as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            retry_delay = args.retry_base_delay * (2**attempt)
            progress(
                f"{retry_label} RETRY attempt={attempt + 2}/{args.retries + 1} "
                f"in={retry_delay:g}s error={exc}",
                error=True,
            )
            time.sleep(retry_delay)
    raise BailianError(f"Failed to repair row after retries: {last_error}")


def process_item(
    ordinal: int,
    total: int,
    dataset_index: int,
    item: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Process one sourced row and return its result for ordered assembly."""
    label = f"[{ordinal}/{total}]"
    source = f"{item['source_file']}:{item['source_line']}"
    started = time.monotonic()
    progress(f"{label} START row={dataset_index} source={source}")

    try:
        if args.dry_run:
            result = None
            action = "dry-run"
        else:
            result = repair_row(item["row"], args, retry_label=label)
            action = result["action"]
        elapsed = time.monotonic() - started
        progress(f"{label} DONE  row={dataset_index} action={action} elapsed={elapsed:.1f}s")
        return {"result": result, "error": None}
    except BailianError as exc:
        elapsed = time.monotonic() - started
        progress(
            f"{label} ERROR row={dataset_index} elapsed={elapsed:.1f}s "
            f"source={source} error={exc}",
            error=True,
        )
        return {"result": None, "error": exc}
    finally:
        if args.delay > 0 and not args.dry_run:
            time.sleep(args.delay)


def make_change_log(
    *,
    change_type: str,
    source_file: str,
    source_line: int,
    old_question: str,
    old_answer: str,
    new_question: str,
    new_answer: str,
    reason: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "change_type": change_type,
        "source_file": source_file,
        "source_line": source_line,
        "question": old_question,
        "old_answer": old_answer,
        "reason": reason,
    }
    if change_type == "modified":
        row["new_question"] = new_question
        row["new_answer"] = new_answer
    return row


def build_artifacts(
    selected: list[dict[str, Any]],
    outcomes: list[dict[str, Any] | None],
    processed_at: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build ordered artifacts from all results currently available."""
    output_rows: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []
    for position, item in enumerate(selected):
        outcome = outcomes[position]
        if outcome is None:
            continue
        row = item["row"]
        _, old_question, old_answer = extract_messages(row)
        if args.dry_run or outcome["error"] is not None:
            output_rows.append(row)
            continue

        result = outcome["result"]
        if result is None:
            raise RuntimeError(f"Missing repair result for selected row {position + 1}")
        action = result["action"]
        new_question = str(result.get("question") or old_question).strip()
        new_answer = str(result.get("answer") or "").strip()
        new_category = str(result.get("category") or row_category(row)).strip()
        reason = str(result.get("reason") or "").strip()

        if action == "remove":
            if args.keep_removed_in_output:
                output_rows.append(row)
            change = make_change_log(
                change_type="removed",
                source_file=item["source_file"],
                source_line=item["source_line"],
                old_question=old_question,
                old_answer=old_answer,
                new_question="",
                new_answer="",
                reason=reason or "百炼联网模型判断该样本无法可靠修正。",
            )
            change["processed_at"] = processed_at
            change_rows.append(change)
        elif action == "modify":
            output_rows.append(set_qa(row, new_question or old_question, new_answer or old_answer, new_category))
            change = make_change_log(
                change_type="modified",
                source_file=item["source_file"],
                source_line=item["source_line"],
                old_question=old_question,
                old_answer=old_answer,
                new_question=new_question or old_question,
                new_answer=new_answer or old_answer,
                reason=reason or "百炼联网模型修正了该样本。",
            )
            change["processed_at"] = processed_at
            change_rows.append(change)
        else:
            output_rows.append(row)
    return output_rows, change_rows


def load_checkpoint(
    path: Path,
    expected_signature: str,
    expected_config: dict[str, Any],
    total: int,
) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BailianError(f"Cannot read checkpoint {path}: {exc}") from exc
    if state.get("version") != 1:
        raise BailianError(f"Unsupported checkpoint version in {path}")
    if state.get("selection_signature") != expected_signature:
        raise BailianError("Checkpoint does not match the selected input rows. Run without --resume to restart.")
    if state.get("config") != expected_config:
        raise BailianError("Checkpoint does not match the current model/output configuration. Run without --resume to restart.")
    completed = state.get("completed")
    if not isinstance(completed, dict):
        raise BailianError(f"Invalid completed results in checkpoint {path}")
    for key, result in completed.items():
        if not key.isdigit() or not 0 <= int(key) < total or not isinstance(result, dict):
            raise BailianError(f"Invalid completed row {key!r} in checkpoint {path}")
    return state


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        print("ERROR: --workers must be at least 1.", file=sys.stderr)
        return 2
    if args.batch_size is not None and args.batch_size < 1:
        print("ERROR: --batch-size must be at least 1.", file=sys.stderr)
        return 2
    if args.delay < 0:
        print("ERROR: --delay cannot be negative.", file=sys.stderr)
        return 2
    if not args.dry_run and args.api_key == DEFAULT_API_KEY_PLACEHOLDER:
        print("ERROR: --api-key or DASHSCOPE_API_KEY is required unless --dry-run is used.", file=sys.stderr)
        return 2

    sourced_rows = read_jsonl_with_source(args.input)
    selected = sourced_rows[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    total = len(selected)
    worker_count = min(args.workers, total) if total else 1
    batch_size = args.batch_size or args.workers
    checkpoint_path = Path(args.checkpoint or f"{args.output}.checkpoint.json")
    signature = selection_signature(selected, args)
    config = checkpoint_config(args)
    outcomes: list[dict[str, Any] | None] = [None] * total

    if args.resume:
        if not checkpoint_path.exists():
            print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
            return 2
        try:
            state = load_checkpoint(checkpoint_path, signature, config, total)
        except BailianError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        processed_at = str(state.get("processed_at") or datetime.now(timezone.utc).isoformat())
        for key, result in state["completed"].items():
            outcomes[int(key)] = {"result": result, "error": None}
        progress(f"Resumed {len(state['completed'])}/{total} successful rows from {checkpoint_path}")
    else:
        processed_at = datetime.now(timezone.utc).isoformat()
        state = {
            "version": 1,
            "selection_signature": signature,
            "config": config,
            "processed_at": processed_at,
            "completed": {},
        }
        if not args.dry_run:
            write_json(checkpoint_path, state)

    pending_positions = [position for position, outcome in enumerate(outcomes) if outcome is None]
    progress(
        f"Processing {len(pending_positions)} pending QA rows with {worker_count} worker(s), "
        f"batch_size={batch_size}..."
    )

    def save_batch() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not args.dry_run:
            write_json(checkpoint_path, state)
        output_rows, change_rows = build_artifacts(selected, outcomes, processed_at, args)
        write_jsonl(Path(args.output), output_rows)
        write_jsonl(Path(args.change_log), change_rows)
        progress(
            f"CHECKPOINT saved={len(state['completed'])}/{total} "
            f"output_rows={len(output_rows)} changes={len(change_rows)} path={checkpoint_path}"
        )
        return output_rows, change_rows

    if args.resume:
        output_rows, change_rows = save_batch()
    else:
        output_rows, change_rows = [], []
        if not pending_positions:
            output_rows, change_rows = save_batch()

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="bailian-repair") as executor:
        for batch_start in range(0, len(pending_positions), batch_size):
            batch = pending_positions[batch_start : batch_start + batch_size]
            future_positions: dict[Future[dict[str, Any]], int] = {}
            for position in batch:
                future = executor.submit(
                    process_item,
                    position + 1,
                    total,
                    args.start + position + 1,
                    selected[position],
                    args,
                )
                future_positions[future] = position

            first_error: BailianError | None = None
            for future in as_completed(future_positions):
                position = future_positions[future]
                outcome = future.result()
                outcomes[position] = outcome
                if outcome["error"] is None and not args.dry_run:
                    state["completed"][str(position)] = outcome["result"]
                elif outcome["error"] is not None and first_error is None:
                    first_error = outcome["error"]

            output_rows, change_rows = save_batch()
            if first_error is not None and args.on_error == "stop":
                raise first_error

    print(f"Wrote complete QA rows: {len(output_rows)} -> {args.output}")
    print(f"Wrote change log rows: {len(change_rows)} -> {args.change_log}")
    failed_count = sum(outcome is not None and outcome["error"] is not None for outcome in outcomes)
    if failed_count:
        progress(
            f"WARNING: {failed_count} row(s) failed and remain pending in the checkpoint; "
            "run again with --resume to retry them.",
            error=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

