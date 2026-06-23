#!/usr/bin/env python3
"""Verify Bailian web search and print the answer together with its sources."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from repair_qa_with_bailian import (
    DEFAULT_API_KEY_PLACEHOLDER,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    chat_completions_url,
    format_http_error,
)


DEFAULT_QUERY = (
    "请联网搜索农业农村部官网最近发布的一条植物保护或农药管理相关信息，"
    "给出标题、发布日期、简要内容，并在回答末尾列出来源标题和完整 URL。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Question that requires current web information.")
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY", DEFAULT_API_KEY_PLACEHOLDER))
    parser.add_argument("--base-url", default=os.getenv("BAILIAN_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("BAILIAN_MODEL", DEFAULT_MODEL))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument(
        "--output",
        default="synthetic_qa/bailian_search_verification.json",
        help="File used to retain the complete raw response.",
    )
    return parser.parse_args()


def call_bailian_with_search(args: argparse.Namespace) -> dict[str, Any]:
    """Use the same urllib/OpenAI-compatible request method as the repair script."""
    body = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是联网搜索验证助手。必须先搜索互联网再回答，不得仅凭模型记忆。"
                    "回答末尾必须列出实际使用的来源标题和完整 URL。"
                ),
            },
            {"role": "user", "content": args.query},
        ],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "enable_search": True,
        "search_options": {
            "forced_search": True,
            "enable_source": True,
            "enable_citation": True,
            "citation_format": "[<number>]",
        },
    }
    request = Request(
        chat_completions_url(args.base_url),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(format_http_error(exc)) from exc
    except (URLError, socket.timeout, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Bailian request failed: {exc}") from exc


def extract_answer(data: dict[str, Any]) -> str:
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected API response: {json.dumps(data, ensure_ascii=False)[:1000]}") from exc


def extract_metadata_sources(data: Any) -> list[dict[str, str]]:
    """Collect URL-bearing source records regardless of their response nesting."""
    collected: list[dict[str, str]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            citation = node.get("url_citation")
            if isinstance(citation, dict):
                walk(citation)
            url = node.get("url") or node.get("link")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                title = node.get("title") or node.get("name") or node.get("site_name") or "未提供标题"
                collected.append({"title": str(title), "url": url})
            for key, value in node.items():
                if key != "url_citation":
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for source in collected:
        if source["url"] not in seen:
            seen.add(source["url"])
            unique.append(source)
    return unique


def extract_answer_urls(answer: str) -> list[str]:
    urls = re.findall(r"https?://[^\s<>\]\[()，。]+", answer)
    return list(dict.fromkeys(url.rstrip(".,;:、") for url in urls))


def main() -> int:
    args = parse_args()
    if args.api_key == DEFAULT_API_KEY_PLACEHOLDER:
        print("ERROR: set DASHSCOPE_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    try:
        raw_response = call_bailian_with_search(args)
        answer = extract_answer(raw_response)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    metadata_sources = extract_metadata_sources(raw_response)
    answer_urls = extract_answer_urls(answer)
    result = {
        "query": args.query,
        "model": args.model,
        "forced_search": True,
        "answer": answer,
        "metadata_sources": metadata_sources,
        "answer_urls": answer_urls,
        "search_metadata_verified": bool(metadata_sources),
        "raw_response": raw_response,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== 模型回答 ===")
    print(answer)
    print("\n=== API 返回的来源 ===")
    if metadata_sources:
        for index, source in enumerate(metadata_sources, start=1):
            print(f"[{index}] {source['title']}\n    {source['url']}")
        print("\n验证结果：通过，API 响应中包含可核验的来源元数据。")
    else:
        print("未发现来源元数据。回答中的 URL 不能单独证明 API 确实执行了搜索。")
        if answer_urls:
            print("回答中出现的 URL：")
            for url in answer_urls:
                print(f"- {url}")
        print("验证结果：未通过。请查看保存的 raw_response 确认当前模型返回结构。")
    print(f"\n完整结果已保存：{output_path}")
    return 0 if metadata_sources else 1


if __name__ == "__main__":
    raise SystemExit(main())
