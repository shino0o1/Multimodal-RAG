#!/usr/bin/env python3
"""Build a web evidence cache for prioritized dataset gap queries.

The script reads gap search queries, searches the web, fetches top result pages,
and writes one JSON object per query. It intentionally uses only the Python
standard library so it can run in the current project environment.
"""

from __future__ import annotations

import argparse
import codecs
import html
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, quote_plus, unquote, urljoin, urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_INPUT = "dataset/build_dataset/gaps/priority_search_queries_top120.jsonl"
DEFAULT_OUTPUT = "dataset/build_dataset/evidence_cache/search_results.jsonl"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
)
PREFERRED_SOURCE_HINTS = [
    "gov.cn",
    "edu.cn",
    "ac.cn",
    "农业农村",
    "农技",
    "植保",
    "农科院",
    "推广站",
    "农业科学院",
    "植物保护",
]
QUERY_NOISE_TERMS = [
    "农技推广",
    "植保站",
    "农科院",
    "农业农村",
    "最佳时间",
    "注意事项",
]
CROP_QUERY_ALIASES = {
    "大白菜": ["白菜", "娃娃菜"],
    "小白菜": ["青菜", "上海青"],
    "甘蓝": ["包菜", "卷心菜", "紫甘蓝"],
    "花椰菜": ["菜花", "花菜"],
    "青花菜": ["西兰花", "西蓝花"],
    "萝卜": ["白萝卜"],
    "油菜": ["油菜薹"],
    "芥菜": ["雪里蕻", "儿菜"],
}
TARGET_QUERY_ALIASES = {
    "小菜蛾": ["吊丝虫", "吊死虫"],
    "黄曲条跳甲": ["黄条跳甲", "跳甲"],
    "菜青虫": ["青菜虫", "菜粉蝶"],
    "根肿病": ["十字花科根肿病"],
    "霜霉病": ["霜霉"],
    "软腐病": ["细菌性软腐病"],
    "病毒病": ["花叶病毒病"],
}
TASK_QUERY_TERMS = {
    "symptom_diagnosis": ["症状", "识别"],
    "occurrence_condition": ["发生规律", "发生条件"],
    "control_timing": ["防治时期"],
    "agronomic_control": ["农业防治"],
    "biological_control": ["绿色防控"],
    "chemical_control": ["药剂防治"],
}
TEXT_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "td", "th"}
SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "nav", "footer"}
BAD_IMAGE_HINTS = [
    "logo",
    "icon",
    "sprite",
    "avatar",
    "copyright",
    "copy_rignt",
    "copy_right",
    "unsubscribe",
    "qrcode",
    "qr_code",
    "new.png",
]


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Gap query JSONL.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Evidence cache JSONL.")
    parser.add_argument("--provider", default="duckduckgo", choices=["duckduckgo"])
    parser.add_argument("--max-results", type=int, default=5)  # 每个搜索 query 最多保留多少条搜索结果
    parser.add_argument("--max-snippets", type=int, default=4) # 每个网页最多提取多少段正文证据
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--snippet-chars", type=int, default=500) # 每段证据文本最多保留多少字符
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N rows.")
    parser.add_argument("--resume", action="store_true", help="Append and skip existing query ids.")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="With --resume, reprocess existing rows whose status is error.",
    )
    parser.add_argument(
        "--retry-no-results",
        action="store_true",
        help="With --resume, reprocess existing rows whose status is no_results.",
    )
    parser.add_argument(
        "--fallback-queries",
        type=int,
        default=6,
        help="Maximum search query variants to try per gap row.",
    )
    parser.add_argument("--skip-fetch", action="store_true", help="Only cache search result titles/snippets/URLs.")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
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


def load_existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            query_id = str(row.get("query_id") or row.get("id") or "").strip()
            if query_id:
                rows[query_id] = row
    return rows


def should_skip_existing(existing: dict[str, Any], args: argparse.Namespace) -> bool:
    status = existing.get("status")
    if status == "error" and args.retry_errors:
        return False
    if status == "no_results" and args.retry_no_results:
        return False
    return True


def http_get(url: str, timeout: float, user_agent: str) -> tuple[str, str]:
    url = iri_to_uri(url)
    request = Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        charset = normalize_charset(response.headers.get_content_charset() or guess_charset(content_type))
        payload = response.read()
    return payload.decode(charset, errors="replace"), content_type


def normalize_charset(charset: str | None) -> str:
    raw = (charset or "utf-8").strip().strip('"').strip("'").lower()
    candidates = [item.strip() for item in re.split(r"[,;/\s]+", raw) if item.strip()]
    candidates.extend(["utf-8", "gb18030"])
    for candidate in candidates:
        try:
            codecs.lookup(candidate)
            return candidate
        except LookupError:
            continue
    return "utf-8"


def iri_to_uri(url: str) -> str:
    """Encode non-ASCII URL path/query characters before urllib requests."""
    parts = urlsplit(url.strip())
    path = quote(unquote(parts.path), safe="/%:@")
    query = quote(unquote(parts.query), safe="=&%:@/?+;,")
    fragment = quote(unquote(parts.fragment), safe="")
    return urlunsplit((parts.scheme, parts.netloc, path, query, fragment))


def guess_charset(content_type: str) -> str:
    match = re.search(r"charset=([\w.-]+)", content_type, flags=re.I)
    if match:
        return match.group(1)
    return "utf-8"


class DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._in_title = False
        self._in_snippet = False
        self._current_href = ""
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []
        self._pending: SearchResult | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        classes = attrs_dict.get("class", "")
        if tag == "a" and "result__a" in classes:
            self._in_title = True
            self._current_href = attrs_dict.get("href", "")
            self._current_title = []
        elif "result__snippet" in classes:
            self._in_snippet = True
            self._current_snippet = []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._current_title.append(data)
        elif self._in_snippet:
            self._current_snippet.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
            title = normalize_space("".join(self._current_title))
            url = normalize_result_url(self._current_href)
            if title and url:
                self._pending = SearchResult(title=title, url=url, snippet="")
                self.results.append(self._pending)
            self._current_href = ""
            self._current_title = []
        elif self._in_snippet and tag in {"a", "div", "span"}:
            snippet = normalize_space("".join(self._current_snippet))
            if snippet and self._pending and not self._pending.snippet:
                self._pending.snippet = snippet
            self._in_snippet = False
            self._current_snippet = []


def normalize_result_url(url: str) -> str:
    url = html.unescape(url or "").strip()
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        uddg = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(uddg)
    return url


def search_duckduckgo(query: str, max_results: int, timeout: float, user_agent: str) -> tuple[str, list[SearchResult]]:
    search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    page, _ = http_get(search_url, timeout=timeout, user_agent=user_agent)
    parser = DuckDuckGoParser()
    parser.feed(page)
    deduped: list[SearchResult] = []
    seen: set[str] = set()
    for result in parser.results:
        if result.url in seen:
            continue
        seen.add(result.url)
        deduped.append(result)
        if len(deduped) >= max_results:
            break
    return search_url, deduped


class PageEvidenceParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title = ""
        self.meta_description = ""
        self.images: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._text_tag_stack: list[str] = []
        self._buffer: list[str] = []
        self.blocks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            self._buffer = []
        elif tag == "meta":
            name = attrs_dict.get("name", "").lower()
            prop = attrs_dict.get("property", "").lower()
            if name == "description" or prop == "og:description":
                self.meta_description = normalize_space(attrs_dict.get("content", ""))
        elif tag == "img":
            src = (
                attrs_dict.get("src")
                or attrs_dict.get("data-src")
                or attrs_dict.get("data-original")
                or attrs_dict.get("data-lazy-src")
                or ""
            )
            if src:
                absolute = urljoin(self.base_url, html.unescape(src))
                if is_probable_image_url(absolute) and absolute not in self.images:
                    self.images.append(absolute)
        elif tag in TEXT_TAGS and self._skip_depth == 0:
            self._text_tag_stack.append(tag)
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title or self._text_tag_stack:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title" and self._in_title:
            self.title = normalize_space("".join(self._buffer))
            self._in_title = False
            self._buffer = []
        elif self._text_tag_stack and tag == self._text_tag_stack[-1]:
            text = normalize_space("".join(self._buffer))
            if len(text) >= 12:
                self.blocks.append(text)
            self._text_tag_stack.pop()
            self._buffer = []


def is_probable_image_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    if any(hint in path for hint in BAD_IMAGE_HINTS):
        return False
    return path.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")) or "image" in path


def fetch_page_evidence(
    url: str,
    query_terms: list[str],
    timeout: float,
    user_agent: str,
    max_snippets: int,
    max_images: int,
    snippet_chars: int,
) -> dict[str, Any]:
    page, content_type = http_get(url, timeout=timeout, user_agent=user_agent)
    parser = PageEvidenceParser(url)
    parser.feed(page)
    blocks = []
    if parser.meta_description:
        blocks.append(parser.meta_description)
    blocks.extend(parser.blocks)
    snippets = select_relevant_snippets(blocks, query_terms, max_snippets, snippet_chars)
    return {
        "fetch_status": "ok",
        "content_type": content_type,
        "page_title": parser.title,
        "meta_description": parser.meta_description,
        "evidence_snippets": snippets,
        "image_urls": parser.images[:max_images],
    }


def select_relevant_snippets(
    blocks: list[str], query_terms: list[str], max_snippets: int, snippet_chars: int
) -> list[str]:
    scored: list[tuple[int, int, str]] = []
    for index, block in enumerate(blocks):
        clean = normalize_space(block)
        if len(clean) < 12:
            continue
        score = sum(2 for term in query_terms if term and term in clean)
        score += sum(1 for word in ["症状", "防治", "发生", "用药", "药剂", "农业防治", "安全"] if word in clean)
        if score <= 0:
            continue
        scored.append((score, -index, truncate(clean, snippet_chars)))
    scored.sort(reverse=True)
    snippets: list[str] = []
    for _, _, snippet in scored:
        if snippet not in snippets:
            snippets.append(snippet)
        if len(snippets) >= max_snippets:
            break
    if not snippets:
        for block in blocks[:max_snippets]:
            clean = normalize_space(block)
            if clean:
                snippets.append(truncate(clean, snippet_chars))
    return snippets


def build_query_terms(row: dict[str, Any]) -> list[str]:
    terms = [
        str(row.get("crop", "")).strip(),
        str(row.get("target", "")).strip(),
        str(row.get("task_type", "")).strip(),
    ]
    terms.extend(str(row.get("query", "")).split())
    deduped: list[str] = []
    for term in terms:
        term = normalize_space(term)
        if term and term not in deduped:
            deduped.append(term)
    return deduped


def build_search_queries(row: dict[str, Any], max_queries: int) -> list[str]:
    crop = normalize_space(str(row.get("crop", "")))
    target = normalize_space(str(row.get("target", "")))
    task_type = normalize_space(str(row.get("task_type", "")))
    original = remove_query_noise(normalize_space(str(row.get("query", ""))))
    task_terms = TASK_QUERY_TERMS.get(task_type, [])

    candidates: list[str] = []
    add_query(candidates, original)
    add_query(candidates, " ".join([crop, target, *task_terms[:1]]))
    add_query(candidates, " ".join([crop, target]))
    for crop_alias in CROP_QUERY_ALIASES.get(crop, []):
        add_query(candidates, " ".join([crop_alias, target, *task_terms[:1]]))
        add_query(candidates, " ".join([crop_alias, target]))
    for target_alias in TARGET_QUERY_ALIASES.get(target, []):
        add_query(candidates, " ".join([crop, target_alias, *task_terms[:1]]))
        add_query(candidates, " ".join([crop, target_alias]))
    for crop_alias in CROP_QUERY_ALIASES.get(crop, [])[:2]:
        for target_alias in TARGET_QUERY_ALIASES.get(target, [])[:2]:
            add_query(candidates, " ".join([crop_alias, target_alias]))
    add_query(candidates, " ".join(["十字花科", target, *task_terms[:1]]))
    add_query(candidates, " ".join(["十字花科蔬菜", target]))
    return candidates[: max(1, max_queries)]


def add_query(candidates: list[str], query: str) -> None:
    cleaned = normalize_space(query)
    if cleaned and cleaned not in candidates:
        candidates.append(cleaned)


def remove_query_noise(query: str) -> str:
    cleaned = query
    for term in QUERY_NOISE_TERMS:
        cleaned = cleaned.replace(term, " ")
    return normalize_space(cleaned)


def source_score(url: str, title: str, snippet: str) -> int:
    text = f"{url}\n{title}\n{snippet}"
    score = 0
    for hint in PREFERRED_SOURCE_HINTS:
        if hint in text:
            score += 2
    parsed = urlparse(url)
    if parsed.netloc.endswith(".gov.cn") or ".gov.cn" in parsed.netloc:
        score += 4
    if parsed.netloc.endswith(".edu.cn") or ".edu.cn" in parsed.netloc:
        score += 2
    return score


def source_type(score: int) -> str:
    if score >= 4:
        return "preferred"
    if score >= 2:
        return "likely_relevant"
    return "unknown"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def build_cache_row(
    query_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    query = remove_query_noise(str(query_row.get("query", "")).strip())
    fetched_at = datetime.now(timezone.utc).isoformat()
    cache_row: dict[str, Any] = {
        "query_id": query_row.get("id"),
        "crop": query_row.get("crop"),
        "target": query_row.get("target"),
        "task_type": query_row.get("task_type"),
        "query": query,
        "priority_score": query_row.get("priority_score"),
        "source_policy": query_row.get("source_policy"),
        "provider": args.provider,
        "retrieved_at": fetched_at,
        "search_attempts": [],
        "results": [],
    }
    if not query:
        cache_row["status"] = "error"
        cache_row["error"] = "empty_query"
        return cache_row

    all_results: list[SearchResult] = []
    seen_urls: set[str] = set()
    last_error = ""
    for search_query in build_search_queries(query_row, args.fallback_queries):
        try:
            search_url, results = search_duckduckgo(
                search_query,
                max_results=args.max_results,
                timeout=args.timeout,
                user_agent=args.user_agent,
            )
            cache_row["search_attempts"].append(
                {
                    "query": search_query,
                    "search_url": search_url,
                    "status": "ok" if results else "no_results",
                    "results": len(results),
                }
            )
            for result in results:
                if result.url in seen_urls:
                    continue
                seen_urls.add(result.url)
                all_results.append(result)
                if len(all_results) >= args.max_results:
                    break
            if len(all_results) >= args.max_results:
                break
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = f"search_failed: {type(exc).__name__}: {exc}"
            cache_row["search_attempts"].append(
                {
                    "query": search_query,
                    "status": "error",
                    "error": last_error,
                }
            )
            continue

    if not all_results and last_error and all(
        attempt.get("status") == "error" for attempt in cache_row["search_attempts"]
    ):
        cache_row["status"] = "error"
        cache_row["error"] = last_error
        return cache_row
    cache_row["search_url"] = next(
        (
            attempt.get("search_url")
            for attempt in cache_row["search_attempts"]
            if attempt.get("search_url")
        ),
        "",
    )

    query_terms = build_query_terms(query_row)
    for rank, result in enumerate(all_results, start=1):
        score = source_score(result.url, result.title, result.snippet)
        item: dict[str, Any] = {
            "rank": rank,
            "title": result.title,
            "url": result.url,
            "search_snippet": result.snippet,
            "source_score": score,
            "source_type": source_type(score),
        }
        if not args.skip_fetch:
            try:
                item.update(
                    fetch_page_evidence(
                        result.url,
                        query_terms=query_terms,
                        timeout=args.timeout,
                        user_agent=args.user_agent,
                        max_snippets=args.max_snippets,
                        max_images=args.max_images,
                        snippet_chars=args.snippet_chars,
                    )
                )
            except (HTTPError, URLError, TimeoutError, OSError, UnicodeError, LookupError) as exc:
                item["fetch_status"] = "error"
                item["fetch_error"] = f"{type(exc).__name__}: {exc}"
                item["evidence_snippets"] = []
                item["image_urls"] = []
        cache_row["results"].append(item)

    cache_row["status"] = "ok" if cache_row["results"] else "no_results"
    return cache_row


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    query_rows = read_jsonl(input_path)
    if args.limit is not None:
        query_rows = query_rows[: max(0, args.limit)]

    existing_rows = load_existing_rows(output_path) if args.resume else {}
    retained_rows: list[dict[str, Any]] = []
    if args.resume:
        requested_ids = {str(row.get("id") or "").strip() for row in query_rows}
        retained_rows = [
            row
            for query_id, row in existing_rows.items()
            if query_id in requested_ids and should_skip_existing(row, args)
        ]
        write_jsonl(output_path, retained_rows)
    elif output_path.exists():
        output_path.unlink()

    processed = 0
    skipped = len(retained_rows)
    for index, query_row in enumerate(query_rows, start=1):
        query_id = str(query_row.get("id") or "").strip()
        existing = existing_rows.get(query_id)
        if existing and should_skip_existing(existing, args):
            continue
        cache_row = build_cache_row(query_row, args)
        write_jsonl_row(output_path, cache_row)
        processed += 1
        print(
            json.dumps(
                {
                    "index": index,
                    "query_id": query_id,
                    "status": cache_row.get("status"),
                    "results": len(cache_row.get("results", [])),
                },
                ensure_ascii=False,
            )
        )
        if args.delay > 0 and index < len(query_rows):
            time.sleep(args.delay)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "processed": processed,
        "skipped": skipped,
        "total_requested": len(query_rows),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
