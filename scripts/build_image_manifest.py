#!/usr/bin/env python3
"""Build an eval image manifest from images named crop_target_index.*."""

from __future__ import annotations

import argparse
import ast
import asyncio
import base64
import inspect
import json
import mimetypes
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


VISION_MODEL = "gemini-2.5-flash"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CABBAGE_PLACEHOLDER = "cabbage"
CRUCIFEROUS_CROP_CHOICES = [
    "青花菜",
    "花椰菜",
    "大白菜",
    "小白菜",
    "芥兰",
    "甘蓝",
    "芥菜",
    "萝卜",
    "小青菜",
    "芜菁",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate eval image manifest JSONL with a VLM."
    )
    parser.add_argument("--image-dir", required=True, help="Directory containing images.")
    parser.add_argument("--output", default="eval_images/manifest.jsonl")
    parser.add_argument("--model", default=VISION_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--relative-to",
        default=".",
        help="Write image_path relative to this directory. Default: project root/current dir.",
    )
    parser.add_argument(
        "--failed-output",
        default=None,
        help="Optional JSONL path for files that fail filename parsing or VLM generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    secrets = load_pipeline_api_defaults(PROJECT_ROOT / "pdf_rag_pipeline.py")
    api_key = secrets.get("api_key")
    base_url = secrets.get("base_url")
    if not api_key:
        raise RuntimeError("No api_key found in pdf_rag_pipeline.py")

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    images = sorted(path for path in image_dir.rglob("*") if is_image(path))
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    failed_output = Path(args.failed_output) if args.failed_output else output.with_suffix(".failed.jsonl")
    failed_output.parent.mkdir(parents=True, exist_ok=True)
    relative_to = Path(args.relative_to).resolve()

    rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                build_row,
                path,
                relative_to,
                args.model,
                api_key,
                base_url,
            ): path
            for path in images
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                rows.append(future.result())
                print(f"[ok] {path}")
            except Exception as exc:
                failed.append({"image_path": str(path), "reason": str(exc)})
                print(f"[failed] {path}: {exc}")

    rows.sort(key=lambda item: item["image_path"])
    write_jsonl(output, rows)
    write_jsonl(failed_output, failed)
    print(f"Manifest written: {output.resolve()} ({len(rows)} rows)")
    print(f"Failed written: {failed_output.resolve()} ({len(failed)} rows)")


def build_row(
    image_path: Path,
    relative_to: Path,
    model: str,
    api_key: str,
    base_url: Optional[str],
) -> Dict[str, Any]:
    crop, target, index = parse_filename(image_path)
    vlm = call_vlm(image_path, crop, target, model, api_key, base_url)
    resolved_crop = resolve_crop(crop, vlm)
    symptoms = string_list(vlm.get("symptoms"))
    notes = str(vlm.get("notes", "")).strip()
    visible_clues = string_list(vlm.get("visible_clues"))
    if not notes:
        notes = "；".join(visible_clues[:4])

    return {
        "image_path": to_manifest_path(image_path, relative_to),
        "labels": {
            "crop": resolved_crop,
            "target": target,
            "symptoms": symptoms,
        },
        "notes": notes,
        "metadata": {
            "filename_index": index,
            "filename_crop": crop,
            "vlm_model": model,
            "visible_clues": visible_clues,
            "crop_inferred": crop.lower() == CABBAGE_PLACEHOLDER,
            "target_match": bool(vlm.get("target_match", True)),
            "confidence": clamp_float(vlm.get("confidence"), 0.0, 1.0, 0.0),
        },
    }


def parse_filename(path: Path) -> Tuple[str, str, str]:
    parts = [part.strip() for part in re.split(r"[_\-]+", path.stem) if part.strip()]
    if len(parts) < 3:
        raise ValueError("filename must be 作物名_病虫害名_序号 or 作物名-病虫害名-序号")
    crop = parts[0].strip()
    target = parts[1].strip()
    index = "_".join(parts[2:]).strip()
    if not crop or not target or not index:
        raise ValueError("filename crop/target/index cannot be empty")
    return crop, target, index


def call_vlm(
    image_path: Path,
    crop: str,
    target: str,
    model: str,
    api_key: str,
    base_url: Optional[str],
) -> Dict[str, Any]:
    from lightrag.llm.openai import openai_complete_if_cache

    media_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    crop_requirement = (
        "文件名 crop 是 cabbage，表示只能确定为十字花科蔬菜。"
        "请你必须根据图片从以下列表中选择一个最可能的具体 crop："
        + "、".join(CRUCIFEROUS_CROP_CHOICES)
        + "。如果不确定，也必须选择最接近的一种，并降低 confidence。"
        if crop.lower() == CABBAGE_PLACEHOLDER
        else "crop 已由文件名给定，请沿用该 crop；如果图片明显不匹配，在 target_match/confidence 中体现。"
    )
    prompt = f"""
请根据图片和文件名标签，生成农业病虫害评测集 image manifest 字段。

文件名标签：
- crop: {crop}
- target: {target}

要求：
1. 只描述图片中可见的视觉线索，不要写防治建议、药剂、剂量。
2. symptoms 写成短词列表，例如 ["叶片孔洞", "幼虫取食"]。
3. 判断图片是否看起来与 crop/target 匹配，无法确定时 target_match=false 并降低 confidence。
4. {crop_requirement}
5. 只输出合法 JSON，不要 Markdown。

输出 JSON：
{{
  "crop": "具体作物名",
  "symptoms": ["可见症状或危害特征"],
  "visible_clues": ["具体可见线索"],
  "notes": "一句简短中文备注",
  "target_match": true,
  "confidence": 0.0
}}
""".strip()
    messages = [
        {
            "role": "system",
            "content": "你是农业病虫害图像标注助手，只输出合法JSON。",
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
    raw = resolve_maybe_awaitable(
        openai_complete_if_cache(
            model,
            "",
            system_prompt=None,
            history_messages=[],
            messages=messages,
            api_key=api_key,
            base_url=base_url,
        )
    )
    payload = extract_json_object(raw)
    if not payload:
        raise RuntimeError(f"VLM did not return JSON: {raw[:200]}")
    return payload


def resolve_maybe_awaitable(value: Any) -> str:
    if inspect.isawaitable(value):
        return str(asyncio.run(value))
    return str(value)


def resolve_crop(filename_crop: str, vlm_payload: Dict[str, Any]) -> str:
    if filename_crop.lower() != CABBAGE_PLACEHOLDER:
        return filename_crop
    predicted = str(vlm_payload.get("crop", "")).strip()
    if predicted in CRUCIFEROUS_CROP_CHOICES:
        return predicted
    return "甘蓝"


def load_pipeline_api_defaults(path: Path) -> Dict[str, str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse {path}: {exc}") from exc

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


def extract_json_object(text: str) -> Dict[str, Any]:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    start = stripped.find("{")
    if start < 0:
        return {}
    depth = 0
    for idx in range(start, len(stripped)):
        if stripped[idx] == "{":
            depth += 1
        elif stripped[idx] == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(stripped[start : idx + 1])
                except json.JSONDecodeError:
                    return {}
                return payload if isinstance(payload, dict) else {}
    return {}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return max(low, min(high, parsed))


def to_manifest_path(path: Path, relative_to: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(relative_to).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
