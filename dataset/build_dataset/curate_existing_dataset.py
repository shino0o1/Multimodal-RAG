#!/usr/bin/env python3
"""Curate the merged agriculture QA dataset for cruciferous vegetable SFT.

This is the first deterministic pass before any LLM judge or web-based
synthetic generation. It retains every QA pair related to cruciferous
vegetables, while topic type and quality signals are labels rather than hard
filters.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CRUCIFEROUS_ALIASES: dict[str, list[str]] = {
    "十字花科蔬菜": ["十字花科蔬菜", "十字花科", "芸薹属蔬菜", "白菜类蔬菜", "甘蓝类蔬菜"],
    "大白菜": ["大白菜", "白菜", "结球白菜", "黄心白菜", "娃娃菜"],
    "小白菜": ["小白菜", "青菜", "小青菜", "上海青", "油白菜", "瓢儿白", "奶白菜", "青梗菜"],
    "甘蓝": ["甘蓝", "卷心菜", "圆白菜", "包菜", "莲花白", "结球甘蓝", "紫甘蓝"],
    "花椰菜": ["花椰菜", "菜花", "花菜"],
    "青花菜": ["青花菜", "西兰花", "西蓝花", "绿花菜"],
    "萝卜": ["萝卜", "白萝卜", "青萝卜", "水萝卜", "樱桃萝卜", "心里美"],
    "芥菜": ["芥菜", "雪里蕻", "雪菜", "榨菜", "大头菜", "儿菜"],
    "芥蓝": ["芥蓝", "芥兰"],
    "油菜": ["油菜", "油菜薹", "菜薹", "红菜薹", "菜心"],
    "苤蓝": ["苤蓝", "球茎甘蓝"],
    "芜菁": ["芜菁", "蔓菁", "圆根"],
    "羽衣甘蓝": ["羽衣甘蓝"],
    "抱子甘蓝": ["抱子甘蓝", "孢子甘蓝"],
    "乌塌菜": ["乌塌菜", "塌菜"],
}

NON_CRUCIFEROUS_FALSE_FRIENDS = [
    "胡萝卜",
    "番茄",
    "西红柿",
    "黄瓜",
    "辣椒",
    "茄子",
    "芸豆",
    "豆角",
    "玉米",
    "小麦",
    "水稻",
    "棉花",
    "苹果",
    "葡萄",
    "草莓",
]

DISEASE_KEYWORDS = [
    "病",
    "霜霉",
    "黑腐",
    "软腐",
    "根肿",
    "菌核",
    "病毒",
    "猝倒",
    "立枯",
    "炭疽",
    "白锈",
    "叶斑",
    "灰霉",
    "枯萎",
    "腐烂",
    "死苗",
    "烂根",
]

PEST_KEYWORDS = [
    "虫",
    "蚜",
    "菜青虫",
    "小菜蛾",
    "夜蛾",
    "跳甲",
    "蓟马",
    "粉虱",
    "斑潜蝇",
    "潜叶蝇",
    "螨",
    "蜗牛",
    "蛞蝓",
    "地蛆",
    "蝼蛄",
    "蛴螬",
]

TARGET_ALIASES: dict[str, list[str]] = {
    "霜霉病": ["霜霉病", "霜霉"],
    "黑腐病": ["黑腐病"],
    "软腐病": ["软腐病", "细菌性软腐病"],
    "根肿病": ["根肿病"],
    "菌核病": ["菌核病"],
    "病毒病": ["病毒病", "花叶病毒病", "病毒"],
    "白锈病": ["白锈病", "白锈"],
    "黑斑病": ["黑斑病", "假黑斑病"],
    "炭疽病": ["炭疽病", "炭疽"],
    "灰霉病": ["灰霉病", "灰霉"],
    "叶斑病": ["叶斑病", "叶斑"],
    "角斑病": ["角斑病", "细菌性角斑病"],
    "猝倒病": ["猝倒病", "猝倒"],
    "立枯病": ["立枯病", "立枯"],
    "干烧心": ["干烧心", "烧心病"],
    "缺硼症": ["缺硼症", "缺硼"],
    "菜青虫": ["菜青虫", "青菜虫", "菜粉蝶"],
    "小菜蛾": ["小菜蛾", "吊丝虫", "吊死虫", "菜蛾"],
    "蚜虫": ["蚜虫", "菜蚜"],
    "黄曲条跳甲": ["黄曲条跳甲", "黄条跳甲", "跳甲"],
    "甜菜夜蛾": ["甜菜夜蛾"],
    "斜纹夜蛾": ["斜纹夜蛾"],
    "甘蓝夜蛾": ["甘蓝夜蛾"],
    "菜螟": ["菜螟", "钻心虫"],
    "蓟马": ["蓟马"],
    "粉虱": ["粉虱", "白粉虱", "烟粉虱"],
    "地蛆": ["地蛆", "根蛆"],
    "蜗牛": ["蜗牛"],
    "蛞蝓": ["蛞蝓"],
    "杂草": ["杂草", "阔叶草", "阔叶杂草"],
}

TARGET_ALIAS_PAIRS = sorted(
    (
        (alias, canonical)
        for canonical, aliases in TARGET_ALIASES.items()
        for alias in aliases
    ),
    key=lambda item: len(item[0]),
    reverse=True,
)

CROP_TARGET_PREFIXES = sorted(
    {
        alias
        for aliases in CRUCIFEROUS_ALIASES.values()
        for alias in aliases
        if len(alias) >= 2
    },
    key=len,
    reverse=True,
)

GENERIC_TARGETS = {
    "",
    "病",
    "虫",
    "病害",
    "虫害",
    "病虫害",
    "多种病害",
    "多种虫害",
    "多种病虫害",
    "未知",
    "不详",
    "无",
}

CONTROL_KEYWORDS = [
    "防治",
    "防控",
    "预防",
    "治疗",
    "用药",
    "药剂",
    "农药",
    "杀虫",
    "杀菌",
    "喷",
    "灌根",
    "拌种",
    "冲施",
    "怎么办",
    "如何处理",
    "怎么治",
    "怎么防",
    "控制",
    "绿色防控",
]

WEAK_TOPIC_KEYWORDS = [
    "品种",
    "选种",
    "栽培",
    "种植",
    "定植",
    "播种",
    "育苗",
    "施肥",
    "底肥",
    "追肥",
    "水肥",
    "浇水",
    "采收",
    "冻害",
    "低温",
    "高温",
    "干旱",
    "涝害",
    "气象",
    "灾害",
    "裂球",
    "抽薹",
]

SAFETY_REVIEW_KEYWORDS = [
    "剧毒",
    "高毒",
    "禁用",
    "甲胺磷",
    "对硫磷",
    "甲基对硫磷",
    "久效磷",
    "磷胺",
    "六六六",
    "滴滴涕",
    "3911",
    "1605",
]

TASK_TYPE_RULES: list[tuple[str, list[str]]] = [
    ("symptom_diagnosis", ["症状", "表现", "识别", "判断", "是不是", "什么病", "什么虫", "诊断"]),
    ("occurrence_condition", ["原因", "为什么", "发生条件", "高发", "流行", "诱因"]),
    ("control_timing", ["时期", "什么时候", "最佳时间", "用药时间", "防治时期"]),
    ("chemical_control", ["药", "农药", "药剂", "喷", "灌根", "拌种", "杀虫剂", "杀菌剂"]),
    ("agronomic_control", ["轮作", "清园", "通风", "排水", "密度", "田间管理", "农业防治"]),
    ("biological_control", ["生物防治", "天敌", "菌剂", "微生物", "枯草芽孢杆菌"]),
    ("safety_correction", ["安全", "药害", "混配", "间隔期", "残留", "禁用", "注意"]),
]

GENERATED_OUTPUT_FILES = [
    "all_labeled.jsonl",
    "cruciferous_all.jsonl",
    "cruciferous_pest_disease.jsonl",
    "cruciferous_agronomy.jsonl",
    "non_cruciferous.jsonl",
    "invalid.jsonl",
    "coverage_matrix.csv",
    "summary.json",
    # Obsolete outputs from the earlier stricter filtering pass.
    "core_train.jsonl",
    "weak_related.jsonl",
    "negative.jsonl",
    "review.jsonl",
    "discard.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="dataset/merged_dataset_v2/qa_merged.jsonl",
        help="Merged source QA JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/build_dataset/curated",
        help="Directory for labeled outputs and reports.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.75,
        help="Only used to add a quality flag; it does not filter cruciferous rows.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.70,
        help="Only used to add a confidence flag; it does not filter cruciferous rows.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append(
                    {
                        "_invalid_json": True,
                        "_line_no": line_no,
                        "_error": str(exc),
                        "_raw": line[:500],
                    }
                )
                continue
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_message_content(row: dict[str, Any], role: str) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == role:
            return str(message.get("content", "")).strip()
    return ""


def norm_text(*parts: Any) -> str:
    return "\n".join(str(part or "") for part in parts)


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword and keyword in text for keyword in keywords)


def normalize_targets(raw_target: str, topic_type: str) -> list[str]:
    """Split composite disease/pest labels and normalize common aliases."""
    parts = split_target_parts(raw_target)
    targets: list[str] = []
    for part in parts:
        normalized = normalize_target_part(part)
        for target in normalized:
            if target not in targets:
                targets.append(target)
    if targets:
        return targets
    return [topic_type or "unknown"]


def split_target_parts(raw_target: str) -> list[str]:
    text = normalize_target_text(raw_target)
    if not text:
        return []
    text = re.sub(r"[，,、；;／/|+]+", "、", text)
    text = re.sub(r"(?:以及|或者|或|和|及|与)", "、", text)
    text = re.sub(r"\s+", "、", text)
    return [part.strip() for part in text.split("、") if part.strip()]


def normalize_target_text(value: str) -> str:
    text = str(value or "").strip()
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("，", ",").replace("；", ";")
    text = re.sub(r"\([^)]*\)", lambda match: "、" + match.group(0)[1:-1] + "、", text)
    text = text.replace(",", "、").replace(";", "、")
    text = re.sub(r"(主要|常见|发生|防治|危害|为害|病害|虫害|等)$", "", text)
    return text.strip(" 、")


def normalize_target_part(part: str) -> list[str]:
    text = cleanup_target_part(part)
    if not text or text in GENERIC_TARGETS:
        return []

    matched: list[str] = []
    for alias, canonical in TARGET_ALIAS_PAIRS:
        if alias in text and canonical not in matched:
            matched.append(canonical)
    if matched:
        return matched

    cleaned = strip_crop_prefixes(text)
    cleaned = cleanup_target_part(cleaned)
    if not cleaned or cleaned in GENERIC_TARGETS:
        return []
    return [cleaned]


def cleanup_target_part(part: str) -> str:
    text = str(part or "").strip()
    text = re.sub(r"^[0-9一二三四五六七八九十]+[大种类个]?", "", text)
    text = re.sub(r"(病害|虫害|病虫害|发生|防治|危害|为害|症状|问题)$", "", text)
    text = text.strip(" ：:，,、；;。.")
    return text


def strip_crop_prefixes(text: str) -> str:
    cleaned = text
    changed = True
    while changed:
        changed = False
        for prefix in CROP_TARGET_PREFIXES:
            if cleaned.startswith(prefix) and len(cleaned) > len(prefix):
                cleaned = cleaned[len(prefix) :]
                changed = True
                break
    return cleaned


def normalize_crop(raw_crop: str, text: str) -> tuple[str, bool, str]:
    crop_text = raw_crop.strip()
    combined = f"{crop_text}\n{text}"
    for false_friend in NON_CRUCIFEROUS_FALSE_FRIENDS:
        if false_friend in crop_text:
            return crop_text, False, "metadata_false_friend"

    for canonical, aliases in CRUCIFEROUS_ALIASES.items():
        if any(alias == crop_text for alias in aliases):
            return canonical, True, "metadata_exact"

    for canonical, aliases in CRUCIFEROUS_ALIASES.items():
        if any(alias in crop_text for alias in aliases):
            return canonical, True, "metadata_contains"

    for false_friend in NON_CRUCIFEROUS_FALSE_FRIENDS:
        if false_friend in combined:
            return crop_text, False, "text_false_friend"

    for canonical, aliases in CRUCIFEROUS_ALIASES.items():
        if any(alias in combined for alias in aliases):
            return canonical, True, "text_contains"

    return crop_text, False, "no_match"


def infer_topic_type(primary_text: str, category: str, disease_or_pest: str) -> str:
    topic_text = f"{category}\n{disease_or_pest}\n{primary_text}"
    if "病害" in category:
        return "disease"
    if "虫害" in category:
        return "pest"
    has_disease = contains_any(topic_text, DISEASE_KEYWORDS)
    has_pest = contains_any(topic_text, PEST_KEYWORDS)
    if has_disease and has_pest:
        return "mixed_pest_disease"
    if has_disease:
        return "disease"
    if has_pest:
        return "pest"
    if "草" in disease_or_pest or "除草" in topic_text or "杂草" in topic_text:
        return "weed"
    if "农药" in category or "用药" in topic_text:
        return "pesticide_use"
    if contains_any(topic_text, WEAK_TOPIC_KEYWORDS):
        return "weak_agronomy"
    return "other"


def infer_task_types(text: str) -> list[str]:
    matched = [
        task_type
        for task_type, keywords in TASK_TYPE_RULES
        if contains_any(text, keywords)
    ]
    return matched or ["general_qa"]


def stable_row_id(row: dict[str, Any]) -> str:
    source = row.get("source") or row.get("metadata", {}).get("source_site") or "unknown"
    line_no = row.get("_line_no", "0")
    return f"qa-{line_no}-{slugify(str(source))}"


def slugify(value: str) -> str:
    value = re.sub(r"\s+", "-", value.strip())
    value = re.sub(r"[^\w\u4e00-\u9fff.-]+", "-", value)
    return value.strip("-") or "unknown"


def classify_row(
    row: dict[str, Any], min_quality: float, min_confidence: float
) -> tuple[str, dict[str, Any]]:
    if row.get("_invalid_json"):
        return "invalid", {
            "row_id": stable_row_id(row),
            "bucket": "invalid",
            "retain": False,
            "canonical_crop": "",
            "raw_crop": "",
            "is_cruciferous": False,
            "crop_match_source": "invalid_json",
            "topic_type": "invalid",
            "normalized_targets": [],
            "is_pest_disease": False,
            "is_control_related": False,
            "weak_related": False,
            "task_types": [],
            "has_images": False,
            "quality_score": 0.0,
            "confidence": 0.0,
            "quality_flags": ["invalid_json"],
            "quality_decision": "invalid",
        }

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    question = get_message_content(row, "user")
    answer = get_message_content(row, "assistant")
    crop_raw = str(metadata.get("crop", "") or "")
    category = str(metadata.get("category", "") or "")
    disease_or_pest = str(metadata.get("disease_or_pest", "") or "")
    source_url = str(metadata.get("source_url", "") or "")
    primary_text = norm_text(question, category, disease_or_pest)
    full_text = norm_text(question, answer, category, disease_or_pest)

    quality_score = to_float(metadata.get("quality_score"), 0.0)
    confidence = to_float(metadata.get("confidence"), 0.0)
    canonical_crop, is_cruciferous, crop_match_source = normalize_crop(crop_raw, primary_text)
    topic_type = infer_topic_type(primary_text, category, disease_or_pest)
    normalized_targets = normalize_targets(disease_or_pest, topic_type)
    is_pest_disease = topic_type in {"disease", "pest", "mixed_pest_disease"}
    is_control_related = contains_any(full_text, CONTROL_KEYWORDS) or "防治" in category
    weak_related = topic_type in {"weak_agronomy", "pesticide_use"} or contains_any(primary_text, WEAK_TOPIC_KEYWORDS)
    task_types = infer_task_types(full_text)
    has_images = bool(metadata.get("images"))
    safety_review = contains_any(full_text, SAFETY_REVIEW_KEYWORDS)

    quality_flags: list[str] = []
    if not question:
        quality_flags.append("missing_question")
    if not answer:
        quality_flags.append("missing_answer")
    if not source_url:
        quality_flags.append("missing_source_url")
    if quality_score < min_quality:
        quality_flags.append("low_quality_score")
    if confidence < min_confidence:
        quality_flags.append("low_confidence")
    if safety_review:
        quality_flags.append("safety_review_keyword")

    if not question or not answer:
        bucket = "invalid"
    elif is_cruciferous:
        bucket = "cruciferous_all"
    else:
        bucket = "non_cruciferous"

    curation = {
        "row_id": stable_row_id(row),
        "bucket": bucket,
        "retain": bucket == "cruciferous_all",
        "canonical_crop": canonical_crop,
        "raw_crop": crop_raw,
        "is_cruciferous": is_cruciferous,
        "crop_match_source": crop_match_source,
        "topic_type": topic_type,
        "normalized_targets": normalized_targets,
        "is_pest_disease": is_pest_disease,
        "is_control_related": is_control_related,
        "weak_related": weak_related,
        "task_types": task_types,
        "has_images": has_images,
        "quality_score": quality_score,
        "confidence": confidence,
        "quality_flags": quality_flags,
        "quality_decision": "retain" if bucket == "cruciferous_all" else bucket,
    }
    return bucket, curation


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def with_curation(row: dict[str, Any], curation: dict[str, Any]) -> dict[str, Any]:
    clean = {key: value for key, value in row.items() if key not in {"_line_no"}}
    metadata = dict(clean.get("metadata") or {})
    metadata["curation"] = curation
    clean["metadata"] = metadata
    return clean


def build_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    image_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        curation = row.get("metadata", {}).get("curation", {})
        targets = curation.get("normalized_targets") or [curation.get("topic_type") or "unknown"]
        for task_type in curation.get("task_types") or ["general_qa"]:
            for target in targets:
                key = (
                    str(curation.get("canonical_crop") or "unknown"),
                    str(target),
                    str(task_type),
                )
                counts[key] += 1
                if curation.get("has_images"):
                    image_counts[key] += 1

    coverage = []
    for (crop, target, task_type), count in sorted(counts.items()):
        coverage.append(
            {
                "crop": crop,
                "target": target,
                "task_type": task_type,
                "count": count,
                "with_images": image_counts[(crop, target, task_type)],
                "gap_level": gap_level(count),
            }
        )
    return coverage


def gap_level(count: int) -> str:
    if count == 0:
        return "empty"
    if count < 3:
        return "severe_gap"
    if count < 8:
        return "thin"
    return "covered"


def write_coverage_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["crop", "target", "task_type", "count", "with_images", "gap_level"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(buckets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    all_rows = [row for rows in buckets.values() for row in rows]
    bucket_counts = {bucket: len(rows) for bucket, rows in sorted(buckets.items())}
    crop_counts = Counter(
        row.get("metadata", {}).get("curation", {}).get("canonical_crop") or "unknown"
        for row in all_rows
        if row.get("metadata", {}).get("curation", {}).get("is_cruciferous")
    )
    topic_counts = Counter(
        row.get("metadata", {}).get("curation", {}).get("topic_type") or "unknown"
        for row in all_rows
    )
    target_counts = Counter(
        target
        for row in all_rows
        if row.get("metadata", {}).get("curation", {}).get("is_cruciferous")
        for target in row.get("metadata", {}).get("curation", {}).get("normalized_targets", [])
    )
    reason_counts = Counter(
        reason
        for row in all_rows
        for reason in row.get("metadata", {}).get("curation", {}).get("quality_flags", [])
    )
    bucket_crop_counts: dict[str, dict[str, int]] = {}
    for bucket, rows in buckets.items():
        bucket_crop_counts[bucket] = dict(
            Counter(
                row.get("metadata", {}).get("curation", {}).get("canonical_crop") or "unknown"
                for row in rows
            ).most_common(20)
        )
    return {
        "total": len(all_rows),
        "bucket_counts": bucket_counts,
        "cruciferous_crop_counts": dict(crop_counts.most_common()),
        "topic_type_counts": dict(topic_counts.most_common()),
        "normalized_target_counts": dict(target_counts.most_common(50)),
        "reason_counts": dict(reason_counts.most_common()),
        "bucket_crop_counts": bucket_crop_counts,
        "cruciferous_all_with_images": sum(
            1
            for row in buckets.get("cruciferous_all", [])
            if row.get("metadata", {}).get("curation", {}).get("has_images")
        ),
        "cruciferous_pest_disease": sum(
            1
            for row in buckets.get("cruciferous_all", [])
            if row.get("metadata", {}).get("curation", {}).get("is_pest_disease")
        ),
        "cruciferous_agronomy": sum(
            1
            for row in buckets.get("cruciferous_all", [])
            if row.get("metadata", {}).get("curation", {}).get("topic_type")
            in {"weak_agronomy", "pesticide_use", "weed", "other"}
        ),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    rows = read_jsonl(input_path)

    buckets: dict[str, list[dict[str, Any]]] = {
        "cruciferous_all": [],
        "non_cruciferous": [],
        "invalid": [],
    }
    all_labeled: list[dict[str, Any]] = []
    for row in rows:
        bucket, curation = classify_row(row, args.min_quality, args.min_confidence)
        labeled = with_curation(row, curation)
        buckets.setdefault(bucket, []).append(labeled)
        all_labeled.append(labeled)

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in GENERATED_OUTPUT_FILES:
        path = output_dir / filename
        if path.exists():
            path.unlink()
    for bucket, bucket_rows in buckets.items():
        write_jsonl(output_dir / f"{bucket}.jsonl", bucket_rows)
    write_jsonl(output_dir / "all_labeled.jsonl", all_labeled)
    cruciferous_pest_disease = [
        row
        for row in buckets["cruciferous_all"]
        if row.get("metadata", {}).get("curation", {}).get("is_pest_disease")
    ]
    cruciferous_agronomy = [
        row
        for row in buckets["cruciferous_all"]
        if row.get("metadata", {}).get("curation", {}).get("topic_type")
        in {"weak_agronomy", "pesticide_use", "weed", "other"}
    ]
    write_jsonl(output_dir / "cruciferous_pest_disease.jsonl", cruciferous_pest_disease)
    write_jsonl(output_dir / "cruciferous_agronomy.jsonl", cruciferous_agronomy)

    coverage = build_coverage(buckets["cruciferous_all"])
    write_coverage_csv(output_dir / "coverage_matrix.csv", coverage)

    summary = summarize(buckets)
    summary["input"] = str(input_path)
    summary["output_dir"] = str(output_dir)
    summary["min_quality"] = args.min_quality
    summary["min_confidence"] = args.min_confidence
    summary["coverage_rows"] = len(coverage)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
