"""
Knowledge graph quality governance utilities.

This module provides:
1) Pre-merge normalization for extracted entities/relations.
2) Post-merge cleanup for stored GraphML files.
3) A small CLI entrypoint for one-shot cleanup.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set, Tuple

# 只要改这个
NODE_DEFAULT_WHITELIST = [
    "病害",
    "虫害",
    "病原菌",
    "作物",
    "生长期",
    "生物分类",
    "药剂",
    "别名",
    "危害症状",
    "形态特征",
    "发病诱因",
    "发生时期",
    "防治要点",
    "生活习性",
    "发生规律",
    "多模态元素",
    "其他",
]

ONTOLOGY_ENTITY_TYPES_CRUCIFEROUS = set(NODE_DEFAULT_WHITELIST)

# 只要改这个
RELATION_SCHEMA_FIXED_ORDERED = [
    "致病",
    "发生于",
    "属于",
    "影响",
    "防治",
    "生命周期",
    "位于",
]

RELATION_SCHEMA_FIXED = set(RELATION_SCHEMA_FIXED_ORDERED)

MULTIMODAL_TYPE_MAP = {
    "image": "多模态元素",
    "table": "多模态元素",
    "equation": "多模态元素",
    "header": "多模态元素",
    "page_number": "多模态元素",
}

RELATION_DOMAIN_RANGE_CRUCIFEROUS: Dict[str, set[Tuple[str, str]]] = {
    "致病": {("病原菌", "病害"), ("虫害", "病害")},
    "发生于": {
        ("病害", "时间"),
        ("虫害", "时间"),
        ("病害", "生长期"),
        ("虫害", "生长期"),
    },
    "影响": {("病害", "作物"), ("虫害", "作物")},
    "防治": {("病害", "药剂"), ("虫害", "药剂"), ("药剂", "病害"), ("药剂", "虫害")},
}

DEFAULT_CORE_ENTITY_TYPES = ["虫害", "病害", "作物", "病原菌", "药剂", "生长期", "生物分类"]
DEFAULT_ANCHOR_NODE_TYPES = ["部位", "时间"]
DEFAULT_ATTRIBUTE_FIELDS = ["形态特征", "危害症状", "发病诱因", "发生时期", "防治要点", "生活习性", "发生规律"]
DEFAULT_NOISE_DROP_TYPES = ["header", "page_number"]
DEFAULT_NOISE_DROP_PATTERNS = [
    r"^\s*None\s*$",
    r"^\s*\d+\s*$",
    r"^\s*第?\s*\d+\s*页\s*$",
    r"蔬菜病虫害诊断[与于]防治原色图谱",
    r"(?:QR|qr|二维码|QR码|扫码)",
    r"^(?:[A-Za-z]:)?[/\\].+\.(?:jpg|jpeg|png|bmp|gif|webp|tiff?)\s*$",
]


@dataclass
class OntologyRegistry:
    """Ontology registry (Ro) for entity/attribute/relation governance."""

    core_entity_types: Set[str]
    anchor_node_types: Set[str]
    attribute_fields: Set[str]
    relation_whitelist: Set[str]
    relation_domain_range: Dict[str, Set[Tuple[str, str]]]

    @property
    def node_whitelist(self) -> Set[str]:
        return (
            set(self.core_entity_types)
            | set(self.anchor_node_types)
            | set(self.attribute_fields)
            | {"多模态元素", "其他"}
        )


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _contains_ascii_word(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}", text or ""))


def _normalize_space(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def _clean_suffix(text: str) -> str:
    # Remove common modality suffixes from entity names, e.g. "(image)".
    text = re.sub(
        r"\s*\((?:image|table|equation|header|page_number|other)\)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _infer_content_modality(entity_name: str, entity_type: str) -> str:
    name = (entity_name or "").strip().lower()
    et = (entity_type or "").strip().lower()

    if et in {"image", "table", "equation", "header", "page_number"}:
        return et

    suffix_match = re.search(
        r"\((image|table|equation|header|page_number)\)\s*$", name
    )
    if suffix_match:
        return suffix_match.group(1)

    if "page " in name and "number" in name:
        return "page_number"
    return ""


def _split_sep_values(value: str) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split("<SEP>") if x.strip()]


def _join_sep_values(values: Iterable[str]) -> str:
    deduped = []
    seen = set()
    for value in values:
        v = value.strip()
        if not v or v in seen:
            continue
        seen.add(v)
        deduped.append(v)
    return "<SEP>".join(deduped)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


@dataclass
class KGQualityManager:
    enabled: bool = True
    canonical_language: str = "zh"
    relation_schema: str = "fixed"
    extraction_mode: str = "ppe"
    ontology_profile: str = "cruciferous_pest_disease"
    enforce_ontology: bool = True
    merge_threshold: float = 0.86
    description_policy: str = "multimodal_only"
    core_entity_types: List[str] = field(
        default_factory=lambda: DEFAULT_CORE_ENTITY_TYPES.copy()
    )
    anchor_node_types: List[str] = field(
        default_factory=lambda: DEFAULT_ANCHOR_NODE_TYPES.copy()
    )
    attribute_fields: List[str] = field(
        default_factory=lambda: DEFAULT_ATTRIBUTE_FIELDS.copy()
    )
    noise_drop_types: List[str] = field(
        default_factory=lambda: DEFAULT_NOISE_DROP_TYPES.copy()
    )
    noise_drop_patterns: List[str] = field(
        default_factory=lambda: DEFAULT_NOISE_DROP_PATTERNS.copy()
    )
    allowed_entity_types: List[str] = field(
        default_factory=lambda: NODE_DEFAULT_WHITELIST.copy()
    )
    ontology_registry: OntologyRegistry | None = field(default=None, init=False, repr=False)
    _noise_type_set: Set[str] = field(default_factory=set, init=False, repr=False)
    _noise_regexes: List[re.Pattern[str]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        relation_whitelist = set(RELATION_SCHEMA_FIXED)
        relation_domain_range = {
            k: set(v) for k, v in RELATION_DOMAIN_RANGE_CRUCIFEROUS.items()
        }
        self.ontology_registry = OntologyRegistry(
            core_entity_types=set(self.core_entity_types),
            anchor_node_types=set(self.anchor_node_types),
            attribute_fields=set(self.attribute_fields),
            relation_whitelist=relation_whitelist,
            relation_domain_range=relation_domain_range,
        )
        self._noise_type_set = {
            _normalize_space(t).lower() for t in self.noise_drop_types if _normalize_space(t)
        }
        self._noise_regexes = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.noise_drop_patterns if pattern
        ]

        if self.ontology_profile in {
            "cruciferous_pest_disease",
            "crop_pest_disease",
            "rice_disease_pest",
        }:
            if self.enforce_ontology:
                self.allowed_entity_types = sorted(
                    self.ontology_registry.node_whitelist | ONTOLOGY_ENTITY_TYPES_CRUCIFEROUS
                )
            else:
                self.allowed_entity_types = sorted(
                    set(self.allowed_entity_types)
                    | self.ontology_registry.node_whitelist
                    | ONTOLOGY_ENTITY_TYPES_CRUCIFEROUS
                )
        else:
            self.allowed_entity_types = sorted(
                set(self.allowed_entity_types) | self.ontology_registry.node_whitelist
            )

    def get_lightrag_entity_types(self) -> List[str]:
        """Entity whitelist exposed to LightRAG extraction prompt."""
        excluded = {"其他", "多模态元素"}
        preferred_order = NODE_DEFAULT_WHITELIST
        picked: List[str] = []
        seen: Set[str] = set()
        for item in preferred_order:
            if item in self.allowed_entity_types and item not in excluded and item not in seen:
                picked.append(item)
                seen.add(item)
        for item in self.allowed_entity_types:
            if item not in excluded and item not in seen:
                picked.append(item)
                seen.add(item)
        return picked

    def get_lightrag_relation_types(self) -> List[str]:
        """Relation whitelist exposed to LightRAG extraction prompt."""
        whitelist = (
            set(self.ontology_registry.relation_whitelist)
            if self.ontology_registry
            else set(RELATION_SCHEMA_FIXED)
        )
        picked: List[str] = []
        seen: Set[str] = set()
        for rel in RELATION_SCHEMA_FIXED_ORDERED:
            if rel in whitelist and rel not in seen:
                picked.append(rel)
                seen.add(rel)
        for rel in sorted(whitelist):
            if rel not in seen:
                picked.append(rel)
                seen.add(rel)
        return picked

    def _normalize_entity_type(self, entity_type: str) -> str:
        raw = _normalize_space(entity_type)
        if not raw:
            return "其他"
        lowered = raw.lower()
        mapped = MULTIMODAL_TYPE_MAP.get(lowered, raw)
        if mapped not in self.allowed_entity_types:
            return "其他"
        return mapped

    def _normalize_entity_name(self, entity_name: str) -> Tuple[str, List[str]]:
        raw = _normalize_space(_clean_suffix(entity_name))
        if not raw:
            return "未命名实体", []

        return raw, [raw]

    def _enforce_type_name_consistency(
        self, entity_name: str, entity_type: str
    ) -> Tuple[str, str]:
        normalized_type = self._normalize_entity_type(entity_type)
        normalized_name, _aliases = self._normalize_entity_name(entity_name)

        # If canonical language is zh and name is still English-only, downgrade to "其他"
        # unless the type is already "其他".
        if (
            self.canonical_language.lower() == "zh"
            and not _contains_cjk(normalized_name)
            and _contains_ascii_word(normalized_name)
            and normalized_type not in {"其他", "多模态元素"}
            and normalized_type
            not in (
                self.ontology_registry.attribute_fields
                if self.ontology_registry
                else set()
            )
        ):
            normalized_type = "其他"

        return normalized_name, normalized_type

    def normalize_entity(self, entity_name: str, entity_type: str) -> Dict[str, Any]:
        content_modality = _infer_content_modality(entity_name, entity_type)
        canonical_name, canonical_type = self._enforce_type_name_consistency(
            entity_name, entity_type
        )
        _, aliases = self._normalize_entity_name(entity_name)
        aliases = [a for a in aliases if a and a != canonical_name]
        return {
            "entity_name": canonical_name,
            "entity_type": canonical_type,
            "aliases": aliases,
            "content_modality": content_modality,
        }

    def _is_core_type(self, entity_type: str) -> bool:
        if not self.ontology_registry:
            return False
        return entity_type in self.ontology_registry.core_entity_types

    def _is_anchor_type(self, entity_type: str) -> bool:
        if not self.ontology_registry:
            return False
        return entity_type in self.ontology_registry.anchor_node_types

    def _is_attribute_type(self, entity_type: str) -> bool:
        if not self.ontology_registry:
            return False
        return entity_type in self.ontology_registry.attribute_fields

    def _keep_as_node_type(self, entity_type: str) -> bool:
        return (
            self._is_core_type(entity_type)
            or self._is_anchor_type(entity_type)
            or entity_type == "多模态元素"
        )

    def _apply_node_description_policy(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(node_data)
        if self.description_policy == "multimodal_only":
            if normalized.get("entity_type") != "多模态元素":
                normalized["description"] = ""
        return normalized

    def _apply_edge_description_policy(self, edge_data: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(edge_data)
        if self.description_policy == "multimodal_only":
            normalized["description"] = ""
        return normalized

    def _relation_to_attribute_field(
        self, relation_type: str, raw_keywords: str = "", description: str = ""
    ) -> str:
        text = f"{relation_type} {raw_keywords} {description}".lower()
        if any(k in text for k in ["形态", "morphology"]):
            return "形态特征"
        if any(k in text for k in ["症状", "symptom", "damage", "危害"]):
            return "危害症状"
        if any(k in text for k in ["诱因", "cause", "trigger"]):
            return "发病诱因"
        if any(k in text for k in ["时期", "生长期", "时间", "life cycle", "lifecycle"]):
            return "发生时期"
        if any(k in text for k in ["防治", "药剂", "control", "prevention", "treatment"]):
            return "防治要点"
        if any(k in text for k in ["习性", "habit"]):
            return "生活习性"
        if any(k in text for k in ["发生规律", "规律", "occurrence"]):
            return "发生规律"
        return ""

    def _extract_attribute_value(
        self, node_name: str, node_data: Dict[str, Any], edge_data: Dict[str, Any] | None = None
    ) -> str:
        node_desc = _normalize_space(str(node_data.get("description", "")))
        edge_desc = _normalize_space(str((edge_data or {}).get("description", "")))
        if len(node_desc) >= len(node_name) and node_desc:
            return node_desc
        if len(edge_desc) >= len(node_name) and edge_desc:
            return edge_desc
        return _normalize_space(node_name)

    def _append_attribute_to_node(
        self,
        node_data: Dict[str, Any],
        field_name: str,
        value: str,
        evidence: Dict[str, str],
    ) -> None:
        if not field_name or not value:
            return

        attrs: Dict[str, List[str]]
        existing_attrs = node_data.get("attributes", "{}")
        if isinstance(existing_attrs, dict):
            attrs = {k: list(v) for k, v in existing_attrs.items()}
        else:
            try:
                attrs = json.loads(existing_attrs) if existing_attrs else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                attrs = {}
        attr_values = attrs.get(field_name, [])
        if value not in attr_values:
            attr_values.append(value)
        attrs[field_name] = attr_values
        node_data["attributes"] = json.dumps(attrs, ensure_ascii=False)

        existing_evidence = node_data.get("attribute_evidence", "{}")
        if isinstance(existing_evidence, dict):
            evidence_map = {k: list(v) for k, v in existing_evidence.items()}
        else:
            try:
                evidence_map = json.loads(existing_evidence) if existing_evidence else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                evidence_map = {}
        evidence_list = evidence_map.get(field_name, [])
        if evidence not in evidence_list:
            evidence_list.append(evidence)
        evidence_map[field_name] = evidence_list
        node_data["attribute_evidence"] = json.dumps(evidence_map, ensure_ascii=False)

    def filter_parsed_content(
        self, content_list: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.enabled:
            return content_list, {"enabled": False, "dropped": 0, "kept": len(content_list)}

        filtered: List[Dict[str, Any]] = []
        by_type: Counter[str] = Counter()
        by_pattern: Counter[str] = Counter()

        def _candidate_texts(item: Dict[str, Any]) -> List[str]:
            texts: List[str] = []
            raw_text = item.get("text")
            if isinstance(raw_text, str):
                texts.append(raw_text)
            for key in ["image_caption", "img_caption", "table_caption", "image_footnote", "img_footnote", "table_footnote"]:
                value = item.get(key)
                if isinstance(value, list):
                    texts.extend([str(v) for v in value if v is not None])
                elif isinstance(value, str):
                    texts.append(value)
            if item.get("type") == "text":
                maybe_path = item.get("img_path")
                if isinstance(maybe_path, str):
                    texts.append(maybe_path)
            image_path = str(item.get("img_path", ""))
            if image_path and re.search(r"(qr|qrcode|二维码)", image_path, re.IGNORECASE):
                texts.append("QR码")
            return [t for t in texts if isinstance(t, str)]

        for item in content_list:
            item_type = _normalize_space(str(item.get("type", ""))).lower()
            if item_type in self._noise_type_set:
                by_type[item_type] += 1
                continue

            dropped = False
            for text in _candidate_texts(item):
                normalized_text = _normalize_space(text)
                if not normalized_text:
                    continue
                for pattern in self._noise_regexes:
                    if pattern.search(normalized_text):
                        by_pattern[pattern.pattern] += 1
                        dropped = True
                        break
                if dropped:
                    break

            if dropped:
                continue
            filtered.append(item)

        report = {
            "enabled": True,
            "total": len(content_list),
            "kept": len(filtered),
            "dropped": len(content_list) - len(filtered),
            "drop_by_type": dict(by_type),
            "drop_by_pattern": dict(by_pattern),
        }
        return filtered, report

    def map_relation_type(
        self,
        keywords: str,
        description: str = "",
        src_type: str = "",
        tgt_type: str = "",
    ) -> str:
        if self.relation_schema != "fixed":
            return ""

        text = f"{keywords or ''} {description or ''}".lower()

        if any(k in text for k in ["belongs_to", "part_of", "contained_in", "属于"]):
            relation = "属于"
        elif any(k in text for k in ["pathogen", "致病菌", "致病"]):
            relation = "致病"
        elif any(
            k in text
            for k in [
                "growth stage",
                "生长期",
                "发生时期",
                "period",
                "history",
                "record time",
                "时间",
            ]
        ):
            relation = "发生于"
        elif any(k in text for k in ["control", "prevention", "management", "防治"]):
            relation = "防治"
        elif any(k in text for k in ["impact", "affect", "cause", "影响"]):
            relation = "影响"
        elif any(
            k in text
            for k in ["life cycle", "lifecycle", "stage", "overwinter", "生命周期"]
        ):
            relation = "生命周期"
        elif any(k in text for k in ["location", "located", "位于", "地理位置", "page "]):
            relation = "位于"
        else:
            relation = ""

        if relation and self.enforce_ontology and not self._relation_allowed(
            relation, src_type, tgt_type
        ):
            return ""
        return relation

    def _relation_allowed(self, relation: str, src_type: str, tgt_type: str) -> bool:
        if relation in {"属于"}:
            return True
        if not relation:
            return False
        if not src_type or not tgt_type:
            return True
        if self.ontology_registry and relation not in self.ontology_registry.relation_whitelist:
            return False
        if self.ontology_profile not in {
            "cruciferous_pest_disease",
            "crop_pest_disease",
            "rice_disease_pest",
        }:
            return True
        allowed_pairs = RELATION_DOMAIN_RANGE_CRUCIFEROUS.get(relation)
        if not allowed_pairs:
            return True
        return (src_type, tgt_type) in allowed_pairs

    def normalize_edge(
        self, edge_data: Dict[str, Any], src_type: str = "", tgt_type: str = ""
    ) -> Dict[str, Any]:
        raw_keywords = _normalize_space(str(edge_data.get("keywords", "")))
        description = str(edge_data.get("description", ""))
        relation_type = self.map_relation_type(
            raw_keywords, description, src_type=src_type, tgt_type=tgt_type
        )
        normalized = dict(edge_data)
        normalized["raw_keywords"] = raw_keywords
        normalized["relation_type"] = relation_type
        # Keep compatibility with current retrieval that still reads "keywords".
        normalized["keywords"] = relation_type
        # LightRAG merge requires relation descriptions to exist; guard against
        # model outputs that violate prompt constraints.
        if not _normalize_space(str(normalized.get("description", ""))):
            normalized["description"] = "N/A"
        # NOTE:
        # LightRAG merge requires relation descriptions to be present.
        # Do not clear relation descriptions in preprocess stage; defer cleanup
        # to GraphML rewrite stage where final policy is enforced.
        return normalized

    def _merge_ieu_nodes(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Incremental entity update with "main attribute first" policy."""
        if not records:
            return {}

        merged = dict(records[0])
        for rec in records[1:]:
            if merged.get("entity_type") == "其他" and rec.get("entity_type") != "其他":
                merged["entity_type"] = rec.get("entity_type")

            if not merged.get("content_modality") and rec.get("content_modality"):
                merged["content_modality"] = rec.get("content_modality")

            for field in ["description", "truncate"]:
                if not merged.get(field) and rec.get(field):
                    merged[field] = rec.get(field)

            for sep_field in ["aliases", "source_id", "file_path"]:
                merged[sep_field] = _join_sep_values(
                    _split_sep_values(str(merged.get(sep_field, "")))
                    + _split_sep_values(str(rec.get(sep_field, "")))
                )

            merged["created_at"] = max(
                _safe_int(merged.get("created_at"), 0),
                _safe_int(rec.get("created_at"), 0),
            )

            for field in ["attributes", "attribute_evidence"]:
                base_raw = merged.get(field, "{}")
                add_raw = rec.get(field, "{}")
                try:
                    base = json.loads(base_raw) if base_raw else {}
                except (TypeError, ValueError, json.JSONDecodeError):
                    base = {}
                try:
                    add = json.loads(add_raw) if add_raw else {}
                except (TypeError, ValueError, json.JSONDecodeError):
                    add = {}
                for k, v in add.items():
                    current = list(base.get(k, []))
                    for item in v:
                        if item not in current:
                            current.append(item)
                    base[k] = current
                merged[field] = json.dumps(base, ensure_ascii=False)

        # Keep original description during preprocess/merge.
        # Final description policy is enforced when rewriting GraphML.
        return merged

    def _append_alias_to_node_records(
        self, nodes: Dict[str, List[Dict[str, Any]]], canonical: str, alias: str
    ) -> None:
        if not canonical or not alias or canonical == alias:
            return
        if canonical not in nodes:
            return
        for rec in nodes[canonical]:
            aliases = _split_sep_values(str(rec.get("aliases", "")))
            aliases.append(alias)
            rec["aliases"] = _join_sep_values(aliases)

    def _preprocess_chunk_results_legacy(self, chunk_results: List[Tuple]) -> List[Tuple]:
        normalized_results = []

        for maybe_nodes, maybe_edges in chunk_results:
            alias_map: Dict[str, str] = {}
            new_nodes: Dict[str, List[Dict[str, Any]]] = {}
            node_type_map: Dict[str, str] = {}

            for raw_name, node_list in maybe_nodes.items():
                first_type = ""
                if node_list and isinstance(node_list[0], dict):
                    first_type = str(node_list[0].get("entity_type", ""))

                norm = self.normalize_entity(raw_name, first_type)
                canonical = norm["entity_name"]
                alias_map[raw_name] = canonical

                if canonical not in new_nodes:
                    new_nodes[canonical] = []

                for node_data in node_list:
                    nd = dict(node_data)
                    nd["entity_id"] = canonical
                    nd["entity_type"] = self._normalize_entity_type(
                        nd.get("entity_type", first_type)
                    )
                    node_type_map[canonical] = nd["entity_type"]
                    modality = (
                        norm.get("content_modality")
                        or nd.get("content_modality")
                        or _infer_content_modality(raw_name, first_type)
                    )
                    if modality:
                        nd["content_modality"] = modality
                    aliases = _split_sep_values(str(nd.get("aliases", "")))
                    aliases.extend(norm["aliases"])
                    if raw_name != canonical:
                        aliases.append(raw_name)
                    nd["aliases"] = _join_sep_values(aliases)
                    new_nodes[canonical].append(self._apply_node_description_policy(nd))

            new_edges: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
            for (src, tgt), edge_list in maybe_edges.items():
                norm_src = alias_map.get(src, src)
                norm_tgt = alias_map.get(tgt, tgt)
                key = (norm_src, norm_tgt)
                if key not in new_edges:
                    new_edges[key] = []

                for edge_data in edge_list:
                    ed = dict(edge_data)
                    ed["src_id"] = norm_src
                    ed["tgt_id"] = norm_tgt
                    normalized_edge = self.normalize_edge(
                        ed,
                        src_type=node_type_map.get(norm_src, ""),
                        tgt_type=node_type_map.get(norm_tgt, ""),
                    )

                    new_edges[key].append(normalized_edge)

            new_edges = {k: v for k, v in new_edges.items() if v}

            normalized_results.append((new_nodes, new_edges))

        return normalized_results

    def _preprocess_chunk_results_ppe(self, chunk_results: List[Tuple]) -> List[Tuple]:
        normalized_results: List[Tuple] = []

        for maybe_nodes, maybe_edges in chunk_results:
            alias_map: Dict[str, str] = {}
            normalized_nodes_by_name: Dict[str, List[Dict[str, Any]]] = {}
            node_type_map: Dict[str, str] = {}

            # PPE Round 1: core/anchor/multimodal entity normalization.
            for raw_name, node_list in maybe_nodes.items():
                first_type = ""
                if node_list and isinstance(node_list[0], dict):
                    first_type = str(node_list[0].get("entity_type", ""))
                norm = self.normalize_entity(raw_name, first_type)
                canonical = norm["entity_name"]
                alias_map[raw_name] = canonical

                if canonical not in normalized_nodes_by_name:
                    normalized_nodes_by_name[canonical] = []

                for node_data in node_list:
                    nd = dict(node_data)
                    normalized_type = self._normalize_entity_type(
                        str(nd.get("entity_type", first_type))
                    )
                    nd["entity_id"] = canonical
                    nd["entity_type"] = normalized_type
                    node_type_map[canonical] = normalized_type
                    modality = (
                        norm.get("content_modality")
                        or nd.get("content_modality")
                        or _infer_content_modality(raw_name, first_type)
                    )
                    if modality:
                        nd["content_modality"] = modality
                    aliases = _split_sep_values(str(nd.get("aliases", "")))
                    aliases.extend(norm["aliases"])
                    if raw_name != canonical:
                        aliases.append(raw_name)
                    nd["aliases"] = _join_sep_values(aliases)
                    nd.setdefault("attributes", "{}")
                    nd.setdefault("attribute_evidence", "{}")
                    normalized_nodes_by_name[canonical].append(nd)

            normalized_edges: List[Tuple[str, str, Dict[str, Any]]] = []
            for (src, tgt), edge_list in maybe_edges.items():
                norm_src = alias_map.get(src, src)
                norm_tgt = alias_map.get(tgt, tgt)
                src_type = node_type_map.get(norm_src, "")
                tgt_type = node_type_map.get(norm_tgt, "")
                for edge_data in edge_list:
                    ed = dict(edge_data)
                    ed["src_id"] = norm_src
                    ed["tgt_id"] = norm_tgt
                    normalized_edge = self.normalize_edge(
                        ed, src_type=src_type, tgt_type=tgt_type
                    )

                    normalized_edges.append((norm_src, norm_tgt, normalized_edge))

            # Build final node set with core + anchor + multimodal nodes only.
            kept_nodes: Dict[str, List[Dict[str, Any]]] = {}
            dropped_attribute_nodes: Dict[str, List[Dict[str, Any]]] = {}
            for canonical, records in normalized_nodes_by_name.items():
                typ = node_type_map.get(canonical, "其他")
                if self._keep_as_node_type(typ):
                    kept_nodes[canonical] = records
                elif self._is_attribute_type(typ):
                    dropped_attribute_nodes[canonical] = records

            # PPE Round 2: attach attributes from dropped nodes to connected core nodes.
            for src, tgt, edge_data in normalized_edges:
                src_type = node_type_map.get(src, "")
                tgt_type = node_type_map.get(tgt, "")
                if src in dropped_attribute_nodes and tgt in kept_nodes:
                    for attr_node in dropped_attribute_nodes[src]:
                        field = src_type if self._is_attribute_type(src_type) else self._relation_to_attribute_field(
                            edge_data.get("relation_type", ""),
                            edge_data.get("raw_keywords", ""),
                            edge_data.get("description", ""),
                        )
                        value = self._extract_attribute_value(src, attr_node, edge_data)
                        evidence = {
                            "chunk": str(edge_data.get("source_id", "")),
                            "file": str(edge_data.get("file_path", "")),
                            "page": str(attr_node.get("page_idx", "")),
                        }
                        for rec in kept_nodes[tgt]:
                            self._append_attribute_to_node(rec, field, value, evidence)
                if tgt in dropped_attribute_nodes and src in kept_nodes:
                    for attr_node in dropped_attribute_nodes[tgt]:
                        field = tgt_type if self._is_attribute_type(tgt_type) else self._relation_to_attribute_field(
                            edge_data.get("relation_type", ""),
                            edge_data.get("raw_keywords", ""),
                            edge_data.get("description", ""),
                        )
                        value = self._extract_attribute_value(tgt, attr_node, edge_data)
                        evidence = {
                            "chunk": str(edge_data.get("source_id", "")),
                            "file": str(edge_data.get("file_path", "")),
                            "page": str(attr_node.get("page_idx", "")),
                        }
                        for rec in kept_nodes[src]:
                            self._append_attribute_to_node(rec, field, value, evidence)

            # PPE Round 3: relation extraction under ontology whitelist.
            final_edges: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
            for src, tgt, edge_data in normalized_edges:
                if src not in kept_nodes or tgt not in kept_nodes:
                    continue
                rel = edge_data.get("relation_type", "")
                if not rel:
                    continue
                if self.ontology_registry and rel not in self.ontology_registry.relation_whitelist:
                    continue
                key = (src, tgt)
                final_edges.setdefault(key, []).append(edge_data)

            merged_nodes: Dict[str, List[Dict[str, Any]]] = {}
            for canonical, records in kept_nodes.items():
                merged_nodes[canonical] = [self._merge_ieu_nodes(records)]

            # Supplemental IEU: infer attributes from edge descriptions on kept nodes.
            for (src, _tgt), edge_list in final_edges.items():
                for edge_data in edge_list:
                    field = self._relation_to_attribute_field(
                        edge_data.get("relation_type", ""),
                        edge_data.get("raw_keywords", ""),
                        edge_data.get("description", ""),
                    )
                    if not field:
                        continue
                    if src not in merged_nodes:
                        continue
                    for rec in merged_nodes[src]:
                        self._append_attribute_to_node(
                            rec,
                            field,
                            _normalize_space(str(edge_data.get("description", ""))),
                            {
                                "chunk": str(edge_data.get("source_id", "")),
                                "file": str(edge_data.get("file_path", "")),
                                "page": "",
                            },
                        )

            merged_nodes = {
                k: [self._apply_node_description_policy(v[0])]
                for k, v in merged_nodes.items()
            }
            final_edges = {
                k: [self._apply_edge_description_policy(ed) for ed in v]
                for k, v in final_edges.items()
            }
            normalized_results.append((merged_nodes, final_edges))

        return normalized_results

    def preprocess_chunk_results(self, chunk_results: List[Tuple]) -> List[Tuple]:
        if not self.enabled:
            return chunk_results
        if self.extraction_mode.lower() == "legacy":
            return self._preprocess_chunk_results_legacy(chunk_results)
        return self._preprocess_chunk_results_ppe(chunk_results)

    def clean_graphml_file(self, graphml_path: str, rewrite: bool = True) -> Dict[str, Any]:
        if not os.path.exists(graphml_path):
            raise FileNotFoundError(f"GraphML not found: {graphml_path}")

        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
        ET.register_namespace("", ns["g"])
        tree = ET.parse(graphml_path)
        root = tree.getroot()
        graph = root.find("g:graph", ns)
        if graph is None:
            raise ValueError("Invalid GraphML: missing <graph> element")

        key_nodes: Dict[str, str] = {}
        key_edges: Dict[str, str] = {}
        all_key_ids = set()
        for key_elem in root.findall("g:key", ns):
            key_id = key_elem.attrib.get("id", "")
            all_key_ids.add(key_id)
            attr_name = key_elem.attrib.get("attr.name", "")
            key_for = key_elem.attrib.get("for", "")
            if key_for == "node":
                key_nodes[attr_name] = key_id
            elif key_for == "edge":
                key_edges[attr_name] = key_id

        def ensure_key(key_for: str, attr_name: str, attr_type: str = "string") -> str:
            mapping = key_nodes if key_for == "node" else key_edges
            if attr_name in mapping:
                return mapping[attr_name]
            idx = 0
            while f"d{idx}" in all_key_ids:
                idx += 1
            key_id = f"d{idx}"
            all_key_ids.add(key_id)
            elem = ET.Element(
                "key",
                {
                    "id": key_id,
                    "for": key_for,
                    "attr.name": attr_name,
                    "attr.type": attr_type,
                },
            )
            root.insert(0, elem)
            mapping[attr_name] = key_id
            return key_id

        node_aliases_key = ensure_key("node", "aliases", "string")
        node_content_modality_key = ensure_key("node", "content_modality", "string")
        node_attributes_key = ensure_key("node", "attributes", "string")
        node_attribute_evidence_key = ensure_key("node", "attribute_evidence", "string")
        edge_relation_type_key = ensure_key("edge", "relation_type", "string")
        edge_raw_keywords_key = ensure_key("edge", "raw_keywords", "string")

        # Parse and normalize nodes.
        nodes = graph.findall("g:node", ns)
        old_to_new: Dict[str, str] = {}
        merged_nodes: Dict[str, Dict[str, Any]] = {}

        for node in nodes:
            node_id = node.attrib.get("id", "")
            node_data: Dict[str, str] = {}
            for data in node.findall("g:data", ns):
                k = data.attrib.get("key", "")
                attr_name = next((n for n, kid in key_nodes.items() if kid == k), k)
                node_data[attr_name] = data.text or ""

            entity_name = node_data.get("entity_id") or node_id
            entity_type = node_data.get("entity_type", "")
            norm = self.normalize_entity(entity_name, entity_type)
            canonical = norm["entity_name"]
            old_to_new[node_id] = canonical
            old_to_new[entity_name] = canonical

            if canonical not in merged_nodes:
                merged_nodes[canonical] = {
                    "entity_id": canonical,
                    "entity_type": norm["entity_type"],
                    "content_modality": norm.get("content_modality", ""),
                    "description": [],
                    "source_id": [],
                    "file_path": [],
                    "created_at": 0,
                    "truncate": "",
                    "aliases": set(norm["aliases"]),
                    "attributes": {},
                    "attribute_evidence": {},
                }

            slot = merged_nodes[canonical]
            if slot["entity_type"] == "其他" and norm["entity_type"] != "其他":
                slot["entity_type"] = norm["entity_type"]
            if not slot.get("content_modality"):
                slot["content_modality"] = (
                    node_data.get("content_modality", "")
                    or norm.get("content_modality", "")
                    or _infer_content_modality(entity_name, entity_type)
                )

            slot["description"].extend(_split_sep_values(node_data.get("description", "")))
            slot["source_id"].extend(_split_sep_values(node_data.get("source_id", "")))
            slot["file_path"].extend(_split_sep_values(node_data.get("file_path", "")))
            slot["created_at"] = max(
                slot["created_at"], _safe_int(node_data.get("created_at"), 0)
            )
            if node_data.get("truncate"):
                slot["truncate"] = node_data.get("truncate", "")
            slot["aliases"].add(entity_name)
            slot["aliases"].add(node_id)
            try:
                attributes = json.loads(node_data.get("attributes", "{}") or "{}")
                for k, v in attributes.items():
                    current = list(slot["attributes"].get(k, []))
                    for item in v:
                        if item not in current:
                            current.append(item)
                    slot["attributes"][k] = current
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            try:
                evidences = json.loads(
                    node_data.get("attribute_evidence", "{}") or "{}"
                )
                for k, v in evidences.items():
                    current = list(slot["attribute_evidence"].get(k, []))
                    for item in v:
                        if item not in current:
                            current.append(item)
                    slot["attribute_evidence"][k] = current
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

        if self.extraction_mode.lower() == "ppe":
            merged_nodes = {
                k: v
                for k, v in merged_nodes.items()
                if self._keep_as_node_type(v.get("entity_type", "其他"))
            }

        # Parse and normalize edges.
        edges = graph.findall("g:edge", ns)
        merged_edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        merged_node_type_map = {
            name: data["entity_type"] for name, data in merged_nodes.items()
        }

        for edge in edges:
            src = old_to_new.get(edge.attrib.get("source", ""), edge.attrib.get("source", ""))
            tgt = old_to_new.get(edge.attrib.get("target", ""), edge.attrib.get("target", ""))
            edge_data: Dict[str, str] = {}
            for data in edge.findall("g:data", ns):
                k = data.attrib.get("key", "")
                attr_name = next((n for n, kid in key_edges.items() if kid == k), k)
                edge_data[attr_name] = data.text or ""

            normalized_edge = self.normalize_edge(
                edge_data,
                src_type=merged_node_type_map.get(src, ""),
                tgt_type=merged_node_type_map.get(tgt, ""),
            )
            relation_type = normalized_edge.get("relation_type", "")
            if not relation_type:
                continue
            if self.ontology_registry and relation_type not in self.ontology_registry.relation_whitelist:
                continue
            if src not in merged_nodes or tgt not in merged_nodes:
                continue
            key = (src, tgt, relation_type)

            if key not in merged_edges:
                merged_edges[key] = {
                    "description": [],
                    "keywords": relation_type,
                    "raw_keywords": [],
                    "relation_type": relation_type,
                    "source_id": [],
                    "file_path": [],
                    "created_at": 0,
                    "truncate": "",
                    "weight": 0.0,
                }

            slot = merged_edges[key]
            slot["description"].extend(_split_sep_values(normalized_edge.get("description", "")))
            if normalized_edge.get("raw_keywords"):
                slot["raw_keywords"].append(normalized_edge["raw_keywords"])
            slot["source_id"].extend(_split_sep_values(normalized_edge.get("source_id", "")))
            slot["file_path"].extend(_split_sep_values(normalized_edge.get("file_path", "")))
            slot["created_at"] = max(
                slot["created_at"], _safe_int(normalized_edge.get("created_at"), 0)
            )
            if normalized_edge.get("truncate"):
                slot["truncate"] = normalized_edge.get("truncate", "")
            try:
                slot["weight"] = max(slot["weight"], float(normalized_edge.get("weight", 0.0)))
            except (TypeError, ValueError):
                pass

        # Rewrite graph structure.
        for elem in list(graph):
            if elem.tag.endswith("node") or elem.tag.endswith("edge"):
                graph.remove(elem)

        node_entity_id_key = key_nodes.get("entity_id")
        node_entity_type_key = key_nodes.get("entity_type")
        node_content_modality_key = key_nodes.get("content_modality", node_content_modality_key)
        node_description_key = key_nodes.get("description")
        node_source_id_key = key_nodes.get("source_id")
        node_file_path_key = key_nodes.get("file_path")
        node_created_at_key = key_nodes.get("created_at")
        node_truncate_key = key_nodes.get("truncate")

        edge_keywords_key = key_edges.get("keywords")
        edge_description_key = key_edges.get("description")
        edge_source_id_key = key_edges.get("source_id")
        edge_file_path_key = key_edges.get("file_path")
        edge_created_at_key = key_edges.get("created_at")
        edge_truncate_key = key_edges.get("truncate")
        edge_weight_key = key_edges.get("weight")

        for canonical, node_data in sorted(merged_nodes.items(), key=lambda x: x[0]):
            rendered_node = self._apply_node_description_policy(
                {
                    "entity_type": node_data["entity_type"],
                    "description": _join_sep_values(node_data["description"]),
                }
            )
            node_elem = ET.SubElement(graph, "node", {"id": canonical})
            payload = {
                node_entity_id_key: canonical,
                node_entity_type_key: node_data["entity_type"],
                node_content_modality_key: node_data.get("content_modality", ""),
                node_description_key: rendered_node.get("description", ""),
                node_source_id_key: _join_sep_values(node_data["source_id"]),
                node_file_path_key: _join_sep_values(node_data["file_path"]),
                node_created_at_key: str(node_data["created_at"]),
                node_truncate_key: node_data["truncate"],
                node_aliases_key: _join_sep_values(
                    a for a in node_data["aliases"] if a and a != canonical
                ),
                node_attributes_key: json.dumps(node_data.get("attributes", {}), ensure_ascii=False),
                node_attribute_evidence_key: json.dumps(
                    node_data.get("attribute_evidence", {}), ensure_ascii=False
                ),
            }
            for key_id, value in payload.items():
                if key_id:
                    data_elem = ET.SubElement(node_elem, "data", {"key": key_id})
                    data_elem.text = value

        for index, ((src, tgt, _relation_type), edge_data) in enumerate(
            sorted(merged_edges.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])),
            start=1,
        ):
            rendered_edge = self._apply_edge_description_policy(
                {"description": _join_sep_values(edge_data["description"])}
            )
            edge_elem = ET.SubElement(
                graph,
                "edge",
                {"id": f"e{index}", "source": src, "target": tgt},
            )
            payload = {
                edge_keywords_key: edge_data["keywords"],
                edge_description_key: rendered_edge.get("description", ""),
                edge_source_id_key: _join_sep_values(edge_data["source_id"]),
                edge_file_path_key: _join_sep_values(edge_data["file_path"]),
                edge_created_at_key: str(edge_data["created_at"]),
                edge_truncate_key: edge_data["truncate"],
                edge_weight_key: str(edge_data["weight"]),
                edge_relation_type_key: edge_data["relation_type"],
                edge_raw_keywords_key: _join_sep_values(edge_data["raw_keywords"]),
            }
            for key_id, value in payload.items():
                if key_id:
                    data_elem = ET.SubElement(edge_elem, "data", {"key": key_id})
                    data_elem.text = value

        type_counter = Counter(node_data["entity_type"] for node_data in merged_nodes.values())
        relation_counter = Counter(key[2] for key in merged_edges.keys())
        non_cjk_nodes = sum(
            1 for name in merged_nodes.keys() if not _contains_cjk(name) and _contains_ascii_word(name)
        )
        total_nodes = len(merged_nodes)

        report = {
            "graphml_path": graphml_path,
            "enabled": self.enabled,
            "canonical_language": self.canonical_language,
            "relation_schema": self.relation_schema,
            "ontology_profile": self.ontology_profile,
            "enforce_ontology": self.enforce_ontology,
            "nodes_after": total_nodes,
            "edges_after": len(merged_edges),
            "non_cjk_entity_ratio": (non_cjk_nodes / total_nodes) if total_nodes else 0.0,
            "entity_type_distribution": dict(type_counter),
            "relation_type_distribution": dict(relation_counter),
        }

        if rewrite:
            tree.write(graphml_path, encoding="utf-8", xml_declaration=True)

        return report


def _build_manager_from_args(args: argparse.Namespace) -> KGQualityManager:
    return KGQualityManager(
        enabled=not args.disable,
        canonical_language=args.canonical_language,
        relation_schema=args.relation_schema,
        extraction_mode=args.extraction_mode,
        ontology_profile=args.ontology_profile,
        enforce_ontology=not args.disable_ontology,
        merge_threshold=args.merge_threshold,
        description_policy=args.description_policy,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and normalize RAG GraphML quality.")
    parser.add_argument("--working-dir", default="./rag_storage", help="RAG working directory")
    parser.add_argument(
        "--graphml-path",
        default=None,
        help="GraphML file path. If omitted, uses <working-dir>/graph_chunk_entity_relation.graphml",
    )
    parser.add_argument("--rewrite", action="store_true", help="Rewrite graph file in place")
    parser.add_argument("--disable", action="store_true", help="Disable quality manager logic")
    parser.add_argument("--canonical-language", default="zh", help="Canonical language, e.g. zh")
    parser.add_argument("--relation-schema", default="fixed", help="Relation schema name")
    parser.add_argument("--extraction-mode", default="ppe", help="Extraction mode: ppe or legacy")
    parser.add_argument(
        "--description-policy",
        default="multimodal_only",
        help="Description policy: multimodal_only or keep_all",
    )
    parser.add_argument(
        "--ontology-profile",
        default="cruciferous_pest_disease",
        help="Ontology profile name",
    )
    parser.add_argument(
        "--disable-ontology",
        action="store_true",
        help="Disable domain-range ontology checks",
    )
    parser.add_argument("--merge-threshold", type=float, default=0.86, help="Merge threshold")
    args = parser.parse_args()

    graphml_path = args.graphml_path or os.path.join(
        args.working_dir, "graph_chunk_entity_relation.graphml"
    )
    manager = _build_manager_from_args(args)
    report = manager.clean_graphml_file(graphml_path, rewrite=args.rewrite)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
