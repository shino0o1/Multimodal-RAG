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
from typing import Any, Dict, Iterable, List, Tuple


NODE_DEFAULT_WHITELIST = [
    "病害",
    "虫害",
    "病原菌",
    "病原属类",
    "病害分类",
    "栽培方式",
    "地理位置",
    "植物",
    "植物生长期",
    "部位",
    "药剂",
    "时间",
    "农业防治",
    "化学防治",
    "生物分类",
    "农药",
    "农作物",
    "防治方法",
    "危害症状",
    "形态特征",
    "生活习性",
    "其他",
]

ONTOLOGY_ENTITY_TYPES_CRUCIFEROUS = set(NODE_DEFAULT_WHITELIST)

ONTOLOGY_RELATIONS_CRUCIFEROUS = {
    "易发生病害",
    "致病属类",
    "致病",
    "隶属",
    "病害类型",
    "地理位置",
    "致病生长期",
    "致病部位",
    "治疗药剂",
    "历史记录时间",
    "防治",
    "症状",
    "生命周期",
    "别名",
    "位于",
    "影响",
    "属于",
    "关联",
}

ENTITY_TYPE_ALIASES = {
    "pest": "虫害",
    "insect pest": "虫害",
    "disease": "病害",
    "pathogen": "病原菌",
    "pathogenic bacteria": "病原菌",
    "pathogenic genus": "病原属类",
    "disease category": "病害分类",
    "nursery method": "栽培方式",
    "cultivation method": "栽培方式",
    "location": "地理位置",
    "plant": "植物",
    "growth stage": "植物生长期",
    "plant growth stage": "植物生长期",
    "part": "部位",
    "organ": "部位",
    "pesticide": "药剂",
    "drug": "药剂",
    "time": "时间",
    "crop": "农作物",
    "control method": "防治方法",
    "symptom": "危害症状",
    "morphology": "形态特征",
    "habit": "生活习性",
    "image": "其他",
    "table": "其他",
    "equation": "其他",
    "header": "其他",
    "page_number": "其他",
    "other": "其他",
}

# First-batch agricultural aliases for common entities observed in current corpus.
ENTITY_ALIASES = {
    "small leaf beetle": "小猿叶甲",
    "xiaoyuan leaf beetle": "小猿叶甲",
    "liaognath leaf beetle": "小猿叶甲",
    "large leaf beetles": "大猿叶甲",
    "large leaf beetle": "大猿叶甲",
    "dayuan leaf beetle": "大猿叶甲",
    "leaf beetles": "猿叶甲",
    "small cabbage weevil": "小猿叶甲",
    "large cabbage weevil": "大猿叶甲",
    "yellow-striped flea beetle larvae": "黄曲条跳甲幼虫",
    "cruciferous vegetables": "十字花科蔬菜",
    "cabbage": "甘蓝",
    "radish": "萝卜",
    "adult pests": "成虫",
    "larvae": "幼虫",
    "pests": "虫害",
    "pest populations": "虫群",
    "vegetable plants": "蔬菜植株",
    "vegetable leaf": "蔬菜叶片",
    "leaf damage": "叶片危害",
}

GENERIC_PEST_TERMS = {
    "pest",
    "pests",
    "pest populations",
    "adult pests",
    "larvae",
    "insects",
}

RELATION_SCHEMA_FIXED = {
    "易发生病害",
    "致病属类",
    "致病",
    "隶属",
    "病害类型",
    "地理位置",
    "致病生长期",
    "致病部位",
    "治疗药剂",
    "历史记录时间",
    "属于",
    "影响",
    "防治",
    "症状",
    "生命周期",
    "别名",
    "位于",
    "关联",
}

RELATION_DOMAIN_RANGE_CRUCIFEROUS: Dict[str, set[Tuple[str, str]]] = {
    "易发生病害": {("栽培方式", "病害")},
    "致病属类": {("病原属类", "病害")},
    "致病": {("病原菌", "病害"), ("虫害", "病害")},
    "隶属": {("病原菌", "病原属类")},
    "病害类型": {("病害分类", "病害")},
    "地理位置": {("地理位置", "病害"), ("地理位置", "虫害")},
    "致病生长期": {("病害", "植物生长期"), ("虫害", "植物生长期")},
    "致病部位": {("病害", "部位"), ("虫害", "部位")},
    "治疗药剂": {("病害", "药剂"), ("虫害", "药剂")},
    "历史记录时间": {("病害", "时间"), ("虫害", "时间")},
}


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
    ontology_profile: str = "cruciferous_pest_disease"
    enforce_ontology: bool = True
    merge_threshold: float = 0.86
    allowed_entity_types: List[str] = field(
        default_factory=lambda: NODE_DEFAULT_WHITELIST.copy()
    )
    entity_aliases: Dict[str, str] = field(default_factory=lambda: ENTITY_ALIASES.copy())

    def __post_init__(self):
        if self.ontology_profile in {
            "cruciferous_pest_disease",
            "crop_pest_disease",
            "rice_disease_pest",
        }:
            if self.enforce_ontology:
                self.allowed_entity_types = sorted(ONTOLOGY_ENTITY_TYPES_CRUCIFEROUS)
            else:
                self.allowed_entity_types = sorted(
                    set(self.allowed_entity_types) | ONTOLOGY_ENTITY_TYPES_CRUCIFEROUS
                )

    def _normalize_entity_type(self, entity_type: str) -> str:
        raw = _normalize_space(entity_type)
        if not raw:
            return "其他"
        lowered = raw.lower()
        mapped = ENTITY_TYPE_ALIASES.get(lowered, raw)
        if mapped not in self.allowed_entity_types:
            return "其他"
        return mapped

    def _normalize_entity_name(self, entity_name: str) -> Tuple[str, List[str]]:
        raw = _normalize_space(_clean_suffix(entity_name))
        if not raw:
            return "未命名实体", []

        alias_candidates = [raw]
        if self.canonical_language.lower() == "zh":
            lowered = raw.lower()
            if lowered in self.entity_aliases:
                canonical = self.entity_aliases[lowered]
            elif _contains_cjk(raw):
                canonical = raw
            else:
                # For unknown English-only entities, keep original and allow later fallback by type.
                canonical = raw
            if canonical != raw:
                alias_candidates.append(raw)
            return canonical, alias_candidates

        return raw, alias_candidates

    def _enforce_type_name_consistency(
        self, entity_name: str, entity_type: str
    ) -> Tuple[str, str]:
        raw_name = _normalize_space(_clean_suffix(entity_name))
        raw_lowered = raw_name.lower()
        normalized_type = self._normalize_entity_type(entity_type)
        normalized_name, aliases = self._normalize_entity_name(entity_name)
        lowered = normalized_name.lower()

        # Rule: generic English terms should not stay as pest nodes.
        if normalized_type == "虫害" and (
            lowered in GENERIC_PEST_TERMS or raw_lowered in GENERIC_PEST_TERMS
        ):
            normalized_type = "其他"

        # If canonical language is zh and name is still English-only, downgrade to "其他"
        # unless the type is already "其他".
        if (
            self.canonical_language.lower() == "zh"
            and not _contains_cjk(normalized_name)
            and _contains_ascii_word(normalized_name)
            and normalized_type != "其他"
        ):
            normalized_type = "其他"

        return normalized_name, normalized_type

    def normalize_entity(
        self, entity_name: str, entity_type: str
    ) -> Dict[str, Any]:
        canonical_name, canonical_type = self._enforce_type_name_consistency(
            entity_name, entity_type
        )
        _, aliases = self._normalize_entity_name(entity_name)
        aliases = [a for a in aliases if a and a != canonical_name]
        return {
            "entity_name": canonical_name,
            "entity_type": canonical_type,
            "aliases": aliases,
        }

    def map_relation_type(
        self,
        keywords: str,
        description: str = "",
        src_type: str = "",
        tgt_type: str = "",
    ) -> str:
        if self.relation_schema != "fixed":
            return "关联"

        text = f"{keywords or ''} {description or ''}".lower()

        if any(
            k in text
            for k in [
                "nursery",
                "seedling",
                "育秧",
                "栽培",
                "cultivation",
                "易发生病害",
                "occurs in nursery",
            ]
        ):
            relation = "易发生病害"
        elif any(k in text for k in ["pathogenic genus", "属类", "致病属类"]):
            relation = "致病属类"
        elif any(k in text for k in ["pathogen", "致病菌", "致病"]):
            relation = "致病"
        elif any(k in text for k in ["belong to genus", "隶属"]):
            relation = "隶属"
        elif any(k in text for k in ["disease type", "分类", "病害类型"]):
            relation = "病害类型"
        elif any(k in text for k in ["growth stage", "生长期", "发生时期"]):
            relation = "致病生长期"
        elif any(k in text for k in ["infected part", "部位", "致病部位"]):
            relation = "致病部位"
        elif any(k in text for k in ["drug", "pesticide", "药剂", "治疗药剂"]):
            relation = "治疗药剂"
        elif any(k in text for k in ["history", "record time", "时间", "历史记录时间"]):
            relation = "历史记录时间"
        if any(k in text for k in ["belongs_to", "part_of", "contained_in", "属于"]):
            relation = "属于"
        elif any(k in text for k in ["control", "prevention", "management", "防治"]):
            relation = "防治"
        elif any(k in text for k in ["symptom", "damage", "harm", "危害"]):
            relation = "症状"
        elif any(
            k in text
            for k in ["life cycle", "lifecycle", "stage", "overwinter", "生命周期"]
        ):
            relation = "生命周期"
        elif any(k in text for k in ["alias", "same as", "aka", "别名"]):
            relation = "别名"
        elif any(k in text for k in ["location", "located", "位于", "地理位置", "page "]):
            relation = "位于"
        elif any(k in text for k in ["impact", "affect", "cause", "影响"]):
            relation = "影响"
        else:
            relation = "关联"

        if self.enforce_ontology and not self._relation_allowed(
            relation, src_type, tgt_type
        ):
            return "关联"
        return relation

    def _relation_allowed(self, relation: str, src_type: str, tgt_type: str) -> bool:
        if relation in {"属于", "关联", "别名", "防治", "症状", "生命周期", "位于", "影响"}:
            return True
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
        return normalized

    def preprocess_chunk_results(self, chunk_results: List[Tuple]) -> List[Tuple]:
        if not self.enabled:
            return chunk_results

        normalized_results = []

        for maybe_nodes, maybe_edges in chunk_results:
            alias_map: Dict[str, str] = {}
            new_nodes: Dict[str, List[Dict[str, Any]]] = {}
            node_type_map: Dict[str, str] = {}

            for raw_name, node_list in maybe_nodes.items():
                # best-effort entity_type from first node record
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
                    aliases = _split_sep_values(str(nd.get("aliases", "")))
                    aliases.extend(norm["aliases"])
                    if raw_name != canonical:
                        aliases.append(raw_name)
                    nd["aliases"] = _join_sep_values(aliases)
                    new_nodes[canonical].append(nd)

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
                    new_edges[key].append(
                        self.normalize_edge(
                            ed,
                            src_type=node_type_map.get(norm_src, ""),
                            tgt_type=node_type_map.get(norm_tgt, ""),
                        )
                    )

            normalized_results.append((new_nodes, new_edges))

        return normalized_results

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
                    "description": [],
                    "source_id": [],
                    "file_path": [],
                    "created_at": 0,
                    "truncate": "",
                    "aliases": set(norm["aliases"]),
                }

            slot = merged_nodes[canonical]
            if slot["entity_type"] == "其他" and norm["entity_type"] != "其他":
                slot["entity_type"] = norm["entity_type"]

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
            relation_type = normalized_edge.get("relation_type", "关联")
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
            node_elem = ET.SubElement(graph, "node", {"id": canonical})
            payload = {
                node_entity_id_key: canonical,
                node_entity_type_key: node_data["entity_type"],
                node_description_key: _join_sep_values(node_data["description"]),
                node_source_id_key: _join_sep_values(node_data["source_id"]),
                node_file_path_key: _join_sep_values(node_data["file_path"]),
                node_created_at_key: str(node_data["created_at"]),
                node_truncate_key: node_data["truncate"],
                node_aliases_key: _join_sep_values(
                    a for a in node_data["aliases"] if a and a != canonical
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
            edge_elem = ET.SubElement(
                graph,
                "edge",
                {"id": f"e{index}", "source": src, "target": tgt},
            )
            payload = {
                edge_keywords_key: edge_data["keywords"],
                edge_description_key: _join_sep_values(edge_data["description"]),
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
        ontology_profile=args.ontology_profile,
        enforce_ontology=not args.disable_ontology,
        merge_threshold=args.merge_threshold,
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
