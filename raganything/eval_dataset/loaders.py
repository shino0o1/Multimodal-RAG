"""Load local RAG storage into evidence packs for eval generation."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SEP = "<SEP>"

GENERIC_RELATION_TERMS = {
    "相关",
    "症状",
    "危害",
    "防治",
    "发生",
    "药剂",
    "识别",
    "诊断",
    "方法",
    "用法",
}

AGRICULTURAL_HINT_TERMS = [
    "症状",
    "危害",
    "防治",
    "药剂",
    "病斑",
    "叶片",
    "倍液",
    "幼虫",
    "成虫",
    "喷雾",
    "轮作",
]


@dataclass
class GraphNode:
    id: str
    entity_type: str = ""
    description: str = ""
    source_ids: List[str] = field(default_factory=list)
    file_path: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    relation_type: str = ""
    description: str = ""
    source_ids: List[str] = field(default_factory=list)
    file_path: str = ""


@dataclass
class Evidence:
    source_type: str
    chunk_id: str
    file_path: str
    quote: str
    focus_entities: List[str] = field(default_factory=list)


@dataclass
class EvidencePack:
    pack_id: str
    task_seed: str
    core_entity: str
    entity_type: str
    expected_entities: List[str]
    expected_relations: List[str]
    evidence: List[Evidence]
    context: str
    modality: str = "text"
    image_path: Optional[str] = None
    image_labels: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class KnowledgeBase:
    rag_dir: Path
    chunks: Dict[str, Dict[str, Any]]
    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]
    entity_chunks: Dict[str, Dict[str, Any]]
    relation_chunks: Dict[str, Dict[str, Any]]
    multimodal_cache: Dict[str, Dict[str, Any]]

    @property
    def entity_names(self) -> set[str]:
        return set(self.nodes) | set(self.entity_chunks)


def load_knowledge_base(rag_dir: str | Path) -> KnowledgeBase:
    base = Path(rag_dir)
    if not base.exists():
        raise FileNotFoundError(f"RAG storage directory does not exist: {base}")

    chunks = _read_json(base / "kv_store_text_chunks.json", required=True)
    entity_chunks = _read_json(base / "kv_store_entity_chunks.json", required=False)
    relation_chunks = _read_json(base / "kv_store_relation_chunks.json", required=False)
    multimodal_cache = _read_json(
        base / "kv_store_multimodal_desc_cache.json", required=False
    )
    nodes, edges = _read_graphml(base / "graph_chunk_entity_relation.graphml")

    return KnowledgeBase(
        rag_dir=base,
        chunks=chunks,
        nodes=nodes,
        edges=edges,
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
        multimodal_cache=multimodal_cache,
    )


def load_image_manifest(path: str | Path | None) -> List[Dict[str, Any]]:
    if not path:
        return []
    manifest = Path(path)
    if not manifest.exists():
        raise FileNotFoundError(f"Image manifest does not exist: {manifest}")

    rows: List[Dict[str, Any]] = []
    if manifest.suffix.lower() == ".json":
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = [r for r in payload if isinstance(r, dict)]
        elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
            rows = [r for r in payload["items"] if isinstance(r, dict)]
        else:
            raise ValueError("JSON image manifest must be a list or {'items': [...]}")
    else:
        for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {manifest}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def build_evidence_packs(
    kb: KnowledgeBase,
    image_manifest: Optional[List[Dict[str, Any]]] = None,
    max_context_chars: int = 2200,
) -> List[EvidencePack]:
    packs: List[EvidencePack] = []

    for node in kb.nodes.values():
        if node.entity_type not in {
            "病害",
            "虫害",
            "病原菌",
            "药剂",
            "作物",
            "部位",
            "生长期",
        }:
            continue
        evidence = _evidence_for_chunk_ids(kb, node.source_ids, [node.id])
        if not evidence:
            continue
        related_edges = _related_edges(kb.edges, node.id)
        relations = _unique([e.relation_type for e in related_edges if e.relation_type])
        neighbor_entities = _unique(
            [
                e.target if e.source == node.id else e.source
                for e in related_edges
                if e.source == node.id or e.target == node.id
            ]
        )
        context = _context_from_evidence(kb, evidence, max_context_chars)
        packs.append(
            EvidencePack(
                pack_id=f"entity::{node.id}",
                task_seed="entity",
                core_entity=node.id,
                entity_type=node.entity_type,
                expected_entities=_unique([node.id] + neighbor_entities[:4]),
                expected_relations=relations[:5],
                evidence=evidence[:3],
                context=context,
            )
        )

    for idx, edge in enumerate(kb.edges):
        if not edge.source_ids:
            continue
        evidence = _evidence_for_chunk_ids(
            kb, edge.source_ids, [edge.source, edge.target, edge.relation_type]
        )
        if not evidence:
            continue
        source_type = kb.nodes.get(edge.source, GraphNode(edge.source)).entity_type
        target_type = kb.nodes.get(edge.target, GraphNode(edge.target)).entity_type
        context = _context_from_evidence(kb, evidence, max_context_chars)
        packs.append(
            EvidencePack(
                pack_id=f"relation::{idx}::{edge.source}::{edge.target}",
                task_seed="relation",
                core_entity=edge.source,
                entity_type=source_type or target_type,
                expected_entities=_unique([edge.source, edge.target]),
                expected_relations=[edge.relation_type or "相关"],
                evidence=evidence[:3],
                context=context,
            )
        )

    packs.extend(_build_image_packs(kb, image_manifest or [], max_context_chars))
    return packs


def _read_json(path: Path, required: bool) -> Dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required RAG storage file missing: {path}")
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _read_graphml(path: Path) -> tuple[Dict[str, GraphNode], List[GraphEdge]]:
    if not path.exists():
        return {}, []

    root = ET.parse(path).getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    key_map = {
        item.attrib.get("id", ""): item.attrib.get("attr.name", "")
        for item in root.findall("g:key", ns)
    }

    graph = root.find("g:graph", ns)
    if graph is None:
        return {}, []

    nodes: Dict[str, GraphNode] = {}
    for node in graph.findall("g:node", ns):
        attrs = _graph_attrs(node, key_map, ns)
        node_id = attrs.get("entity_id") or node.attrib.get("id", "")
        if not node_id:
            continue
        nodes[node_id] = GraphNode(
            id=node_id,
            entity_type=attrs.get("entity_type", ""),
            description=attrs.get("description", ""),
            source_ids=_split_source_ids(attrs.get("source_id", "")),
            file_path=attrs.get("file_path", ""),
            attributes=_loads_dict(attrs.get("attributes", "{}")),
        )

    edges: List[GraphEdge] = []
    for edge in graph.findall("g:edge", ns):
        attrs = _graph_attrs(edge, key_map, ns)
        edges.append(
            GraphEdge(
                source=edge.attrib.get("source", ""),
                target=edge.attrib.get("target", ""),
                relation_type=attrs.get("relation_type") or attrs.get("keywords", ""),
                description=attrs.get("description", ""),
                source_ids=_split_source_ids(attrs.get("source_id", "")),
                file_path=attrs.get("file_path", ""),
            )
        )
    return nodes, edges


def _graph_attrs(item: ET.Element, key_map: Dict[str, str], ns: Dict[str, str]) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for data in item.findall("g:data", ns):
        key = key_map.get(data.attrib.get("key", ""), data.attrib.get("key", ""))
        attrs[key] = (data.text or "").strip()
    return attrs


def _split_source_ids(value: str) -> List[str]:
    return _unique([part.strip() for part in (value or "").split(SEP) if part.strip()])


def _loads_dict(value: str) -> Dict[str, Any]:
    try:
        payload = json.loads(value or "{}")
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _related_edges(edges: Iterable[GraphEdge], entity: str) -> List[GraphEdge]:
    return [edge for edge in edges if edge.source == entity or edge.target == entity]


def _evidence_for_chunk_ids(
    kb: KnowledgeBase, chunk_ids: Iterable[str], keywords: List[str]
) -> List[Evidence]:
    evidence: List[Evidence] = []
    for chunk_id in _unique(list(chunk_ids)):
        chunk = kb.chunks.get(chunk_id)
        if not isinstance(chunk, dict):
            continue
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue
        quote = _select_quote(content, keywords)
        if not quote:
            continue
        evidence.append(
            Evidence(
                source_type="text_chunk",
                chunk_id=chunk_id,
                file_path=str(chunk.get("file_path", "") or ""),
                quote=quote,
                focus_entities=[str(item) for item in keywords if str(item).strip()],
            )
        )
    return evidence


def _select_quote(content: str, keywords: List[str], max_len: int = 420) -> str:
    cleaned = _clean_quote_source(content)
    if not cleaned:
        return ""
    sentences = _split_sentences(cleaned)
    primary = [str(item).strip() for item in keywords if str(item).strip()]
    focus_terms = [
        item
        for item in primary
        if item not in GENERIC_RELATION_TERMS and len(item) >= 2
    ]
    if not focus_terms:
        return ""

    scored: List[tuple[int, int]] = []
    for idx, sent in enumerate(sentences):
        focus_hits = [key for key in focus_terms if key in sent]
        if not focus_hits:
            continue
        score = 3 * len(focus_hits)
        score += sum(1 for term in AGRICULTURAL_HINT_TERMS if term in sent)
        if score > 0:
            scored.append((score, idx))
    if not scored:
        return ""

    _, best_idx = max(scored, key=lambda item: (item[0], -item[1]))
    window_indices = [best_idx]
    for neighbor_idx in (best_idx - 1, best_idx + 1):
        if neighbor_idx < 0 or neighbor_idx >= len(sentences):
            continue
        if any(term in sentences[neighbor_idx] for term in focus_terms):
            window_indices.append(neighbor_idx)
    quote = "".join(sentences[idx] for idx in sorted(window_indices)).strip()
    if len(quote) > max_len:
        quote = _trim_around_focus(quote, focus_terms, max_len)
    return quote


def _split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[。！？；;])\s*|(?:\n+)", text)
    sentences = [item.strip() for item in chunks if item.strip()]
    return sentences or [text.strip()]


def _trim_around_focus(text: str, focus_terms: List[str], max_len: int) -> str:
    positions = [text.find(term) for term in focus_terms if term and text.find(term) >= 0]
    if not positions:
        return text[:max_len].rstrip()
    center = min(positions)
    start = max(0, center - max_len // 3)
    end = min(len(text), start + max_len)
    start = max(0, end - max_len)
    piece = text[start:end].strip()
    if start > 0:
        piece = "..." + piece
    if end < len(text):
        piece = piece + "..."
    return piece


def _clean_quote_source(content: str) -> str:
    cleaned = re.sub(r"\s+", " ", content or "").strip()
    cleaned = re.sub(
        r"(?:图片路径|Image Path)\s*[:：]\s*\S+\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"图片内容分析\s*[:：]\s*", "", cleaned)
    cleaned = re.sub(r"标注\s*[:：]\s*None\s*", "", cleaned)
    cleaned = re.sub(r"脚注\s*[:：]\s*None\s*", "", cleaned)
    return cleaned.strip()


def _context_from_evidence(
    kb: KnowledgeBase, evidence: List[Evidence], max_context_chars: int
) -> str:
    parts: List[str] = []
    used = 0
    for item in evidence:
        chunk = kb.chunks.get(item.chunk_id, {})
        content = re.sub(r"\s+", " ", str(chunk.get("content", ""))).strip()
        content = _select_quote(content, item.focus_entities, max_len=900) or item.quote
        if not content:
            continue
        remaining = max_context_chars - used
        if remaining <= 0:
            break
        piece = content[:remaining]
        parts.append(f"[{item.chunk_id}] {piece}")
        used += len(piece)
    return "\n".join(parts)


def _build_image_packs(
    kb: KnowledgeBase, manifest: List[Dict[str, Any]], max_context_chars: int
) -> List[EvidencePack]:
    packs: List[EvidencePack] = []
    for idx, row in enumerate(manifest):
        image_path = str(row.get("image_path", "")).strip()
        labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
        notes = str(row.get("notes", "")).strip()
        label_terms = _flatten_label_terms(labels)
        target = str(labels.get("target", "") or "").strip()
        if target and target not in kb.entity_names:
            continue
        matched_entities = [term for term in label_terms if term in kb.entity_names]
        if target:
            matched_entities = _unique([target] + matched_entities)
        if not matched_entities:
            continue

        chunk_ids: List[str] = []
        for entity in matched_entities:
            if entity in kb.nodes:
                chunk_ids.extend(kb.nodes[entity].source_ids)
            if entity in kb.entity_chunks:
                chunk_ids.extend(kb.entity_chunks[entity].get("chunk_ids") or [])
        evidence = _evidence_for_chunk_ids(kb, chunk_ids, matched_entities)
        if not evidence:
            continue
        context = _context_from_evidence(kb, evidence, max_context_chars)
        target = target or matched_entities[0]
        packs.append(
            EvidencePack(
                pack_id=f"image::{idx}::{Path(image_path).name}",
                task_seed="image",
                core_entity=target,
                entity_type=kb.nodes.get(target, GraphNode(target)).entity_type or "图像实体",
                expected_entities=_unique(matched_entities),
                expected_relations=[],
                evidence=evidence[:3],
                context=context,
                modality="image",
                image_path=image_path,
                image_labels=labels,
                notes=notes,
            )
        )
    return packs


def _flatten_label_terms(labels: Dict[str, Any]) -> List[str]:
    terms: List[str] = []
    for value in labels.values():
        if isinstance(value, str):
            terms.append(value.strip())
        elif isinstance(value, list):
            terms.extend(str(item).strip() for item in value if str(item).strip())
    return _unique([term for term in terms if term])


def _unique(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
