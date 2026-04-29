"""Service layer for Streamlit-based RAG-Anything UI."""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import os
import re
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from .job_manager import JobManager
except ImportError:
    # Fallback when imported as top-level module (e.g. from ui/app.py local import).
    from job_manager import JobManager

CHUNK_ID_PATTERN = re.compile(r"\bchunk-[0-9a-f]{32}\b", re.IGNORECASE)
IMAGE_PATH_PATTERN = re.compile(
    r"(?:Image\s*Path|图片路径)\s*[:：]\s*([^\n\r]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))",
    re.IGNORECASE,
)
SUPPORTED_UPLOAD_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
}
SUPPORTED_QUERY_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Citation:
    chunk_id: str
    file_path: str
    page_idx: int
    snippet: str
    modality: str
    source_type: str
    asset_ref: Dict[str, Any]


@dataclass
class QueryResponse:
    answer: str
    citations: List[Dict[str, Any]]
    graph_focus: Dict[str, List[str]]
    debug: Dict[str, Any]
    timings: Dict[str, float]


def load_api_config_from_pipeline(script_path: Optional[Path] = None) -> Dict[str, str]:
    """Load api_key/base_url from pdf_rag_pipeline.py style assignments.

    This reads assignments inside `main()` like:
    `api_key = "..."`
    `base_url = "..."`
    """
    target = (
        script_path
        if script_path is not None
        else (Path(__file__).resolve().parents[1] / "pdf_rag_pipeline.py")
    )
    if not target.exists():
        return {}

    try:
        source = target.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return {}

    values: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "main":
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                    continue
                tgt = stmt.targets[0]
                if not isinstance(tgt, ast.Name):
                    continue
                if tgt.id not in {"api_key", "base_url"}:
                    continue
                if isinstance(stmt.value, ast.Constant) and isinstance(
                    stmt.value.value, str
                ):
                    values[tgt.id] = stmt.value.value.strip()
            break

    return values


def extract_chunk_ids(text: str) -> List[str]:
    """Extract ordered unique chunk ids from text."""
    if not text:
        return []
    seen = set()
    ordered: List[str] = []
    for m in CHUNK_ID_PATTERN.finditer(text):
        cid = m.group(0)
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered


def extract_image_paths(text: str) -> List[str]:
    """Extract ordered unique image paths from retrieval context."""
    if not text:
        return []
    seen = set()
    ordered: List[str] = []
    for m in IMAGE_PATH_PATTERN.finditer(text):
        path = m.group(1).strip()
        if path and path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def split_source_ids(value: Any) -> List[str]:
    """Split GraphML source ids joined by `<SEP>` or commas."""
    if value is None:
        return []

    if isinstance(value, list):
        result: List[str] = []
        seen = set()
        for item in value:
            for part in split_source_ids(item):
                if part not in seen:
                    seen.add(part)
                    result.append(part)
        return result

    raw = str(value).strip()
    if not raw:
        return []

    raw = raw.replace("&lt;SEP&gt;", "<SEP>")
    parts = re.split(r"<SEP>|[;,]\s*", raw)

    seen = set()
    result = []
    for part in parts:
        item = part.strip()
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def select_subgraph_by_chunks(
    nodes: List[Dict[str, Any]],
    links: List[Dict[str, Any]],
    chunk_ids: Iterable[str],
    full: bool = False,
    max_nodes: int = 180,
    max_edges: int = 360,
) -> Dict[str, Any]:
    """Select full graph or answer-focused subgraph by chunk ids."""
    chunk_set = {c for c in chunk_ids if c}

    highlight_node_ids = {
        n["id"] for n in nodes if chunk_set.intersection(set(n.get("source_ids", [])))
    }
    highlight_edge_ids = {
        e["id"] for e in links if chunk_set.intersection(set(e.get("source_ids", [])))
    }

    if full:
        return {
            "nodes": nodes,
            "links": links,
            "highlight_node_ids": sorted(highlight_node_ids),
            "highlight_edge_ids": sorted(highlight_edge_ids),
        }

    if chunk_set:
        edge_filtered = [
            e for e in links if chunk_set.intersection(set(e.get("source_ids", [])))
        ]
        node_ids = set(highlight_node_ids)
        for edge in edge_filtered:
            node_ids.add(edge.get("source"))
            node_ids.add(edge.get("target"))

        node_filtered = [n for n in nodes if n.get("id") in node_ids]

        if node_filtered or edge_filtered:
            return {
                "nodes": node_filtered[:max_nodes],
                "links": edge_filtered[:max_edges],
                "highlight_node_ids": sorted(highlight_node_ids),
                "highlight_edge_ids": sorted(highlight_edge_ids),
            }

    fallback_nodes = nodes[:max_nodes]
    fallback_node_ids = {n.get("id") for n in fallback_nodes}
    fallback_links = [
        e
        for e in links
        if e.get("source") in fallback_node_ids and e.get("target") in fallback_node_ids
    ][:max_edges]

    return {
        "nodes": fallback_nodes,
        "links": fallback_links,
        "highlight_node_ids": sorted(highlight_node_ids),
        "highlight_edge_ids": sorted(highlight_edge_ids),
    }


class RAGUIService:
    """Stateful service for KB creation, query, and graph visualization."""

    def __init__(
        self,
        rag_storage_root: str = "rag_storage_ui",
        output_root: str = "output_ui",
        uploads_root: str = "uploads_ui",
        meta_root: str = "meta_ui",
        parser: str = "mineru",
    ) -> None:
        self.parser = parser
        self.rag_storage_root = Path(rag_storage_root)
        self.output_root = Path(output_root)
        self.uploads_root = Path(uploads_root)
        self.meta_root = Path(meta_root)

        for path in [
            self.rag_storage_root,
            self.output_root,
            self.uploads_root,
            self.meta_root,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self._rag_cache: Dict[str, Any] = {}
        self._job_manager = JobManager(self._build_rag, on_job_update=self._on_job_update)
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(
            target=self._run_async_loop_forever,
            name="rag-ui-async-loop",
            daemon=True,
        )
        self._async_thread.start()

    @staticmethod
    def supported_upload_types() -> List[str]:
        """Return supported upload extensions for Streamlit file_uploader(type=...)."""
        return sorted(s.lstrip(".") for s in SUPPORTED_UPLOAD_SUFFIXES)

    @staticmethod
    def supported_query_image_types() -> List[str]:
        """Return supported query-image extensions for Streamlit file_uploader(type=...)."""
        return sorted(s.lstrip(".") for s in SUPPORTED_QUERY_IMAGE_SUFFIXES)

    def _run_async_loop_forever(self) -> None:
        """Run a dedicated event loop for all async LightRAG operations."""
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()

    def _run_async(self, coro: Any) -> Any:
        """Run coroutine on the dedicated persistent event loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
        return future.result()

    def _build_rag(
        self,
        working_dir: str,
        output_dir: str,
        callback: Any,
    ) -> Any:
        """Create a RAGAnything instance for a single KB."""
        try:
            from lightrag.llm.openai import openai_complete_if_cache, openai_embed
            from lightrag.utils import EmbeddingFunc
            from raganything import RAGAnything, RAGAnythingConfig, set_prompt_language
            from raganything.callbacks import ProcessingCallback
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency `lightrag-hku`. Install requirements first."
            ) from exc

        fallback_cfg = load_api_config_from_pipeline()

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            api_key = fallback_cfg.get("api_key", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for UI querying/ingestion "
                "(or define api_key in pdf_rag_pipeline.py main())"
            )

        base_url_env = os.getenv("OPENAI_BASE_URL", "").strip()
        base_url = base_url_env or fallback_cfg.get("base_url", "").strip() or None

        config = RAGAnythingConfig(
            working_dir=working_dir,
            parser_output_dir=output_dir,
            parser=self.parser,
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=False,
            kg_quality_enabled=True,
        )

        # Backward compatibility for old UI env variable names.
        legacy_llm_model = os.getenv("RAG_UI_LLM_MODEL", "").strip()
        legacy_vision_model = os.getenv("RAG_UI_VISION_MODEL", "").strip()
        legacy_embed_model = os.getenv("RAG_UI_EMBED_MODEL", "").strip()
        if legacy_llm_model:
            config.model_answer = legacy_llm_model
        if legacy_vision_model:
            config.model_vision = legacy_vision_model
            if not os.getenv("RAG_MODEL_IMAGE_DESCRIPTION", "").strip():
                config.model_image_description = legacy_vision_model
        if legacy_embed_model:
            config.model_embedding = legacy_embed_model

        llm_model = config.model_answer
        planner_model = config.model_planner
        vision_model = config.model_vision
        image_desc_model = config.model_image_description
        embedding_model = config.model_embedding
        embedding_dim = config.embedding_dim

        set_prompt_language("zh")
        os.environ.setdefault("SUMMARY_LANGUAGE", "Chinese")

        def _text_model_func(
            model_name: str, prompt, system_prompt=None, history_messages=[], **kwargs
        ):
            return openai_complete_if_cache(
                model_name,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return _text_model_func(
                llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )

        def planner_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return _text_model_func(
                planner_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )

        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            if messages:
                return openai_complete_if_cache(
                    vision_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )

            if image_data:
                return openai_complete_if_cache(
                    vision_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt}
                        if system_prompt
                        else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        },
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )

            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        def image_description_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            if messages:
                return openai_complete_if_cache(
                    image_desc_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )

            if image_data:
                return openai_complete_if_cache(
                    image_desc_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt}
                        if system_prompt
                        else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        },
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )

            return planner_model_func(prompt, system_prompt, history_messages, **kwargs)

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: openai_embed.func(
                texts,
                model=embedding_model,
                api_key=api_key,
                base_url=base_url,
            ),
        )

        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            planner_model_func=planner_model_func,
            vision_model_func=vision_model_func,
            image_description_model_func=image_description_model_func,
            embedding_func=embedding_func,
        )
        if callback is None:
            callback = ProcessingCallback()
        elif not isinstance(callback, ProcessingCallback):
            delegate = callback

            class _DelegatingCallback(ProcessingCallback):
                def _forward(self, name: str, *args: Any, **kwargs: Any) -> None:
                    fn = getattr(delegate, name, None)
                    if callable(fn):
                        fn(*args, **kwargs)

                def on_parse_start(self, file_path: str, parser: str = "", **kwargs: Any) -> None:
                    self._forward("on_parse_start", file_path=file_path, parser=parser, **kwargs)

                def on_parse_complete(self, file_path: str, content_blocks: int = 0, doc_id: str = "", duration_seconds: float = 0.0, **kwargs: Any) -> None:
                    self._forward(
                        "on_parse_complete",
                        file_path=file_path,
                        content_blocks=content_blocks,
                        doc_id=doc_id,
                        duration_seconds=duration_seconds,
                        **kwargs,
                    )

                def on_parse_error(self, file_path: str, error: BaseException | str = "", **kwargs: Any) -> None:
                    self._forward("on_parse_error", file_path=file_path, error=error, **kwargs)

                def on_text_insert_start(self, file_path: str, text_length: int = 0, **kwargs: Any) -> None:
                    self._forward("on_text_insert_start", file_path=file_path, text_length=text_length, **kwargs)

                def on_text_insert_complete(self, file_path: str, duration_seconds: float = 0.0, **kwargs: Any) -> None:
                    self._forward("on_text_insert_complete", file_path=file_path, duration_seconds=duration_seconds, **kwargs)

                def on_multimodal_start(self, file_path: str, item_count: int = 0, **kwargs: Any) -> None:
                    self._forward("on_multimodal_start", file_path=file_path, item_count=item_count, **kwargs)

                def on_multimodal_item_complete(self, file_path: str, item_index: int = 0, item_type: str = "", total_items: int = 0, **kwargs: Any) -> None:
                    self._forward(
                        "on_multimodal_item_complete",
                        file_path=file_path,
                        item_index=item_index,
                        item_type=item_type,
                        total_items=total_items,
                        **kwargs,
                    )

                def on_multimodal_complete(self, file_path: str, processed_count: int = 0, duration_seconds: float = 0.0, **kwargs: Any) -> None:
                    self._forward(
                        "on_multimodal_complete",
                        file_path=file_path,
                        processed_count=processed_count,
                        duration_seconds=duration_seconds,
                        **kwargs,
                    )

                def on_query_start(self, query: str, mode: str = "", **kwargs: Any) -> None:
                    self._forward("on_query_start", query=query, mode=mode, **kwargs)

                def on_query_complete(self, query: str, mode: str = "", duration_seconds: float = 0.0, result_length: int = 0, **kwargs: Any) -> None:
                    self._forward(
                        "on_query_complete",
                        query=query,
                        mode=mode,
                        duration_seconds=duration_seconds,
                        result_length=result_length,
                        **kwargs,
                    )

                def on_query_error(self, query: str, mode: str = "", error: BaseException | str = "", **kwargs: Any) -> None:
                    self._forward("on_query_error", query=query, mode=mode, error=error, **kwargs)

                def on_document_complete(self, file_path: str, doc_id: str = "", duration_seconds: float = 0.0, **kwargs: Any) -> None:
                    self._forward(
                        "on_document_complete",
                        file_path=file_path,
                        doc_id=doc_id,
                        duration_seconds=duration_seconds,
                        **kwargs,
                    )

                def on_document_error(self, file_path: str, error: BaseException | str = "", stage: str = "", **kwargs: Any) -> None:
                    self._forward("on_document_error", file_path=file_path, error=error, stage=stage, **kwargs)

                def on_batch_start(self, file_count: int = 0, **kwargs: Any) -> None:
                    self._forward("on_batch_start", file_count=file_count, **kwargs)

                def on_batch_complete(self, total_files: int = 0, successful: int = 0, failed: int = 0, duration_seconds: float = 0.0, **kwargs: Any) -> None:
                    self._forward(
                        "on_batch_complete",
                        total_files=total_files,
                        successful=successful,
                        failed=failed,
                        duration_seconds=duration_seconds,
                        **kwargs,
                    )

            callback = _DelegatingCallback()
        rag.callback_manager.register(callback)
        return rag

    def _on_job_update(self, payload: Dict[str, Any]) -> None:
        """Sync job updates to KB metadata."""
        kb_id = payload.get("kb_id")
        if not kb_id:
            return

        meta = self.get_kb_meta(kb_id)
        if not meta:
            return

        meta["status"] = payload.get("status", meta.get("status", ""))
        meta["stage"] = payload.get("stage", meta.get("stage", ""))
        meta["progress"] = int(payload.get("progress", meta.get("progress", 0)))
        meta["error"] = payload.get("error", meta.get("error", ""))
        meta["job_id"] = payload.get("job_id", meta.get("job_id", ""))
        meta["updated_at"] = time.time()
        self._write_meta(kb_id, meta)

    def _meta_path(self, kb_id: str) -> Path:
        return self.meta_root / f"{kb_id}.json"

    def _write_meta(self, kb_id: str, payload: Dict[str, Any]) -> None:
        with self._meta_path(kb_id).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _read_uploaded_bytes(self, uploaded_file: Any) -> bytes:
        """Read bytes from Streamlit UploadedFile-like object safely."""
        if hasattr(uploaded_file, "getvalue"):
            data = uploaded_file.getvalue()
        elif hasattr(uploaded_file, "read"):
            data = uploaded_file.read()
        else:
            data = bytes(uploaded_file)
        return data or b""

    def _save_uploaded_file(self, kb_id: str, uploaded_file: Any) -> Path:
        filename = getattr(uploaded_file, "name", "uploaded")
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
            allowed = ", ".join(sorted(SUPPORTED_UPLOAD_SUFFIXES))
            raise ValueError(f"Unsupported upload type: {suffix or '<none>'}. Allowed: {allowed}")

        kb_upload_dir = self.uploads_root / kb_id
        kb_upload_dir.mkdir(parents=True, exist_ok=True)

        base_name = Path(filename).name
        target = kb_upload_dir / base_name
        if target.exists():
            stem = Path(base_name).stem
            ext = Path(base_name).suffix
            target = kb_upload_dir / f"{stem}_{uuid.uuid4().hex[:6]}{ext}"
        data = self._read_uploaded_bytes(uploaded_file)
        if not data:
            raise ValueError("Uploaded file is empty")

        with target.open("wb") as f:
            f.write(data)
        return target

    def _save_query_image_input(self, kb_id: str, image_file: Any) -> Path:
        """Save ad-hoc query image (not ingested into KB) under uploads folder."""
        filename = getattr(image_file, "name", "query_image.png")
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_QUERY_IMAGE_SUFFIXES:
            allowed = ", ".join(sorted(SUPPORTED_QUERY_IMAGE_SUFFIXES))
            raise ValueError(
                f"Unsupported query image type: {suffix or '<none>'}. Allowed: {allowed}"
            )

        kb_upload_dir = self.uploads_root / kb_id / "query_inputs"
        kb_upload_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        target = kb_upload_dir / f"{ts}_{uuid.uuid4().hex[:6]}_{Path(filename).name}"

        data = self._read_uploaded_bytes(image_file)
        if not data:
            raise ValueError("Uploaded query image is empty")

        with target.open("wb") as f:
            f.write(data)
        return target.resolve()

    def register_existing_kb(
        self,
        working_dir: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register an existing local KB storage directory for direct querying."""
        if not working_dir:
            raise ValueError("working_dir is required")

        working_path = Path(working_dir).expanduser().resolve()
        if not working_path.exists() or not working_path.is_dir():
            raise ValueError(
                f"working_dir does not exist or is not a directory: {working_dir}"
            )

        graph_file = working_path / "graph_chunk_entity_relation.graphml"
        chunks_file = working_path / "kv_store_text_chunks.json"
        if not graph_file.exists() and not chunks_file.exists():
            raise ValueError(
                "Directory is not a valid KB storage (missing graph/chunk storage files)"
            )

        for item in self.list_kbs():
            existing = item.get("working_dir", "")
            if existing and Path(existing).expanduser().resolve() == working_path:
                return {"kb_id": item.get("kb_id", ""), "existed": True}

        output_path = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else Path("./output").resolve()
        )

        kb_id = f"kb_existing_{working_path.name}_{uuid.uuid4().hex[:8]}"
        meta = {
            "kb_id": kb_id,
            "file_name": f"[existing] {working_path.name}",
            "upload_path": "",
            "working_dir": str(working_path),
            "output_dir": str(output_path),
            "status": "ready",
            "stage": "ready",
            "progress": 100,
            "error": "",
            "parser": self.parser,
            "job_id": "",
            "source": "existing",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._write_meta(kb_id, meta)
        return {"kb_id": kb_id, "existed": False}

    def create_kb(self, uploaded_file: Any) -> Dict[str, str]:
        """Create isolated KB directories, save file(s), and start ingest job."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        kb_id = f"kb_{ts}_{uuid.uuid4().hex[:8]}"

        files: List[Any]
        if isinstance(uploaded_file, list):
            files = [f for f in uploaded_file if f is not None]
        else:
            files = [uploaded_file] if uploaded_file is not None else []
        if not files:
            raise ValueError("No files provided for KB creation")

        upload_paths = [self._save_uploaded_file(kb_id, f) for f in files]

        working_dir = self.rag_storage_root / kb_id
        output_dir = self.output_root / kb_id
        working_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        job_id = self._job_manager.create_ingest_job(
            kb_id=kb_id,
            file_path=str(upload_paths[0]),
            working_dir=str(working_dir),
            output_dir=str(output_dir),
            file_paths=[str(p) for p in upload_paths],
        )

        file_names = [p.name for p in upload_paths]
        if len(file_names) == 1:
            file_name_label = file_names[0]
        else:
            file_name_label = f"{len(file_names)} files: " + ", ".join(file_names[:3])
            if len(file_names) > 3:
                file_name_label += " ..."

        meta = {
            "kb_id": kb_id,
            "file_name": file_name_label,
            "upload_path": str(upload_paths[0]),
            "upload_paths": [str(p) for p in upload_paths],
            "file_count": len(upload_paths),
            "working_dir": str(working_dir),
            "output_dir": str(output_dir),
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "error": "",
            "parser": self.parser,
            "job_id": job_id,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._write_meta(kb_id, meta)

        return {"kb_id": kb_id, "job_id": job_id}

    def list_kbs(self) -> List[Dict[str, Any]]:
        """List known KB metadata sorted by creation time desc."""
        metas: List[Dict[str, Any]] = []
        for meta_file in self.meta_root.glob("*.json"):
            try:
                with meta_file.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    metas.append(payload)
            except Exception:
                continue

        metas.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return metas

    def get_kb_meta(self, kb_id: str) -> Dict[str, Any]:
        path = self._meta_path(kb_id)
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
                return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job state by id."""
        state = self._job_manager.get_job(job_id) or {
            "job_id": job_id,
            "status": "unknown",
            "stage": "unknown",
            "progress": 0,
            "events": [],
            "error": "",
        }

        kb_id = state.get("kb_id")
        if kb_id:
            self._on_job_update(state)
        return state

    def _get_or_create_rag(self, kb_id: str) -> Any:
        cached = self._rag_cache.get(kb_id)
        if cached is not None:
            return cached

        meta = self.get_kb_meta(kb_id)
        if not meta:
            raise ValueError(f"Unknown kb_id: {kb_id}")

        callback = None
        rag = self._build_rag(
            meta["working_dir"],
            meta["output_dir"],
            callback,
        )
        self._rag_cache[kb_id] = rag
        return rag

    def _record_timing(
        self,
        timings: Dict[str, float],
        stage: str,
        started_at: float,
    ) -> None:
        timings[stage] = round(time.perf_counter() - started_at, 4)

    def _log_query_timings(
        self,
        *,
        kb_id: str,
        mode: str,
        has_image: bool,
        planner_enabled: bool,
        timings: Dict[str, float],
    ) -> None:
        timing_text = ", ".join(f"{key}={value:.4f}s" for key, value in timings.items())
        logger.info(
            "UI query timing | kb_id=%s mode=%s has_image=%s planner_enabled=%s | %s",
            kb_id,
            mode,
            has_image,
            planner_enabled,
            timing_text,
        )

    def _run_planner_only(
        self,
        rag: Any,
        query_text: str,
        mode: str,
        timings: Dict[str, float],
    ) -> Dict[str, Any]:
        """Run planner and KG context fetch without generating a final answer."""
        planner_started_at = time.perf_counter()

        stage_started_at = time.perf_counter()
        plan = self._run_async(rag._build_plan_for_query(query_text))
        self._record_timing(timings, "planner_plan_llm", stage_started_at)

        kg_context = ""
        if "kg" in [str(t).lower() for t in plan.get("tool_plan", [])]:
            stage_started_at = time.perf_counter()
            kg_context = self._run_async(
                rag._get_kg_context_for_planner_query(query_text, mode=mode)
            )
            self._record_timing(timings, "planner_context_fetch", stage_started_at)

        self._record_timing(timings, "planner_total", planner_started_at)
        return {
            "plan": plan,
            "kg_context": kg_context,
            "kg_context_preview": kg_context[:2000],
        }

    def _build_planner_answer_prompt(
        self,
        query_text: str,
        planner_payload: Dict[str, Any],
    ) -> str:
        """Build final-answer prompt from a planner-only result."""
        plan = planner_payload.get("plan", {})
        kg_context = str(planner_payload.get("kg_context", ""))
        return f"""
你是农业病虫害问答助手。请基于给定证据回答问题。

用户问题:
{query_text}

规划结果:
{json.dumps(plan, ensure_ascii=False)}

本地KG检索证据:
{kg_context[:12000] if kg_context else "（无）"}

回答要求:
1. 优先使用本地KG证据。
2. 不要编造事实；证据不足时明确说明。
3. 以中文回答，结尾给出“证据来源简表”（KG）。
""".strip()

    def _augment_query_with_planner_context(
        self,
        query_text: str,
        planner_payload: Dict[str, Any],
    ) -> str:
        """Append planner output as guidance for the final multimodal answer."""
        plan = planner_payload.get("plan", {})
        kg_context = str(planner_payload.get("kg_context", ""))
        if not plan and not kg_context:
            return query_text

        return f"""
用户原始问题:
{query_text}

检索规划:
{json.dumps(plan, ensure_ascii=False)}

本地KG预检索证据:
{kg_context[:8000] if kg_context else "（无）"}

请先理解用户上传的图片，再结合上述规划和本地KG证据给出最终回答；证据不足时明确说明。
""".strip()

    def _fetch_query_context(self, rag: Any, question: str, mode: str) -> str:
        """Try only_need_context first; fallback to only_need_prompt."""
        try:
            from lightrag import QueryParam
        except ImportError as exc:
            raise RuntimeError(
                "`lightrag-hku` is required for querying from UI"
            ) from exc

        if rag.lightrag is None:
            self._run_async(rag._ensure_lightrag_initialized())

        assert rag.lightrag is not None
        config = getattr(rag, "config", None)
        query_kwargs: Dict[str, Any] = {}
        if config is not None:
            query_top_k = getattr(config, "query_top_k", None)
            if query_top_k is not None:
                query_kwargs["top_k"] = query_top_k
            query_enable_rerank = getattr(config, "query_enable_rerank", None)
            if query_enable_rerank is not None:
                query_kwargs["enable_rerank"] = bool(query_enable_rerank)

        try:
            context = self._run_async(
                rag.lightrag.aquery(
                    question,
                    param=QueryParam(
                        mode=mode,
                        only_need_context=True,
                        **query_kwargs,
                    ),
                )
            )
            return str(context)
        except Exception:
            prompt = self._run_async(
                rag.lightrag.aquery(
                    question,
                    param=QueryParam(
                        mode=mode,
                        only_need_prompt=True,
                        **query_kwargs,
                    ),
                )
            )
            return str(prompt)

    def _infer_modality_and_asset(self, chunk_content: str) -> tuple[str, Dict[str, Any]]:
        text = chunk_content or ""

        if "图片内容分析" in text or "Image Content Analysis" in text:
            img = extract_image_paths(text)
            return "image", {"image_path": img[0] if img else ""}

        if "表格分析" in text or "Table Analysis" in text:
            html_match = re.search(r"(?:结构|Structure)[:：](.*)", text, re.DOTALL)
            table_html = html_match.group(1).strip() if html_match else ""
            img = extract_image_paths(text)
            return "table", {
                "table_html": table_html,
                "image_path": img[0] if img else "",
            }

        if "数学公式分析" in text or "Mathematical Equation Analysis" in text:
            return "equation", {"equation_text": text[:1200]}

        return "text", {}

    def _build_citations(self, kb_id: str, context_raw: str) -> Dict[str, Any]:
        meta = self.get_kb_meta(kb_id)
        working_dir = Path(meta.get("working_dir", ""))

        chunks_db = self._load_json(working_dir / "kv_store_text_chunks.json")
        chunk_ids = extract_chunk_ids(context_raw)
        image_paths = extract_image_paths(context_raw)

        citations: List[Dict[str, Any]] = []

        for chunk_id in chunk_ids:
            chunk_data = chunks_db.get(chunk_id, {})
            content = str(chunk_data.get("content", ""))
            modality, asset_ref = self._infer_modality_and_asset(content)
            snippet = re.sub(r"\s+", " ", content).strip()[:320]

            citation = Citation(
                chunk_id=chunk_id,
                file_path=str(chunk_data.get("file_path", "")),
                page_idx=int(chunk_data.get("page_idx", -1)),
                snippet=snippet,
                modality=modality,
                source_type="retrieved_chunk",
                asset_ref=asset_ref,
            )
            citations.append(asdict(citation))

        if not citations and image_paths:
            # Context may not expose chunk ids in some retrieval modes.
            for path in image_paths[:3]:
                citations.append(
                    asdict(
                        Citation(
                            chunk_id="",
                            file_path="",
                            page_idx=-1,
                            snippet=f"Image Path: {path}",
                            modality="image",
                            source_type="context_path",
                            asset_ref={"image_path": path},
                        )
                    )
                )

        return {
            "citations": citations,
            "chunk_ids": [c["chunk_id"] for c in citations if c.get("chunk_id")],
        }

    def _read_graph_payload(self, kb_id: str) -> Dict[str, Any]:
        meta = self.get_kb_meta(kb_id)
        working_dir = Path(meta.get("working_dir", ""))
        graphml_path = working_dir / "graph_chunk_entity_relation.graphml"

        if not graphml_path.exists():
            return {
                "nodes": [],
                "links": [],
                "highlight_node_ids": [],
                "highlight_edge_ids": [],
                "error": f"GraphML not found: {graphml_path}",
            }

        tree = ET.parse(graphml_path)
        root = tree.getroot()
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}

        key_map: Dict[str, str] = {}
        for key in root.findall("g:key", ns):
            key_id = key.attrib.get("id", "")
            attr_name = key.attrib.get("attr.name", "")
            if key_id and attr_name:
                key_map[key_id] = attr_name

        graph = root.find("g:graph", ns)
        if graph is None:
            return {
                "nodes": [],
                "links": [],
                "highlight_node_ids": [],
                "highlight_edge_ids": [],
                "error": "Invalid GraphML: missing graph element",
            }

        nodes: List[Dict[str, Any]] = []
        links: List[Dict[str, Any]] = []

        for node in graph.findall("g:node", ns):
            node_id = node.attrib.get("id", "")
            attrs: Dict[str, str] = {}
            for data in node.findall("g:data", ns):
                key = data.attrib.get("key", "")
                attrs[key_map.get(key, key)] = (data.text or "").strip()

            entity_type = attrs.get("entity_type") or "Unknown"
            source_ids = split_source_ids(attrs.get("source_id", ""))
            node_payload = {
                "id": node_id,
                "name": attrs.get("entity_id") or node_id,
                "category": entity_type,
                "value": attrs.get("description", ""),
                "source_ids": source_ids,
                "symbolSize": 22 if attrs.get("content_modality") else 16,
                "itemStyle": {
                    "opacity": 0.92,
                },
            }
            nodes.append(node_payload)

        for edge in graph.findall("g:edge", ns):
            edge_id = edge.attrib.get("id", "")
            source = edge.attrib.get("source", "")
            target = edge.attrib.get("target", "")
            attrs: Dict[str, str] = {}
            for data in edge.findall("g:data", ns):
                key = data.attrib.get("key", "")
                attrs[key_map.get(key, key)] = (data.text or "").strip()

            weight = attrs.get("weight", "1")
            try:
                width = max(1.0, min(6.0, float(weight)))
            except Exception:
                width = 1.0

            links.append(
                {
                    "id": edge_id,
                    "source": source,
                    "target": target,
                    "name": attrs.get("relation_type") or attrs.get("keywords") or "",
                    "value": attrs.get("description", ""),
                    "source_ids": split_source_ids(attrs.get("source_id", "")),
                    "lineStyle": {"width": width, "opacity": 0.72},
                }
            )

        return {
            "nodes": nodes,
            "links": links,
            "highlight_node_ids": [],
            "highlight_edge_ids": [],
        }

    def get_graph(
        self,
        kb_id: str,
        focus_chunk_ids: Optional[List[str]] = None,
        full: bool = False,
    ) -> Dict[str, Any]:
        """Get ECharts graph payload, optionally filtered by chunk focus."""
        graph = self._read_graph_payload(kb_id)
        if graph.get("error"):
            return graph

        selected = select_subgraph_by_chunks(
            nodes=graph["nodes"],
            links=graph["links"],
            chunk_ids=focus_chunk_ids or [],
            full=full,
        )
        return selected

    def query(
        self,
        kb_id: str,
        question: str,
        mode: str = "hybrid",
        debug: bool = False,
        planner_enabled: bool = False,
    ) -> Dict[str, Any]:
        """Run query and return structured answer + citations + graph focus."""
        total_started_at = time.perf_counter()
        timings: Dict[str, float] = {}

        stage_started_at = time.perf_counter()
        rag = self._get_or_create_rag(kb_id)
        self._record_timing(timings, "load_rag", stage_started_at)

        # Ensure LightRAG backend is initialized before calling aquery().
        # QueryMixin.aquery() raises immediately when self.lightrag is None.
        stage_started_at = time.perf_counter()
        init_result = self._run_async(rag._ensure_lightrag_initialized())
        if isinstance(init_result, dict) and not init_result.get("success", False):
            raise RuntimeError(
                init_result.get("error")
                or "Failed to initialize LightRAG for querying"
            )
        self._record_timing(timings, "initialize_lightrag", stage_started_at)

        planner_payload: Dict[str, Any] = {}
        if planner_enabled:
            planner_payload = self._run_planner_only(rag, question, mode, timings)
            final_prompt = self._build_planner_answer_prompt(question, planner_payload)
            stage_started_at = time.perf_counter()
            answer = self._run_async(
                rag._call_text_llm(
                    final_prompt,
                    system_prompt="你是严谨的农业病虫害专家助手，强调可验证和低幻觉。",
                )
            )
        else:
            stage_started_at = time.perf_counter()
            answer = self._run_async(rag.aquery(question, mode=mode))
        self._record_timing(timings, "answer_generation", stage_started_at)

        context_raw = ""
        citations: List[Dict[str, Any]] = []
        graph_focus = {"node_ids": [], "edge_ids": [], "chunk_ids": []}

        try:
            stage_started_at = time.perf_counter()
            context_raw = self._fetch_query_context(rag, question, mode)
            self._record_timing(timings, "context_fetch", stage_started_at)

            stage_started_at = time.perf_counter()
            citation_bundle = self._build_citations(kb_id, context_raw)
            citations = citation_bundle["citations"]
            chunk_ids = citation_bundle["chunk_ids"]
            self._record_timing(timings, "citation_build", stage_started_at)

            stage_started_at = time.perf_counter()
            graph_payload = self.get_graph(kb_id, focus_chunk_ids=chunk_ids, full=False)
            graph_focus = {
                "node_ids": graph_payload.get("highlight_node_ids", []),
                "edge_ids": graph_payload.get("highlight_edge_ids", []),
                "chunk_ids": chunk_ids,
            }
            self._record_timing(timings, "graph_focus", stage_started_at)
        except Exception:
            # Keep answer available even if evidence extraction fails.
            citations = []
            graph_focus = {"node_ids": [], "edge_ids": [], "chunk_ids": []}

        self._record_timing(timings, "backend_total", total_started_at)
        self._log_query_timings(
            kb_id=kb_id,
            mode=mode,
            has_image=False,
            planner_enabled=planner_enabled,
            timings=timings,
        )

        response = QueryResponse(
            answer=str(answer),
            citations=citations,
            graph_focus=graph_focus,
            debug={
                **{
                    key: value
                    for key, value in planner_payload.items()
                    if key != "kg_context"
                },
                **({"context_raw": context_raw} if debug else {}),
            },
            timings=timings,
        )
        return asdict(response)

    def query_with_image(
        self,
        kb_id: str,
        question: str,
        image_file: Any,
        mode: str = "hybrid",
        debug: bool = False,
        planner_enabled: bool = False,
    ) -> Dict[str, Any]:
        """Run multimodal query with user-provided image + optional text question."""
        total_started_at = time.perf_counter()
        timings: Dict[str, float] = {}

        stage_started_at = time.perf_counter()
        rag = self._get_or_create_rag(kb_id)
        self._record_timing(timings, "load_rag", stage_started_at)

        query_text = (question or "").strip()
        if not query_text:
            query_text = "请结合知识库分析这张图片，并给出关键判断依据。"

        stage_started_at = time.perf_counter()
        image_path = self._save_query_image_input(kb_id, image_file)
        multimodal_content = [{"type": "image", "img_path": str(image_path)}]
        self._record_timing(timings, "save_query_image", stage_started_at)

        stage_started_at = time.perf_counter()
        init_result = self._run_async(rag._ensure_lightrag_initialized())
        if isinstance(init_result, dict) and not init_result.get("success", False):
            raise RuntimeError(
                init_result.get("error")
                or "Failed to initialize LightRAG for querying"
            )
        self._record_timing(timings, "initialize_lightrag", stage_started_at)

        planner_payload: Dict[str, Any] = {}
        if planner_enabled:
            planner_payload = self._run_planner_only(rag, query_text, mode, timings)

        answer_query_text = (
            self._augment_query_with_planner_context(query_text, planner_payload)
            if planner_payload
            else query_text
        )
        stage_started_at = time.perf_counter()
        answer = self._run_async(
            rag.aquery_with_multimodal(
                query=answer_query_text,
                multimodal_content=multimodal_content,
                mode=mode,
            )
        )
        self._record_timing(timings, "answer_generation", stage_started_at)

        context_raw = ""
        citations: List[Dict[str, Any]] = []
        graph_focus = {"node_ids": [], "edge_ids": [], "chunk_ids": []}
        try:
            stage_started_at = time.perf_counter()
            context_raw = self._fetch_query_context(rag, query_text, mode)
            self._record_timing(timings, "context_fetch", stage_started_at)

            stage_started_at = time.perf_counter()
            citation_bundle = self._build_citations(kb_id, context_raw)
            citations = citation_bundle["citations"]
            chunk_ids = citation_bundle["chunk_ids"]
            self._record_timing(timings, "citation_build", stage_started_at)

            stage_started_at = time.perf_counter()
            graph_payload = self.get_graph(kb_id, focus_chunk_ids=chunk_ids, full=False)
            graph_focus = {
                "node_ids": graph_payload.get("highlight_node_ids", []),
                "edge_ids": graph_payload.get("highlight_edge_ids", []),
                "chunk_ids": chunk_ids,
            }
            self._record_timing(timings, "graph_focus", stage_started_at)
        except Exception:
            citations = []
            graph_focus = {"node_ids": [], "edge_ids": [], "chunk_ids": []}

        debug_payload: Dict[str, Any] = {}
        if debug:
            debug_payload["context_raw"] = context_raw
            debug_payload["query_image_path"] = str(image_path)
        if planner_payload:
            debug_payload.update(
                {
                    key: value
                    for key, value in planner_payload.items()
                    if key != "kg_context"
                }
            )

        self._record_timing(timings, "backend_total", total_started_at)
        self._log_query_timings(
            kb_id=kb_id,
            mode=mode,
            has_image=True,
            planner_enabled=planner_enabled,
            timings=timings,
        )

        response = QueryResponse(
            answer=str(answer),
            citations=citations,
            graph_focus=graph_focus,
            debug=debug_payload,
            timings=timings,
        )
        return asdict(response)
