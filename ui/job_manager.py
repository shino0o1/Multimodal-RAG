"""Background job manager for UI ingestion tasks."""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    from raganything.callbacks import ProcessingCallback as _BaseCallback
except Exception:  # pragma: no cover - enables utility-only imports without lightrag
    class _BaseCallback:  # type: ignore[too-many-ancestors]
        pass


@dataclass
class JobRecord:
    """In-memory job record for ingestion tasks."""

    job_id: str
    kb_id: str
    file_path: str
    file_paths: List[str]
    working_dir: str
    output_dir: str
    status: str = "queued"
    stage: str = "queued"
    progress: int = 0
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    events: List[Dict[str, Any]] = field(default_factory=list)


class _JobProgressCallback(_BaseCallback):
    """Bridge RAGAnything callback events into job state updates."""

    def __init__(self, manager: "JobManager", job_id: str) -> None:
        self._manager = manager
        self._job_id = job_id

    def on_parse_start(self, file_path: str, parser: str = "", **kwargs: Any) -> None:
        self._manager.update_job(
            self._job_id,
            stage="parse",
            progress=max(self._manager.get_job_progress(self._job_id), 10),
            event={
                "event": "on_parse_start",
                "file_path": file_path,
                "parser": parser,
            },
        )

    def on_parse_complete(
        self,
        file_path: str,
        content_blocks: int = 0,
        doc_id: str = "",
        duration_seconds: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self._manager.update_job(
            self._job_id,
            stage="parse",
            progress=max(self._manager.get_job_progress(self._job_id), 30),
            event={
                "event": "on_parse_complete",
                "file_path": file_path,
                "content_blocks": content_blocks,
                "doc_id": doc_id,
                "duration_seconds": duration_seconds,
            },
        )

    def on_text_insert_start(
        self, file_path: str, text_length: int = 0, **kwargs: Any
    ) -> None:
        self._manager.update_job(
            self._job_id,
            stage="text_insert",
            progress=max(self._manager.get_job_progress(self._job_id), 40),
            event={
                "event": "on_text_insert_start",
                "file_path": file_path,
                "text_length": text_length,
            },
        )

    def on_text_insert_complete(
        self, file_path: str, duration_seconds: float = 0.0, **kwargs: Any
    ) -> None:
        self._manager.update_job(
            self._job_id,
            stage="text_insert",
            progress=max(self._manager.get_job_progress(self._job_id), 55),
            event={
                "event": "on_text_insert_complete",
                "file_path": file_path,
                "duration_seconds": duration_seconds,
            },
        )

    def on_multimodal_start(
        self, file_path: str, item_count: int = 0, **kwargs: Any
    ) -> None:
        self._manager.update_job(
            self._job_id,
            stage="multimodal",
            progress=max(self._manager.get_job_progress(self._job_id), 65),
            event={
                "event": "on_multimodal_start",
                "file_path": file_path,
                "item_count": item_count,
            },
        )

    def on_multimodal_complete(
        self,
        file_path: str,
        processed_count: int = 0,
        duration_seconds: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self._manager.update_job(
            self._job_id,
            stage="multimodal",
            progress=max(self._manager.get_job_progress(self._job_id), 90),
            event={
                "event": "on_multimodal_complete",
                "file_path": file_path,
                "processed_count": processed_count,
                "duration_seconds": duration_seconds,
            },
        )

    def on_document_complete(
        self,
        file_path: str,
        doc_id: str = "",
        duration_seconds: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self._manager.update_job(
            self._job_id,
            stage="document_complete",
            progress=max(self._manager.get_job_progress(self._job_id), 90),
            event={
                "event": "on_document_complete",
                "file_path": file_path,
                "doc_id": doc_id,
                "duration_seconds": duration_seconds,
            },
        )

    def on_document_error(
        self,
        file_path: str,
        error: BaseException | str = "",
        stage: str = "",
        **kwargs: Any,
    ) -> None:
        self._manager.update_job(
            self._job_id,
            status="failed",
            stage=stage or "error",
            error=str(error),
            event={
                "event": "on_document_error",
                "file_path": file_path,
                "stage": stage,
                "error": str(error),
            },
        )


class JobManager:
    """Threaded ingestion job manager for Streamlit UI."""

    def __init__(
        self,
        rag_builder: Callable[[str, str, Any], Any],
        on_job_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._rag_builder = rag_builder
        self._on_job_update = on_job_update
        self._jobs: Dict[str, JobRecord] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.RLock()

    def create_ingest_job(
        self,
        kb_id: str,
        file_path: str,
        working_dir: str,
        output_dir: str,
        file_paths: Optional[List[str]] = None,
    ) -> str:
        """Create and start a new ingestion job."""
        job_id = f"job-{uuid.uuid4().hex[:10]}"
        normalized_paths = [p for p in (file_paths or [file_path]) if p]
        record = JobRecord(
            job_id=job_id,
            kb_id=kb_id,
            file_path=normalized_paths[0] if normalized_paths else "",
            file_paths=normalized_paths,
            working_dir=working_dir,
            output_dir=output_dir,
        )

        with self._lock:
            self._jobs[job_id] = record

        thread = threading.Thread(
            target=self._run_ingest_job,
            args=(job_id,),
            daemon=True,
            name=f"rag-ui-{job_id}",
        )
        with self._lock:
            self._threads[job_id] = thread
        thread.start()
        return job_id

    def _run_ingest_job(self, job_id: str) -> None:
        rag = None
        try:
            job = self.get_job(job_id)
            if not job:
                return

            self.update_job(
                job_id,
                status="running",
                stage="initializing",
                progress=5,
                event={
                    "event": "job_started",
                    "file_count": len(job.get("file_paths") or []),
                },
            )

            callback = _JobProgressCallback(self, job_id)
            rag = self._rag_builder(
                job["working_dir"],
                job["output_dir"],
                callback,
            )

            file_paths = [p for p in (job.get("file_paths") or []) if p]
            if not file_paths and job.get("file_path"):
                file_paths = [job["file_path"]]
            if not file_paths:
                raise ValueError("No input file paths for ingestion job")

            total_files = len(file_paths)
            for idx, file_path in enumerate(file_paths, start=1):
                start_progress = 5 + int(((idx - 1) / total_files) * 90)
                self.update_job(
                    job_id,
                    stage=f"ingesting_{idx}/{total_files}",
                    progress=start_progress,
                    event={
                        "event": "ingest_file_started",
                        "file_index": idx,
                        "total_files": total_files,
                        "file_path": file_path,
                    },
                )

                asyncio.run(
                    rag.process_document_complete(
                        file_path=file_path,
                        output_dir=job["output_dir"],
                        parse_method="auto",
                    )
                )

                done_progress = 5 + int((idx / total_files) * 90)
                self.update_job(
                    job_id,
                    stage=f"ingesting_{idx}/{total_files}",
                    progress=done_progress,
                    event={
                        "event": "ingest_file_completed",
                        "file_index": idx,
                        "total_files": total_files,
                        "file_path": file_path,
                    },
                )

            latest = self.get_job(job_id)
            if latest and latest["status"] != "completed":
                self.update_job(
                    job_id,
                    status="completed",
                    stage="completed",
                    progress=100,
                    event={"event": "job_finished"},
                )

        except Exception as exc:
            self.update_job(
                job_id,
                status="failed",
                stage="error",
                error=str(exc),
                event={
                    "event": "job_exception",
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=20),
                },
            )
        finally:
            if rag is not None:
                try:
                    asyncio.run(rag.finalize_storages())
                except Exception:
                    pass

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        progress: Optional[int] = None,
        error: Optional[str] = None,
        event: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update a job state and append optional event."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return

            if status is not None:
                record.status = status
            if stage is not None:
                record.stage = stage
            if progress is not None:
                record.progress = max(0, min(100, int(progress)))
            if error is not None:
                record.error = error

            record.updated_at = time.time()
            if event is not None:
                record.events.append(
                    {
                        "timestamp": record.updated_at,
                        **event,
                    }
                )

            payload = self._serialize_record(record)

        if self._on_job_update is not None:
            try:
                self._on_job_update(payload)
            except Exception:
                pass

    def get_job_progress(self, job_id: str) -> int:
        """Get current progress for a job; returns 0 when absent."""
        with self._lock:
            record = self._jobs.get(job_id)
            return int(record.progress) if record else 0

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by id."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            return self._serialize_record(record)

    def _serialize_record(self, record: JobRecord) -> Dict[str, Any]:
        return {
            "job_id": record.job_id,
            "kb_id": record.kb_id,
            "file_path": record.file_path,
            "file_paths": list(record.file_paths),
            "working_dir": record.working_dir,
            "output_dir": record.output_dir,
            "status": record.status,
            "stage": record.stage,
            "progress": int(record.progress),
            "error": record.error,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "events": list(record.events),
        }
