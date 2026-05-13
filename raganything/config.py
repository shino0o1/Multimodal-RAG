"""
Configuration classes for RAGAnything

Contains configuration dataclasses with environment variable support
"""

from dataclasses import dataclass, field
from typing import List
from lightrag.utils import get_env_value
from raganything.kg_quality import (
    DEFAULT_CORE_ENTITY_TYPES,
    DEFAULT_ANCHOR_NODE_TYPES,
    DEFAULT_ATTRIBUTE_FIELDS,
)


def _get_env_list(name: str, default: str) -> List[str]:
    raw = get_env_value(name, default, str)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class RAGAnythingConfig:
    """Configuration class for RAGAnything with environment variable support"""

    # Directory Configuration
    # ---
    working_dir: str = field(default=get_env_value("WORKING_DIR", "./rag_storage", str))
    """Directory where RAG storage and cache files are stored."""

    # Parser Configuration
    # ---
    parse_method: str = field(default=get_env_value("PARSE_METHOD", "auto", str))
    """Default parsing method for document parsing: 'auto', 'ocr', or 'txt'."""

    parser_output_dir: str = field(default=get_env_value("OUTPUT_DIR", "./output", str))
    """Default output directory for parsed content."""

    parser: str = field(default=get_env_value("PARSER", "mineru", str))
    """Parser selection: 'mineru', 'docling', or 'paddleocr'."""

    display_content_stats: bool = field(
        default=get_env_value("DISPLAY_CONTENT_STATS", True, bool)
    )
    """Whether to display content statistics during parsing."""

    # Multimodal Processing Configuration
    # ---
    enable_image_processing: bool = field(
        default=get_env_value("ENABLE_IMAGE_PROCESSING", True, bool)
    )
    """Enable image content processing."""

    enable_table_processing: bool = field(
        default=get_env_value("ENABLE_TABLE_PROCESSING", True, bool)
    )
    """Enable table content processing."""

    enable_equation_processing: bool = field(
        default=get_env_value("ENABLE_EQUATION_PROCESSING", True, bool)
    )
    """Enable equation content processing."""

    multimodal_desc_cache_enabled: bool = field(
        default=get_env_value("MULTIMODAL_DESC_CACHE_ENABLED", True, bool)
    )
    """Enable multimodal description cache keyed by doc_id + item hash."""

    multimodal_hard_skip_enabled: bool = field(
        default=get_env_value("MULTIMODAL_HARD_SKIP_ENABLED", True, bool)
    )
    """Enable hard-skip when multimodal completion state matches current item signature."""

    # Batch Processing Configuration
    # ---
    max_concurrent_files: int = field(
        default=get_env_value("MAX_CONCURRENT_FILES", 1, int)
    )
    """Maximum number of files to process concurrently."""

    supported_file_extensions: List[str] = field(
        default_factory=lambda: get_env_value(
            "SUPPORTED_FILE_EXTENSIONS",
            ".pdf,.jpg,.jpeg,.png,.bmp,.tiff,.tif,.gif,.webp,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.txt,.md",
            str,
        ).split(",")
    )
    """List of supported file extensions for batch processing."""

    recursive_folder_processing: bool = field(
        default=get_env_value("RECURSIVE_FOLDER_PROCESSING", True, bool)
    )
    """Whether to recursively process subfolders in batch mode."""

    # Context Extraction Configuration
    # ---
    context_window: int = field(default=get_env_value("CONTEXT_WINDOW", 1, int))
    """Number of pages/chunks to include before and after current item for context."""

    context_mode: str = field(default=get_env_value("CONTEXT_MODE", "page", str))
    """Context extraction mode: 'page' for page-based, 'chunk' for chunk-based."""

    max_context_tokens: int = field(
        default=get_env_value("MAX_CONTEXT_TOKENS", 2000, int)
    )
    """Maximum number of tokens in extracted context."""

    include_headers: bool = field(default=get_env_value("INCLUDE_HEADERS", True, bool))
    """Whether to include document headers and titles in context."""

    include_captions: bool = field(
        default=get_env_value("INCLUDE_CAPTIONS", True, bool)
    )
    """Whether to include image/table captions in context."""

    context_filter_content_types: List[str] = field(
        default_factory=lambda: get_env_value(
            "CONTEXT_FILTER_CONTENT_TYPES", "text", str
        ).split(",")
    )
    """Content types to include in context extraction (e.g., 'text', 'image', 'table')."""

    content_format: str = field(default=get_env_value("CONTENT_FORMAT", "minerU", str))
    """Default content format for context extraction when processing documents."""

    # Query Runtime Configuration
    # ---
    query_mode: str = field(default=get_env_value("QUERY_MODE", "hybrid", str))
    """Default query mode: 'local', 'global', 'hybrid', 'naive', 'mix', or 'bypass'."""

    query_top_k: int = field(default=get_env_value("QUERY_TOP_K", 10, int))
    """Default top_k used by query retrieval."""

    query_enable_rerank: bool = field(
        default=get_env_value("QUERY_ENABLE_RERANK", False, bool)
    )
    """Whether query-time rerank is enabled by default."""

    query_vlm_enhanced: bool = field(
        default=get_env_value("QUERY_VLM_ENHANCED", False, bool)
    )
    """Whether VLM-enhanced query branch is enabled by default."""

    # Model Routing Configuration
    # ---
    model_answer: str = field(
        default=get_env_value("RAG_MODEL_ANSWER", "gemini-2.5-flash", str)
    )
    """Model for final answer generation."""

    model_planner: str = field(default=get_env_value("RAG_MODEL_PLANNER", "gemini-2.5-flash", str))
    """Model for planning stage; empty means fallback to model_answer."""

    model_judge: str = field(
        default=get_env_value("RAG_MODEL_JUDGE", "gemini-3.1-pro-preview", str)
    )
    """Model for evaluation dataset judging."""

    model_vision: str = field(default=get_env_value("RAG_MODEL_VISION", "gemini-2.5-flash", str))
    """Model for vision reasoning (image input); empty means fallback to model_answer."""
    # 如果不配置RAG_MODEL_IMAGE_DESCRIPTION, 会复用RAG_MODEL_VISION
    model_image_description: str = field(
        default=get_env_value("RAG_MODEL_IMAGE_DESCRIPTION", "", str)
    )
    """Model for image description stage; empty means fallback to model_vision."""

    model_embedding: str = field(
        default=get_env_value("RAG_MODEL_EMBEDDING", "text-embedding-3-large", str)
    )
    """Embedding model for vector retrieval."""

    embedding_dim: int = field(default=get_env_value("EMBEDDING_DIM", 3072, int))
    """Embedding vector dimension."""

    rerank_model: str = field(default=get_env_value("RAG_MODEL_RERANK", "", str))
    """Rerank model identifier. Empty means no rerank model configured."""

    # OpenAI-Compatible Routing & Reasoning Configuration
    # ---
    llm_api_key: str = field(default=get_env_value("RAG_LLM_API_KEY", "sk-MwcAPesgu8ol4F0ePPNP0hkGiseYaNEbfoLv4phN03ldl3AV", str))
    """API key for OpenAI-compatible chat completion calls."""

    llm_base_url: str = field(default=get_env_value("RAG_LLM_BASE_URL", "https://yunwu.ai/v1", str))
    """Base URL for OpenAI-compatible API endpoint, e.g. https://yunwu.ai/v1."""

    reasoning_effort_default: str = field(
        default=get_env_value("RAG_REASONING_EFFORT_DEFAULT", "medium", str)
    )
    """Default reasoning effort for reasoning models. Allowed: low/medium/high."""

    reasoning_effort_answer: str = field(
        default=get_env_value("RAG_REASONING_EFFORT_ANSWER", "", str)
    )
    """Reasoning effort for answer model stage; falls back to default when empty."""

    reasoning_effort_planner: str = field(
        default=get_env_value("RAG_REASONING_EFFORT_PLANNER", "", str)
    )
    """Reasoning effort for planner model stage; falls back to default when empty."""

    reasoning_effort_vision: str = field(
        default=get_env_value("RAG_REASONING_EFFORT_VISION", "", str)
    )
    """Reasoning effort for vision model stage; falls back to default when empty."""

    reasoning_effort_image_description: str = field(
        default=get_env_value("RAG_REASONING_EFFORT_IMAGE_DESCRIPTION", "", str)
    )
    """Reasoning effort for image-description stage; falls back to default when empty."""

    # Path Handling Configuration
    # ---
    use_full_path: bool = field(default=get_env_value("USE_FULL_PATH", False, bool))
    """Whether to use full file path (True) or just basename (False) for file references in LightRAG."""

    # KG Quality Configuration
    # ---
    kg_quality_enabled: bool = field(
        default=get_env_value("KG_QUALITY_ENABLED", True, bool)
    )
    """Enable knowledge graph quality governance."""

    kg_canonical_language: str = field(
        default=get_env_value("KG_CANONICAL_LANGUAGE", "zh", str)
    )
    """Canonical language for entity naming (e.g. 'zh')."""

    kg_relation_schema: str = field(
        default=get_env_value("KG_RELATION_SCHEMA", "fixed", str)
    )
    """Relation schema policy. 'fixed' enables controlled relation enums."""

    kg_extraction_mode: str = field(
        default=get_env_value("KG_EXTRACTION_MODE", "ppe", str)
    )
    """KG extraction mode: 'ppe' (default) or 'legacy'."""

    kg_core_entity_types: List[str] = field(
        default_factory=lambda: _get_env_list(
            "KG_CORE_ENTITY_TYPES", ",".join(DEFAULT_CORE_ENTITY_TYPES)
        )
    )
    """Core entity types retained as main nodes."""

    kg_anchor_node_types: List[str] = field(
        default_factory=lambda: _get_env_list(
            "KG_ANCHOR_NODE_TYPES", ",".join(DEFAULT_ANCHOR_NODE_TYPES)
        )
    )
    """Anchor node types retained for temporal/spatial/plant-part linking."""

    kg_attribute_fields: List[str] = field(
        default_factory=lambda: _get_env_list(
            "KG_ATTRIBUTE_FIELDS", ",".join(DEFAULT_ATTRIBUTE_FIELDS)
        )
    )
    """Attribute fields attached to entities instead of fragment nodes."""

    kg_attribute_host_types: List[str] = field(
        default_factory=lambda: _get_env_list(
            "KG_ATTRIBUTE_HOST_TYPES", "虫害,病害,病原菌"
        )
    )
    """Entity types that are allowed to host projected attribute fields."""

    kg_noise_drop_types: List[str] = field(
        default_factory=lambda: _get_env_list(
            "KG_NOISE_DROP_TYPES", "header,page_number"
        )
    )
    """Content block types dropped at source before extraction."""

    kg_noise_drop_patterns: List[str] = field(
        default_factory=lambda: _get_env_list(
            "KG_NOISE_DROP_PATTERNS",
            r"^\s*None\s*$,^\s*\d+\s*$,蔬菜病虫害诊断[与于]防治原色图谱,(?:QR|qr|二维码|QR码|扫码),^(?:[A-Za-z]:)?[/\\].+\.(?:jpg|jpeg|png|bmp|gif|webp|tiff?)\s*$",
        )
    )
    """Regex patterns for dropping noisy OCR/parsing content."""

    kg_description_policy: str = field(
        default=get_env_value("KG_DESCRIPTION_POLICY", "multimodal_only", str)
    )
    """Description policy. 'multimodal_only' keeps descriptions only on multimodal nodes."""

    kg_multimodal_min_desc_chars: int = field(
        default=get_env_value("KG_MULTIMODAL_MIN_DESC_CHARS", 30, int)
    )
    """Minimum multimodal description length before one retry is attempted."""

    kg_drop_empty_multimodal: bool = field(
        default=get_env_value("KG_DROP_EMPTY_MULTIMODAL", True, bool)
    )
    """Drop multimodal nodes/items whose detailed description and summary are both empty."""

    kg_ontology_profile: str = field(
        default=get_env_value("KG_ONTOLOGY_PROFILE", "cruciferous_pest_disease", str)
    )
    """Domain ontology profile, e.g. cruciferous_pest_disease."""

    kg_enforce_ontology: bool = field(
        default=get_env_value("KG_ENFORCE_ONTOLOGY", True, bool)
    )
    """Enable ontology domain-range validation for relation types."""

    kg_merge_threshold: float = field(
        default=get_env_value("KG_MERGE_THRESHOLD", 0.85, float)
    )
    """Gray-zone lower threshold for semantic deduplication (vector strict threshold is derived internally)."""

    kg_llm_semantic_merge_enabled: bool = field(
        default=get_env_value("KG_LLM_SEMANTIC_MERGE_ENABLED", False, bool)
    )
    """Enable LLM arbitration only for vector gray-zone candidates during GraphML cleanup."""

    kg_llm_semantic_merge_types: List[str] = field(
        default_factory=lambda: _get_env_list(
            "KG_LLM_SEMANTIC_MERGE_TYPES",
            "作物,生物分类,病原菌,药剂,病害,虫害",
        )
    )
    """Entity types eligible for LLM-assisted semantic deduplication."""

    kg_llm_semantic_name_sim_threshold: float = field(
        default=get_env_value("KG_LLM_SEMANTIC_NAME_SIM_THRESHOLD", 0.75, float)
    )
    """Name similarity threshold used for generating semantic dedup candidate groups."""

    kg_llm_semantic_merge_min_confidence: float = field(
        default=get_env_value("KG_LLM_SEMANTIC_MERGE_MIN_CONFIDENCE", 0.90, float)
    )
    """Minimum LLM confidence required to apply a dedup merge decision."""

    kg_llm_semantic_merge_max_group_size: int = field(
        default=get_env_value("KG_LLM_SEMANTIC_MERGE_MAX_GROUP_SIZE", 12, int)
    )
    """Maximum candidate group size sent to LLM for one dedup decision."""

    kg_llm_semantic_merge_max_groups: int = field(
        default=get_env_value("KG_LLM_SEMANTIC_MERGE_MAX_GROUPS", 80, int)
    )
    """Maximum number of LLM dedup groups processed per cleanup run."""

    kg_llm_timeout_seconds: int = field(
        default=get_env_value("KG_LLM_TIMEOUT_SECONDS", 90, int)
    )
    """Timeout in seconds for one LLM dedup call."""

    def __post_init__(self):
        """Post-initialization setup for backward compatibility"""
        # Support legacy environment variable names for backward compatibility
        legacy_parse_method = get_env_value("MINERU_PARSE_METHOD", None, str)
        if legacy_parse_method and not get_env_value("PARSE_METHOD", None, str):
            self.parse_method = legacy_parse_method
            import warnings

            warnings.warn(
                "MINERU_PARSE_METHOD is deprecated. Use PARSE_METHOD instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Keep model routing predictable when optional stages are not configured.
        if not self.model_planner:
            self.model_planner = self.model_answer
        if not self.model_judge:
            self.model_judge = "gemini-3.1-pro-preview"
        if not self.model_vision:
            self.model_vision = self.model_answer
        if not self.model_image_description:
            self.model_image_description = self.model_vision

        self.reasoning_effort_default = self._normalize_reasoning_effort(
            self.reasoning_effort_default
        )
        self.reasoning_effort_answer = self._normalize_reasoning_effort(
            self.reasoning_effort_answer
        )
        self.reasoning_effort_planner = self._normalize_reasoning_effort(
            self.reasoning_effort_planner
        )
        self.reasoning_effort_vision = self._normalize_reasoning_effort(
            self.reasoning_effort_vision
        )
        self.reasoning_effort_image_description = self._normalize_reasoning_effort(
            self.reasoning_effort_image_description
        )

        if self.query_top_k <= 0:
            self.query_top_k = 10

    @staticmethod
    def _normalize_reasoning_effort(value: str) -> str:
        if not value:
            return ""
        normalized = value.strip().lower()
        return normalized if normalized in {"low", "medium", "high"} else ""

    def get_reasoning_effort(self, stage: str) -> str:
        stage_to_effort = {
            "answer": self.reasoning_effort_answer,
            "planner": self.reasoning_effort_planner,
            "vision": self.reasoning_effort_vision,
            "image_description": self.reasoning_effort_image_description,
        }
        return stage_to_effort.get(stage, "") or self.reasoning_effort_default

    @property
    def mineru_parse_method(self) -> str:
        """
        Backward compatibility property for old code.

        .. deprecated::
           Use `parse_method` instead. This property will be removed in a future version.
        """
        import warnings

        warnings.warn(
            "mineru_parse_method is deprecated. Use parse_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.parse_method

    @mineru_parse_method.setter
    def mineru_parse_method(self, value: str):
        """Setter for backward compatibility"""
        import warnings

        warnings.warn(
            "mineru_parse_method is deprecated. Use parse_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.parse_method = value
