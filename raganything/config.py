"""
Configuration classes for RAGAnything

Contains configuration dataclasses with environment variable support
"""

from dataclasses import dataclass, field
from typing import List
from lightrag.utils import get_env_value


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
            "KG_CORE_ENTITY_TYPES", "虫害,病害,作物,病原菌,药剂,生长期,生物分类"
        )
    )
    """Core entity types retained as main nodes."""

    kg_anchor_node_types: List[str] = field(
        default_factory=lambda: _get_env_list("KG_ANCHOR_NODE_TYPES", "部位,时间")
    )
    """Anchor node types retained for temporal/spatial linking."""

    kg_attribute_fields: List[str] = field(
        default_factory=lambda: _get_env_list(
            "KG_ATTRIBUTE_FIELDS", "形态特征,危害症状,发病诱因,发生时期,防治要点,生活习性,发生规律"
        )
    )
    """Attribute fields attached to entities instead of fragment nodes."""

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
    """Merge threshold for semantic deduplication (reserved for iterative tuning)."""

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
