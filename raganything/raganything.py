"""
Complete document parsing + multimodal content insertion Pipeline

This script integrates:
1. Document parsing (using configurable parsers)
2. Pure text content LightRAG insertion
3. Specialized processing for multimodal content (using different processors)
"""

import os
from typing import Dict, Any, Optional, Callable
import sys
import asyncio
import atexit
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Add project root directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file BEFORE importing LightRAG
# This is critical for TIKTOKEN_CACHE_DIR to work properly in offline environments
# The OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

from lightrag import LightRAG
from lightrag.utils import logger

# Import configuration and modules
from raganything.config import RAGAnythingConfig
from raganything.query import QueryMixin
from raganything.processor import ProcessorMixin
from raganything.batch import BatchMixin
from raganything.utils import get_processor_supports
from raganything.parser import MineruParser, SUPPORTED_PARSERS, get_parser
from raganything.callbacks import CallbackManager
from raganything.kg_quality import KGQualityManager

# Import specialized processors
from raganything.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    EquationModalProcessor,
    GenericModalProcessor,
    ContextExtractor,
    ContextConfig,
)


@dataclass
class RAGAnything(QueryMixin, ProcessorMixin, BatchMixin):
    """Multimodal Document Processing Pipeline - Complete document parsing and insertion pipeline"""

    # Core Components
    # ---
    lightrag: Optional[LightRAG] = field(default=None)
    """Optional pre-initialized LightRAG instance."""

    llm_model_func: Optional[Callable] = field(default=None)
    """LLM model function for text analysis."""

    vision_model_func: Optional[Callable] = field(default=None)
    """Vision model function for image analysis."""

    web_search_func: Optional[Callable] = field(default=None)
    """Optional web search function for plan-then-retrieve querying."""

    embedding_func: Optional[Callable] = field(default=None)
    """Embedding function for text vectorization."""

    config: Optional[RAGAnythingConfig] = field(default=None)
    """Configuration object, if None will create with environment variables."""

    # LightRAG Configuration
    # ---
    lightrag_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments for LightRAG initialization when lightrag is not provided.
    This allows passing all LightRAG configuration parameters like:
    - kv_storage, vector_storage, graph_storage, doc_status_storage
    - top_k, chunk_top_k, max_entity_tokens, max_relation_tokens, max_total_tokens
    - cosine_threshold, related_chunk_number
    - chunk_token_size, chunk_overlap_token_size, tokenizer, tiktoken_model_name
    - embedding_batch_num, embedding_func_max_async, embedding_cache_config
    - llm_model_name, llm_model_max_token_size, llm_model_max_async, llm_model_kwargs
    - rerank_model_func, vector_db_storage_cls_kwargs, enable_llm_cache
    - max_parallel_insert, max_graph_nodes, addon_params, etc.
    """

    # Internal State
    # ---
    modal_processors: Dict[str, Any] = field(default_factory=dict, init=False)
    """Dictionary of multimodal processors."""

    context_extractor: Optional[ContextExtractor] = field(default=None, init=False)
    """Context extractor for providing surrounding content to modal processors."""

    parse_cache: Optional[Any] = field(default=None, init=False)
    """Parse result cache storage using LightRAG KV storage."""

    multimodal_desc_cache: Optional[Any] = field(default=None, init=False)
    """Multimodal description cache keyed by doc_id + item hash."""

    kg_quality_manager: Optional[KGQualityManager] = field(default=None, init=False)
    """Knowledge graph quality manager for normalization and cleanup."""

    callback_manager: CallbackManager = field(
        default_factory=CallbackManager, init=False, repr=False
    )
    """Processing callbacks manager (optional hooks for observability and metrics)."""

    _parser_installation_checked: bool = field(default=False, init=False)
    """Flag to track if parser installation has been checked."""

    def __post_init__(self):
        """Post-initialization setup following LightRAG pattern"""
        # Initialize configuration if not provided
        if self.config is None:
            self.config = RAGAnythingConfig()

        # Set working directory
        self.working_dir = self.config.working_dir

        # Set up logger (use existing logger, don't configure it)
        self.logger = logger

        # Initialize KG quality manager early so all pipelines can reuse it.
        self.kg_quality_manager = KGQualityManager(
            enabled=self.config.kg_quality_enabled,
            canonical_language=self.config.kg_canonical_language,
            relation_schema=self.config.kg_relation_schema,
            extraction_mode=self.config.kg_extraction_mode,
            ontology_profile=self.config.kg_ontology_profile,
            enforce_ontology=self.config.kg_enforce_ontology,
            merge_threshold=self.config.kg_merge_threshold,
            description_policy=self.config.kg_description_policy,
            core_entity_types=self.config.kg_core_entity_types,
            anchor_node_types=self.config.kg_anchor_node_types,
            attribute_fields=self.config.kg_attribute_fields,
            attribute_host_types=self.config.kg_attribute_host_types,
            noise_drop_types=self.config.kg_noise_drop_types,
            noise_drop_patterns=self.config.kg_noise_drop_patterns,
            multimodal_min_desc_chars=self.config.kg_multimodal_min_desc_chars,
            drop_empty_multimodal=self.config.kg_drop_empty_multimodal,
            llm_model_func=self.llm_model_func,
            llm_semantic_merge_enabled=self.config.kg_llm_semantic_merge_enabled,
            llm_semantic_merge_types=self.config.kg_llm_semantic_merge_types,
            llm_semantic_name_sim_threshold=self.config.kg_llm_semantic_name_sim_threshold,
            llm_semantic_merge_min_confidence=self.config.kg_llm_semantic_merge_min_confidence,
            llm_semantic_merge_max_group_size=self.config.kg_llm_semantic_merge_max_group_size,
            llm_semantic_merge_max_groups=self.config.kg_llm_semantic_merge_max_groups,
            llm_timeout_seconds=self.config.kg_llm_timeout_seconds,
        )

        # Keep prompts and summary language aligned with canonical language.
        if (
            self.kg_quality_manager.enabled
            and self.config.kg_canonical_language.lower() == "zh"
        ):
            try:
                from raganything.prompt_manager import set_prompt_language

                set_prompt_language("zh")
                self.logger.info("Prompt language set to Chinese for KG consistency")
            except Exception as exc:
                self.logger.warning(f"Failed to set prompt language to zh: {exc}")

            os.environ["SUMMARY_LANGUAGE"] = "Chinese"

        # Set up document parser
        self.doc_parser = get_parser(self.config.parser)

        # Register close method for cleanup
        atexit.register(self.close)

        # Create working directory if needed
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            self.logger.info(f"Created working directory: {self.working_dir}")

        # Log configuration info
        self.logger.info("RAGAnything initialized with config:")
        self.logger.info(f"  Working directory: {self.config.working_dir}")
        self.logger.info(f"  Parser: {self.config.parser}")
        self.logger.info(f"  Parse method: {self.config.parse_method}")
        self.logger.info(
            f"  Multimodal processing - Image: {self.config.enable_image_processing}, "
            f"Table: {self.config.enable_table_processing}, "
            f"Equation: {self.config.enable_equation_processing}"
        )
        self.logger.info(f"  Max concurrent files: {self.config.max_concurrent_files}")
        self.logger.info(
            "  KG quality - enabled: %s, canonical_language: %s, relation_schema: %s, extraction_mode: %s, ontology_profile: %s, enforce_ontology: %s, description_policy: %s",
            self.config.kg_quality_enabled,
            self.config.kg_canonical_language,
            self.config.kg_relation_schema,
            self.config.kg_extraction_mode,
            self.config.kg_ontology_profile,
            self.config.kg_enforce_ontology,
            self.config.kg_description_policy,
        )
        self.logger.info(
            "  KG quality - attribute_hosts: %s, multimodal_min_desc_chars: %s, drop_empty_multimodal: %s",
            self.config.kg_attribute_host_types,
            self.config.kg_multimodal_min_desc_chars,
            self.config.kg_drop_empty_multimodal,
        )
        self.logger.info(
            "  Multimodal cache - desc_cache_enabled: %s, hard_skip_enabled: %s",
            self.config.multimodal_desc_cache_enabled,
            self.config.multimodal_hard_skip_enabled,
        )

    def close(self):
        """Cleanup resources when object is destroyed.

        Handles three common scenarios:
        1. Inside a running async context (e.g., FastAPI shutdown) -> schedule task
        2. No event loop in thread (typical atexit) -> create one with asyncio.run()
        3. Event loop exists but is closed/closing (atexit race) -> create new loop
        """
        try:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                # Case 1: We're inside a running event loop, schedule cleanup task
                loop.create_task(self.finalize_storages())
            else:
                # Case 2/3: No running loop. Clean up any stale loop reference
                # so asyncio.run() can create a fresh one (Python 3.10+ raises
                # RuntimeError if a loop is already set for the thread).
                if loop is not None:
                    try:
                        loop.close()
                    except Exception:
                        pass
                    asyncio.set_event_loop(None)
                asyncio.run(self.finalize_storages())
        except Exception:
            # Silently ignore during interpreter shutdown - the event loop and
            # resources are being torn down anyway, and printing may fail if
            # stdout/stderr are already closed. This avoids the noisy
            # "There is no current event loop in thread 'MainThread'" warning
            # that confused users (#135).
            pass

    def _create_context_config(self) -> ContextConfig:
        """Create context configuration from RAGAnything config"""
        return ContextConfig(
            context_window=self.config.context_window,
            context_mode=self.config.context_mode,
            max_context_tokens=self.config.max_context_tokens,
            include_headers=self.config.include_headers,
            include_captions=self.config.include_captions,
            filter_content_types=self.config.context_filter_content_types,
        )

    def _create_context_extractor(self) -> ContextExtractor:
        """Create context extractor with tokenizer from LightRAG"""
        if self.lightrag is None:
            raise ValueError(
                "LightRAG must be initialized before creating context extractor"
            )

        context_config = self._create_context_config()
        return ContextExtractor(
            config=context_config, tokenizer=self.lightrag.tokenizer
        )

    def _initialize_processors(self):
        """Initialize multimodal processors with appropriate model functions"""
        if self.lightrag is None:
            raise ValueError(
                "LightRAG instance must be initialized before creating processors"
            )

        # Create context extractor
        self.context_extractor = self._create_context_extractor()

        # Create different multimodal processors based on configuration
        self.modal_processors = {}

        if self.config.enable_image_processing:
            self.modal_processors["image"] = ImageModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.vision_model_func or self.llm_model_func,
                context_extractor=self.context_extractor,
                kg_quality_manager=self.kg_quality_manager,
            )

        if self.config.enable_table_processing:
            self.modal_processors["table"] = TableModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.llm_model_func,
                context_extractor=self.context_extractor,
                kg_quality_manager=self.kg_quality_manager,
            )

        if self.config.enable_equation_processing:
            self.modal_processors["equation"] = EquationModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.llm_model_func,
                context_extractor=self.context_extractor,
                kg_quality_manager=self.kg_quality_manager,
            )

        # Always include generic processor as fallback
        self.modal_processors["generic"] = GenericModalProcessor(
            lightrag=self.lightrag,
            modal_caption_func=self.llm_model_func,
            context_extractor=self.context_extractor,
            kg_quality_manager=self.kg_quality_manager,
        )

        self.logger.info("Multimodal processors initialized with context support")
        self.logger.info(f"Available processors: {list(self.modal_processors.keys())}")
        self.logger.info(f"Context configuration: {self._create_context_config()}")

    def update_config(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown config parameter: {key}")

    async def _ensure_lightrag_initialized(self):
        """Ensure LightRAG instance is initialized, create if necessary"""
        try:
            # Check parser installation first
            if not self._parser_installation_checked:
                if not self.doc_parser.check_installation():
                    error_msg = (
                        f"Parser '{self.config.parser}' is not properly installed. "
                        "Please install it using 'pip install' or 'uv pip install'."
                    )
                    self.logger.error(error_msg)
                    return {"success": False, "error": error_msg}

                self._parser_installation_checked = True
                self.logger.info(f"Parser '{self.config.parser}' installation verified")

            if self.lightrag is not None:
                # LightRAG was pre-provided, but we need to ensure it's properly initialized
                # Inherit model functions from LightRAG if not explicitly provided
                if self.llm_model_func is None and hasattr(
                    self.lightrag, "llm_model_func"
                ):
                    self.llm_model_func = self.lightrag.llm_model_func
                    self.logger.debug("Inherited llm_model_func from LightRAG instance")
                if self.kg_quality_manager is not None:
                    self.kg_quality_manager.set_llm_model_func(self.llm_model_func)

                if self.embedding_func is None and hasattr(
                    self.lightrag, "embedding_func"
                ):
                    self.embedding_func = self.lightrag.embedding_func
                    self.logger.debug("Inherited embedding_func from LightRAG instance")

                try:
                    # Ensure LightRAG storages are initialized
                    if (
                        not hasattr(self.lightrag, "_storages_status")
                        or self.lightrag._storages_status.name != "INITIALIZED"
                    ):
                        self.logger.info(
                            "Initializing storages for pre-provided LightRAG instance"
                        )
                        await self.lightrag.initialize_storages()
                        from lightrag.kg.shared_storage import (
                            initialize_pipeline_status,
                        )

                        await initialize_pipeline_status()

                    # Initialize parse cache if not already done
                    if self.parse_cache is None:
                        self.logger.info(
                            "Initializing parse cache for pre-provided LightRAG instance"
                        )
                        self.parse_cache = (
                            self.lightrag.key_string_value_json_storage_cls(
                                namespace="parse_cache",
                                workspace=self.lightrag.workspace,
                                global_config=self.lightrag.__dict__,
                                embedding_func=self.embedding_func,
                            )
                        )
                        await self.parse_cache.initialize()
                    if (
                        self.config.multimodal_desc_cache_enabled
                        and self.multimodal_desc_cache is None
                    ):
                        self.logger.info(
                            "Initializing multimodal description cache for pre-provided LightRAG instance"
                        )
                        self.multimodal_desc_cache = (
                            self.lightrag.key_string_value_json_storage_cls(
                                namespace="multimodal_desc_cache",
                                workspace=self.lightrag.workspace,
                                global_config=self.lightrag.__dict__,
                                embedding_func=self.embedding_func,
                            )
                        )
                        await self.multimodal_desc_cache.initialize()

                    # Initialize processors if not already done
                    if not self.modal_processors:
                        self._initialize_processors()

                    return {"success": True}

                except Exception as e:
                    error_msg = (
                        f"Failed to initialize pre-provided LightRAG instance: {str(e)}"
                    )
                    self.logger.error(error_msg, exc_info=True)
                    return {"success": False, "error": error_msg}

            # Validate required functions for creating new LightRAG instance
            if self.llm_model_func is None:
                error_msg = "llm_model_func must be provided when LightRAG is not pre-initialized"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            if self.embedding_func is None:
                error_msg = "embedding_func must be provided when LightRAG is not pre-initialized"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            from lightrag.kg.shared_storage import initialize_pipeline_status

            # Prepare LightRAG initialization parameters
            lightrag_params = {
                "working_dir": self.working_dir,
                "llm_model_func": self.llm_model_func,
                "embedding_func": self.embedding_func,
            }

            # Merge user-provided lightrag_kwargs, which can override defaults
            lightrag_params.update(self.lightrag_kwargs)

            # Unify extraction white-lists from kg_quality manager.
            manager = self.kg_quality_manager
            if manager and manager.enabled:
                addon_params = dict(lightrag_params.get("addon_params", {}))
                addon_params["entity_types"] = manager.get_lightrag_entity_types()
                addon_params["relationship_types"] = manager.get_lightrag_relation_types()
                addon_params.setdefault(
                    "language",
                    "Chinese"
                    if self.config.kg_canonical_language.lower() == "zh"
                    else "English",
                )
                lightrag_params["addon_params"] = addon_params

            # Log the parameters being used for initialization (excluding sensitive data)
            log_params = {
                k: v
                for k, v in lightrag_params.items()
                if not callable(v)
                and k not in ["llm_model_kwargs", "vector_db_storage_cls_kwargs"]
            }
            self.logger.info(f"Initializing LightRAG with parameters: {log_params}")

            try:
                # Create LightRAG instance with merged parameters
                self.lightrag = LightRAG(**lightrag_params)
                if self.kg_quality_manager is not None:
                    self.kg_quality_manager.set_llm_model_func(self.llm_model_func)
                await self.lightrag.initialize_storages()
                await initialize_pipeline_status()

                # Initialize parse cache storage using LightRAG's KV storage
                self.parse_cache = self.lightrag.key_string_value_json_storage_cls(
                    namespace="parse_cache",
                    workspace=self.lightrag.workspace,
                    global_config=self.lightrag.__dict__,
                    embedding_func=self.embedding_func,
                )
                await self.parse_cache.initialize()
                if self.config.multimodal_desc_cache_enabled:
                    self.multimodal_desc_cache = (
                        self.lightrag.key_string_value_json_storage_cls(
                            namespace="multimodal_desc_cache",
                            workspace=self.lightrag.workspace,
                            global_config=self.lightrag.__dict__,
                            embedding_func=self.embedding_func,
                        )
                    )
                    await self.multimodal_desc_cache.initialize()

                # Initialize processors after LightRAG is ready
                self._initialize_processors()

                self.logger.info(
                    "LightRAG, parse cache, multimodal desc cache, and multimodal processors initialized"
                )
                return {"success": True}

            except Exception as e:
                error_msg = f"Failed to initialize LightRAG instance: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"Unexpected error during LightRAG initialization: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {"success": False, "error": error_msg}

    async def finalize_storages(self):
        """Finalize all storages including parse cache and LightRAG storages

        This method should be called when shutting down to properly clean up resources
        and persist any cached data. It will finalize both the parse cache and LightRAG's
        internal storages.

        Example usage:
            try:
                rag_anything = RAGAnything(...)
                await rag_anything.process_file("document.pdf")
                # ... other operations ...
            finally:
                # Always finalize storages to clean up resources
                if rag_anything:
                    await rag_anything.finalize_storages()

        Note:
            - This method is automatically called in __del__ when the object is destroyed
            - Manual calling is recommended in production environments
            - All finalization tasks run concurrently for better performance
        """
        try:
            tasks = []

            # Finalize parse cache if it exists
            if self.parse_cache is not None:
                tasks.append(self.parse_cache.finalize())
                self.logger.debug("Scheduled parse cache finalization")
            if self.multimodal_desc_cache is not None:
                tasks.append(self.multimodal_desc_cache.finalize())
                self.logger.debug("Scheduled multimodal desc cache finalization")

            # Finalize LightRAG storages if LightRAG is initialized
            if self.lightrag is not None:
                tasks.append(self.lightrag.finalize_storages())
                self.logger.debug("Scheduled LightRAG storages finalization")

            # Run all finalization tasks concurrently
            if tasks:
                await asyncio.gather(*tasks)
                self.logger.info("Successfully finalized all RAGAnything storages")
            else:
                self.logger.debug("No storages to finalize")

        except Exception as e:
            self.logger.error(f"Error during storage finalization: {e}")
            raise

    def check_parser_installation(self) -> bool:
        """
        Check if the configured parser is properly installed

        Returns:
            bool: True if the configured parser is properly installed
        """
        return self.doc_parser.check_installation()

    def verify_parser_installation_once(self) -> bool:
        if not self._parser_installation_checked:
            if not self.doc_parser.check_installation():
                raise RuntimeError(
                    f"Parser '{self.config.parser}' is not properly installed. "
                    "Please install it using pip install or uv pip install."
                )
            self._parser_installation_checked = True
            self.logger.info(f"Parser '{self.config.parser}' installation verified")
        return True

    def get_config_info(self) -> Dict[str, Any]:
        """Get current configuration information"""
        config_info = {
            "directory": {
                "working_dir": self.config.working_dir,
                "parser_output_dir": self.config.parser_output_dir,
            },
            "parsing": {
                "parser": self.config.parser,
                "parse_method": self.config.parse_method,
                "display_content_stats": self.config.display_content_stats,
            },
            "multimodal_processing": {
                "enable_image_processing": self.config.enable_image_processing,
                "enable_table_processing": self.config.enable_table_processing,
                "enable_equation_processing": self.config.enable_equation_processing,
            },
            "context_extraction": {
                "context_window": self.config.context_window,
                "context_mode": self.config.context_mode,
                "max_context_tokens": self.config.max_context_tokens,
                "include_headers": self.config.include_headers,
                "include_captions": self.config.include_captions,
                "filter_content_types": self.config.context_filter_content_types,
            },
            "batch_processing": {
                "max_concurrent_files": self.config.max_concurrent_files,
                "supported_file_extensions": self.config.supported_file_extensions,
                "recursive_folder_processing": self.config.recursive_folder_processing,
            },
            "kg_quality": {
                "kg_quality_enabled": self.config.kg_quality_enabled,
                "kg_canonical_language": self.config.kg_canonical_language,
                "kg_relation_schema": self.config.kg_relation_schema,
                "kg_ontology_profile": self.config.kg_ontology_profile,
                "kg_enforce_ontology": self.config.kg_enforce_ontology,
                "kg_merge_threshold": self.config.kg_merge_threshold,
                "kg_llm_semantic_merge_enabled": self.config.kg_llm_semantic_merge_enabled,
                "kg_llm_semantic_merge_types": self.config.kg_llm_semantic_merge_types,
                "kg_llm_semantic_name_sim_threshold": self.config.kg_llm_semantic_name_sim_threshold,
                "kg_llm_semantic_merge_min_confidence": self.config.kg_llm_semantic_merge_min_confidence,
            },
            "logging": {
                "note": "Logging fields have been removed - configure logging externally",
            },
        }

        # Add LightRAG configuration if available
        if self.lightrag_kwargs:
            # Filter out sensitive data and callable objects for display
            safe_kwargs = {
                k: v
                for k, v in self.lightrag_kwargs.items()
                if not callable(v)
                and k not in ["llm_model_kwargs", "vector_db_storage_cls_kwargs"]
            }
            config_info["lightrag_config"] = {
                "custom_parameters": safe_kwargs,
                "note": "LightRAG will be initialized with these additional parameters",
            }
        else:
            config_info["lightrag_config"] = {
                "custom_parameters": {},
                "note": "Using default LightRAG parameters",
            }

        return config_info

    def clean_kg(self, graphml_path: str | None = None, rewrite: bool = True) -> Dict[str, Any]:
        """
        Run one-shot GraphML quality cleanup.

        Args:
            graphml_path: Optional custom GraphML path.
            rewrite: Rewrite file in-place if True.

        Returns:
            cleanup report dictionary.
        """
        if not self.kg_quality_manager or not self.kg_quality_manager.enabled:
            return {"enabled": False, "message": "KG quality manager disabled"}

        target_path = graphml_path or os.path.join(
            self.config.working_dir, "graph_chunk_entity_relation.graphml"
        )
        report = self.kg_quality_manager.clean_graphml_file(target_path, rewrite=rewrite)
        self.logger.info(
            "KG cleanup complete: nodes=%s edges=%s non_cjk_entity_ratio=%.4f",
            report.get("nodes_after"),
            report.get("edges_after"),
            report.get("non_cjk_entity_ratio", 0.0),
        )
        return report

    def set_content_source_for_context(
        self, content_source, content_format: str = "auto"
    ):
        """Set content source for context extraction in all modal processors

        Args:
            content_source: Source content for context extraction (e.g., MinerU content list)
            content_format: Format of content source ("minerU", "text_chunks", "auto")
        """
        if not self.modal_processors:
            self.logger.warning(
                "Modal processors not initialized. Content source will be set when processors are created."
            )
            return

        for processor_name, processor in self.modal_processors.items():
            try:
                processor.set_content_source(content_source, content_format)
                self.logger.debug(f"Set content source for {processor_name} processor")
            except Exception as e:
                self.logger.error(
                    f"Failed to set content source for {processor_name}: {e}"
                )

        self.logger.info(
            f"Content source set for context extraction (format: {content_format})"
        )

    def update_context_config(self, **context_kwargs):
        """Update context extraction configuration

        Args:
            **context_kwargs: Context configuration parameters to update
                (context_window, context_mode, max_context_tokens, etc.)
        """
        # Update the main config
        for key, value in context_kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated context config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown context config parameter: {key}")

        # Recreate context extractor with new config if processors are initialized
        if self.lightrag and self.modal_processors:
            try:
                self.context_extractor = self._create_context_extractor()
                # Update all processors with new context extractor
                for processor_name, processor in self.modal_processors.items():
                    processor.context_extractor = self.context_extractor

                self.logger.info(
                    "Context configuration updated and applied to all processors"
                )
                self.logger.info(
                    f"New context configuration: {self._create_context_config()}"
                )
            except Exception as e:
                self.logger.error(f"Failed to update context configuration: {e}")

    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information"""
        base_info = {
            "mineru_installed": MineruParser.check_installation(MineruParser()),
            "parser_installation": {
                parser_name: get_parser(parser_name).check_installation()
                for parser_name in SUPPORTED_PARSERS
            },
            "config": self.get_config_info(),
            "models": {
                "llm_model": "External function"
                if self.llm_model_func
                else "Not provided",
                "vision_model": "External function"
                if self.vision_model_func
                else "Not provided",
                "embedding_model": "External function"
                if self.embedding_func
                else "Not provided",
            },
        }

        if not self.modal_processors:
            base_info["status"] = "Not initialized"
            base_info["processors"] = {}
        else:
            base_info["status"] = "Initialized"
            base_info["processors"] = {}

            for proc_type, processor in self.modal_processors.items():
                base_info["processors"][proc_type] = {
                    "class": processor.__class__.__name__,
                    "supports": get_processor_supports(proc_type),
                    "enabled": True,
                }

        return base_info
