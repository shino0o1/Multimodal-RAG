# KG 语义去重改造说明（向量主导 + LLM 灰区仲裁）

## 本次目标
- 将 `kg_llm_semantic_merge_enabled` 默认作为**可选仲裁开关**，默认关闭。
- 语义去重流程改为：
  1. 向量相似（字符 n-gram + 余弦）主导严格合并。
  2. 仅在灰区候选上（可选）调用 LLM 做仲裁。

## 主要代码改动

### 1) `raganything/kg_quality.py`
- 新增名称向量相似能力：
  - `_char_ngram_vector()`
  - `_cosine_counter_similarity()`
  - `_vector_name_similarity()`
- 改造 `_name_similarity()`：由纯 `SequenceMatcher` 变为 `max(序列相似, 向量相似)`。
- 改造候选分组 `_build_candidate_groups()`：支持阈值与上界，允许构建“严格区/灰区”两类分组。
- 新增 `_select_canonical_name()`：规则化选择 canonical（中文优先、信息量优先、短名优先）。
- 重写 `_apply_llm_semantic_merge()` 为混合策略：
  - Stage 1: 向量严格合并（不依赖 LLM）
  - Stage 2: LLM 灰区仲裁（仅当 `kg_llm_semantic_merge_enabled=true` 且有 `llm_model_func`）
- 报告字段扩展：
  - `mode=vector_primary_llm_gray`
  - `vector_*` 和 `llm_*` 分阶段统计
  - `node_pairs_merged/edge_pairs_merged` 为总计

### 2) `pdf_rag_pipeline.py`
- 将 `kg_llm_semantic_merge_enabled` 改为 `False`（默认关闭 LLM 仲裁）。
- `kg_merge_threshold` 注释调整为灰区阈值语义。
- 降级重试提示文案从“关闭 LLM 语义去重”改为“关闭 LLM 灰区仲裁”。

### 3) `raganything/config.py`
- 配置文档语义更新：
  - `kg_merge_threshold`：灰区下阈值
  - `kg_llm_semantic_merge_enabled`：仅控制灰区 LLM 仲裁

### 4) `raganything/processor.py`
- KG 清洗日志增加模式与分阶段统计，便于观察：
  - 总合并 vs 向量合并 vs LLM 合并

### 5) `env.example`
- 增加注释：默认策略为“向量主导 + LLM 灰区仲裁（LLM 默认关闭）”。

## 默认行为（改造后）
- 即使 `kg_llm_semantic_merge_enabled=false`，仍会执行向量严格去重。
- 仅当显式开启 `kg_llm_semantic_merge_enabled=true` 时，才会对灰区候选调用 LLM。

## 阈值说明
- `kg_merge_threshold`：灰区下阈值（例如 0.85）。
- 严格阈值在代码内自动推导为更高值（默认至少 0.92），用于无 LLM 的高置信合并。

## 验证结果
- `python -m compileall raganything/kg_quality.py raganything/config.py raganything/processor.py pdf_rag_pipeline.py` 通过。
- `PYTHONPATH=. pytest -q tests/test_kg_quality.py` 通过（18 passed）。
- `tests/test_ppe_extraction.py` 在当前环境因缺少 `lightrag` 包无法收集执行（与本次改造逻辑无关）。
