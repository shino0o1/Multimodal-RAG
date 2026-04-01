### RAG-Anything 农业知识图谱质量治理方案（统一中文 + 全链路改造）

#### Summary
- 目标：解决你指出的 4 类问题（实体类型错配、重复实体、中英文混杂、关系语义混乱），并让同一套规则同时覆盖文本链路和多模态链路。
- 基线（当前图谱）：`101` 节点、`205` 边；`entity_id` 英文样式占比较高（73/101）；`虫害` 下存在明显错配实体；关系 `keywords` 离散度高（79 种）。
- 成功标准：`entity_type` 合规率 100%，主实体命名统一中文，重复实体显著下降，关系收敛为固定枚举并可解释。

#### Key Changes
- 新增“图谱质量层”（建议新模块：`raganything/kg_quality.py`），在入图前后执行统一治理：
  - 入图前标准化：对 `extract_entities` 结果做实体名规范化、类型校验、关系映射。
  - 入图后清洗：对全图做别名合并与关系重写，确保文本/多模态两条链路都被覆盖。
- 统一实体命名与类型约束：
  - 主键语言固定中文（英文仅保留在 `aliases` 字段）。
  - 建立农业领域 `entity_type` 白名单（你当前的 8 类为主），不在白名单的类型映射到 `其他` 并记录审计日志。
  - 引入“类型-名称一致性规则”（如 `虫害` 不允许 `Pest Populations` 这类泛词直接入主键）。
- 重复实体治理（两阶段）：
  - 规则归一：大小写、复数、括号后缀（如 `(image)`）、标点、空白、常见同义词先归一。
  - 语义合并：在同 `entity_type` 内用 `alias词典 + embedding相似度阈值` 合并到 canonical 节点，保留 `source_id/file_path` 证据链。
- 关系治理（固定枚举 + 映射）：
  - 新增 `relation_type` 枚举（如 `属于/影响/防治/症状/生命周期/别名/位于`），把现有自由 `keywords` 映射到枚举。
  - 原始 `keywords` 不丢弃，转存 `raw_keywords` 作为证据；检索和图遍历只用 `relation_type`。
  - `belongs_to` 保留，但只用于文档结构关系，不与领域语义关系混用。
- 语言一致性改造：
  - 在流程入口显式启用中文 prompt（`set_prompt_language("zh")`）。
  - 把摘要语言配置切到中文（`SUMMARY_LANGUAGE=Chinese`），避免合并摘要继续产出英文描述。
- 接口与配置变更（公开）：
  - `RAGAnythingConfig` 增加：`kg_quality_enabled`, `kg_canonical_language`, `kg_relation_schema`, `kg_merge_threshold`。
  - 提供一次性清洗入口（建议）：`rag.clean_kg()` 或独立脚本 `python -m raganything.kg_quality --working-dir ./rag_storage --rewrite`，用于修复历史图谱。

#### Test Plan
- 单元测试：
  - 实体规范化：中英混杂、括号后缀、复数词、同义词映射。
  - 类型校验：异常 `entity_id` + 错误 `entity_type` 能被纠正或降级。
  - 关系映射：自由 `keywords` 到 `relation_type` 映射稳定且可回溯。
- 集成测试（用你当前病虫害 PDF）：
  - 生成前后对比指标：英文主键占比、重复实体率、非法类型率、关系类型数量。
  - 验证 RAG 查询质量：同一问题回答稳定性与证据可解释性提升。
- 验收阈值（首版）：
  - 非中文主实体占比 < 5%；
  - `entity_type` 白名单外占比 = 0；
  - `relation_type` 种类数收敛到预设枚举（不再随文档剧烈波动）；
  - 重复实体（按 canonical）下降至少 50%。

#### Assumptions
- 采用你已确认的默认决策：`统一中文`、`全链路改造`、`固定枚举+映射`。
- 保留英文信息但不作为主键，仅作为别名和检索召回辅助。
- 首期优先“质量稳定和可解释”，允许少量召回下降；后续再通过别名词典和阈值微调补回召回。
