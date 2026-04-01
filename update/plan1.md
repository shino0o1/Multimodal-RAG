### RAG-Anything 农业知识图谱质量治理方案（统一中文 + 全链路改造）

#### 1. Summary
- 目标：解决 4 类问题（实体类型错配、重复实体、中英文混杂、关系语义混乱），并让同一套规则同时覆盖文本链路和多模态链路。
- 基线（改造前样本）：`101` 节点、`205` 边；`entity_id` 英文样式占比较高（73/101）；`虫害` 下存在明显错配实体；关系 `keywords` 离散度高（79 种）。
- 成功标准：`entity_type` 合规率 100%，主实体命名统一中文，重复实体显著下降，关系收敛为固定枚举并可解释。

#### 2. 治理方案（设计）
- 新增“图谱质量层”（`raganything/kg_quality.py`）：
  - 入图前标准化：对 `extract_entities` 结果做实体名规范化、类型校验、关系映射。
  - 入图后清洗：对全图做别名合并与关系重写，确保文本/多模态两条链路都治理到。
- 统一实体命名与类型约束：
  - 主键语言固定中文（英文保留在 `aliases`）。
  - 实体类型白名单；不合规类型降级为 `其他`。
  - 引入类型-名称一致性规则（如泛化英文虫害词不直接落在 `虫害`）。
- 重复实体治理：
  - 规则归一：大小写、复数、括号后缀（如 `(image)`）、标点、空白。
  - 语义归并：别名词典 + 阈值策略，保留 `source_id/file_path` 证据链。
- 关系治理（固定枚举 + 回溯）：
  - 抽取关系统一到 `relation_type`。
  - 原始 `keywords` 转存 `raw_keywords`。
  - `belongs_to` 保留但语义收敛为 `属于` 并区分结构关系。
- 语言一致性：
  - 流程入口显式 `set_prompt_language("zh")`。
  - `SUMMARY_LANGUAGE=Chinese`。

#### 3. 水稻本体迁移到十字花科后的本体设置（当前默认）
- 说明：本体结构采用“病虫害通用框架”，默认 profile 已从 `rice_disease_pest` 迁移为 `cruciferous_pest_disease`，并保留旧值兼容。
- 默认实体类型（节选）：
  - `病害`、`虫害`、`病原菌`、`病原属类`、`病害分类`、`栽培方式`、`地理位置`、`植物`、`植物生长期`、`部位`、`药剂`、`时间`、`农业防治`、`化学防治`、`生物分类`、`农作物`、`防治方法`、`危害症状`、`形态特征`、`生活习性`、`其他`。
- 默认关系类型（节选）：
  - `易发生病害`、`致病属类`、`致病`、`隶属`、`病害类型`、`地理位置`、`致病生长期`、`致病部位`、`治疗药剂`、`历史记录时间`。
  - 通用：`属于`、`防治`、`症状`、`生命周期`、`影响`、`别名`、`位于`、`关联`。
- 关系主宾类型约束（domain-range）：
  - 关键关系启用主语/宾语类型校验；不匹配时降级为 `关联`。
  - 例如：`治疗药剂` 期望 `(病害|虫害) -> 药剂`。

#### 4. 已完成代码修改（实现同步）
- 新增模块：
  - `raganything/kg_quality.py`
    - 新增 `KGQualityManager`。
    - 新增本体 profile：`cruciferous_pest_disease`（兼容 `rice_disease_pest`）。
    - 新增实体标准化、关系映射、GraphML 清洗、CLI。
    - 支持 `ontology_profile` + `enforce_ontology`。
- 配置扩展：
  - `raganything/config.py`
    - `kg_quality_enabled`
    - `kg_canonical_language`
    - `kg_relation_schema`
    - `kg_ontology_profile`（默认 `cruciferous_pest_disease`）
    - `kg_enforce_ontology`
    - `kg_merge_threshold`
- 主流程接入：
  - `raganything/raganything.py`
    - 初始化并注入 `kg_quality_manager`。
    - 初始化时设置中文 prompt 与摘要语言。
    - 暴露 `clean_kg()` 一次性清洗接口。
  - `raganything/processor.py`
    - 多模态实体入图前标准化。
    - `extract_entities` 后结果标准化。
    - 文档完成后自动触发全图清洗（覆盖文本+多模态）。
  - `raganything/modalprocessors.py`
    - 单条模态抽取链路接入标准化。
    - 边数据补充 `relation_type/raw_keywords`。
- 其它同步：
  - `raganything/__init__.py` 导出 `KGQualityManager`。
  - `pdf_rag_pipeline.py` 使用 `kg_ontology_profile="cruciferous_pest_disease"`。
  - `env.example` 增加 KG 治理配置项。

#### 5. 当前推荐配置（十字花科）
```python
config = RAGAnythingConfig(
    kg_quality_enabled=True,
    kg_canonical_language="zh",
    kg_relation_schema="fixed",
    kg_ontology_profile="cruciferous_pest_disease",
    kg_enforce_ontology=True,
    kg_merge_threshold=0.86,
)
```

#### 6. 清洗与运行命令
- 一次性清洗当前图谱：
```bash
python raganything/kg_quality.py \
  --graphml-path rag_storage/graph_chunk_entity_relation.graphml \
  --ontology-profile cruciferous_pest_disease \
  --rewrite
```
- 或在代码中调用：
```python
report = rag.clean_kg()
print(report)
```

#### 7. 测试与验证结果
- 新增测试：`tests/test_kg_quality.py`
  - 覆盖实体规范化、关系映射、本体约束降级、GraphML 清洗。
- 本地验证结果：
  - `pytest -q tests/test_kg_quality.py` -> `6 passed`
  - `python -m compileall ...` 通过。

#### 8. 结论与后续建议
- 结论：当前这套本体结构可以用于十字花科蔬菜知识图谱（包含多作物），本质是作物无关的病虫害通用框架。
- 后续建议：
  - 扩充 `ENTITY_ALIASES`（甘蓝、白菜、萝卜、花椰菜等同名/俗名/英文名）。
  - 继续细化本体属性槽位（如“发病诱因/发生时期/主要病原菌”结构化字段），减少 `description` 自由文本噪声。
