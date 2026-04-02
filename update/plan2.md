### 十字花科病虫害图谱质量重构方案（PPE + IEU + 属性中心图谱）

#### Summary
- 目标：把当前“节点爆炸+噪声多+语义碎片化”的图谱，重构为“核心节点少、属性信息完整、关系可控”的农业病虫害图谱。
- 采用策略：`源头硬过滤` + `三轮PPE递进抽取` + `IEU增量实体更新` + `属性中心图谱投影`。
- 已确认决策：
  - 图谱形态：属性中心图谱
  - description策略：仅多模态节点保留
  - 噪声策略：源头硬过滤
  - PPE深度：三轮完整PPE
  - IEU策略：主属性优先
  - 节点策略：核心节点自定义集合，`部位/时间`保留为节点，其余特征类信息转属性
  - 验收：强约束（节点下降≥60%，无关节点<10%，核心节点>70%）

#### Key Changes
- 本体注册与抽取约束（Ro）：
  - 在 `raganything/kg_quality.py` 增加“本体注册中心”结构（实体Schema、关系Schema、属性Schema、抽取指令模板索引）。
  - 核心节点类型固定为：`虫害/病害/作物/病原菌/药剂/生长期/生物分类`，并保留 `部位/时间` 为关系锚点节点。
  - 将 `形态特征/危害症状/发病诱因/发生时期/防治要点` 归为实体属性，不再独立成大量碎片节点。
- 源头硬过滤（先减噪再抽取）：
  - 在解析后、入图前过滤 `header/page_number/QR码/图片路径/None` 等无关内容。
  - 增加噪声规则：类型过滤 + 文本模式过滤 + 路径模式过滤，防止无关内容进入PPE。
  - 过滤结果保留审计计数（丢弃条数、命中规则）用于质量报告。
- PPE三轮抽取（替代当前一次性抽取）：
  - 第1轮：核心实体抽取（只抽本体允许的核心实体与锚点节点）。
  - 第2轮：属性抽取（将形态、症状等整句信息作为结构化属性挂到对应实体，避免“初孵幼虫/淡黄色”碎片化）。
  - 第3轮：关系抽取（仅在核心实体与锚点节点之间抽取本体允许关系）。
  - 抽取输出统一为结构化JSON，进入IEU合并器，不直接写图。
- IEU增量实体更新（主属性优先）：
  - 以“实体主记录”为中心进行增量合并，冲突时主属性优先，新信息仅补充不覆盖冲突字段。
  - 每个属性保留来源证据（chunk_id/page_idx/file_path）和置信度，支持可追溯更新。
- 属性中心图谱投影与写入：
  - 图谱节点只保留核心实体 + 锚点节点（部位/时间/必要多模态元素）。
  - 边只保留本体关系集合；默认不写边description。
  - description策略改为 `multimodal_only`：仅 `多模态元素` 节点保留 description，其他节点description留空。
  - 保留 `content_modality`（image/table/equation）字段用于多模态检索解释。
- 配置与接口扩展（`raganything/config.py`）：
  - 新增：
    - `kg_extraction_mode`（`ppe`/`legacy`）
    - `kg_core_entity_types`
    - `kg_anchor_node_types`（默认含 `部位/时间`）
    - `kg_attribute_fields`
    - `kg_noise_drop_types`
    - `kg_noise_drop_patterns`
    - `kg_description_policy`（默认 `multimodal_only`）
  - 运行入口（`pdf_rag_pipeline.py`）默认启用 `kg_extraction_mode=\"ppe\"` 与上述策略。
  - `rag.clean_kg()` 保留为后处理兜底，但主治理前移到“抽取前+抽取中”。

#### Test Plan
- 单元测试：
  - 噪声过滤：`header/page_number/QR码/路径/None` 正确丢弃。
  - PPE轮次输出：实体轮只产核心实体；属性轮不产生碎片节点；关系轮只产本体关系。
  - IEU冲突合并：主属性优先规则稳定，新增信息只补充。
  - description策略：仅多模态节点有description，普通节点/边无description。
- 集成测试（基于你当前 `rag_storage_test1` 数据）：
  - 指标对比（改造前后）：
    - 节点总数下降 ≥ 60%
    - 无关节点占比 < 10%
    - 核心节点占比 > 70%
    - `其他` 类型占比显著下降
    - `属于/关联` 泛化边占比显著下降
  - 业务验证：
    - “小猿叶甲形态特征/危害症状/防治方法”查询能返回完整属性句，不再碎片化。
    - 页眉、页码、QR码、图片路径不再主导图谱结构。

#### Assumptions
- 默认以十字花科病虫害为主，但本体注册中心保持可扩展到多作物。
- 首版重点是“结构质量和可解释性”，允许召回轻微波动；后续通过别名词典与属性补全迭代提升召回。
- 兼容现有链路：保留 `legacy` 模式可回退；`ppe` 模式作为新默认用于新文档构建。




### 修改记录
/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/prompt.py
entity_extraction_system_prompt
entity_extraction_user_prompt
entity_continue_extraction_user_prompt


防止 relation description 为空导致崩溃
文件：operate.py
在关系合并处把空 description 回填为 N/A
原先 raise ValueError("Relation ... has no description") 的致命路径改为 fallback，不再直接中断


### 待办
把影响改为危害
属性提取的还是不够好
图里还是有"部位"，明明已经删掉了
到底用了哪些prompt？