### 十字花科病虫害图谱质量重构方案（无集成测试版）

#### Summary
- 目标：将当前图谱改为“属性中心 + 噪声前置过滤 + PPE递进抽取 + IEU增量更新”，显著减少无关节点和碎片化关系。
- 约束：不做集成测试，仅做单元与模块级验证。
- 已锁定：`源头硬过滤`、`三轮PPE`、`主属性优先IEU`、`仅多模态保留description`、`核心节点+部位/时间锚点`。

#### Key Changes
- 抽取与本体治理：
  - 在 `raganything/kg_quality.py` 增加“本体注册中心（Ro）”配置对象，显式定义：
    - 核心实体类型：`虫害/病害/作物/病原菌/药剂/生长期/生物分类`
    - 属性字段：`形态特征/危害症状/发病诱因/发生时期/防治要点` 等
    - 关系白名单与主宾约束（domain-range）
  - description策略改为 `multimodal_only`：仅 `多模态元素` 节点保留 description，普通节点和边默认清空 description。
- 源头硬过滤：
  - 在解析后、入图前新增过滤层（在 `processor.py` 管线中插入），直接剔除：
    - 类型：`header/page_number` 及其它无关结构项
    - 内容模式：页眉书名、页码、QR码、图片路径、`None`
  - 保留过滤统计（命中规则计数）用于日志审计。
- PPE三轮抽取（替代单轮粗抽）：
  - 第1轮：核心实体抽取（只产本体允许实体）
  - 第2轮：属性抽取（把完整语句挂到实体属性，避免碎片节点）
  - 第3轮：关系抽取（只产本体关系）
  - 三轮输出统一结构化JSON，中间结果不直接写图。
- IEU增量实体更新：
  - 以实体主记录为中心做增量合并，冲突时“主属性优先”，新增信息做补充。
  - 为每个属性保留来源证据（chunk/page/file）。
- 配置扩展（`raganything/config.py`）：
  - 新增并默认启用：
    - `kg_extraction_mode="ppe"`
    - `kg_core_entity_types`
    - `kg_anchor_node_types`
    - `kg_attribute_fields`
    - `kg_noise_drop_types`
    - `kg_noise_drop_patterns`
    - `kg_description_policy="multimodal_only"`
- 运行入口：
  - `pdf_rag_pipeline.py` 使用 PPE 模式与上述新配置；保留 `legacy` 回退开关。

#### Test Plan（仅单元/模块）
- 单元测试（`tests/test_kg_quality.py` 与新增 `tests/test_ppe_extraction.py`）：
  - 噪声过滤命中：页眉/页码/QR码/路径/None 被剔除。
  - PPE轮次约束：
    - 实体轮仅输出核心实体
    - 属性轮不生成碎片节点
    - 关系轮仅输出白名单关系
  - IEU规则：冲突时主属性优先；补充字段合并正确；来源证据保留。
  - description策略：仅多模态节点有description，其他节点和边为空。
  - `content_modality` 写入正确（image/table/equation）。
- 模块级验证：
  - 对 `kg_quality` CLI 做离线样本文件验证（不跑全流程集成）。
  - 语法与静态检查：`compileall` + 相关测试子集。

#### Assumptions
- 先做“结构质量”优先，不以端到端召回指标作为首版门槛。
- 不做集成测试；上线前仅依赖单元与模块级验证结果。
- 后续若需要，可再追加端到端评估脚本（独立于本次交付）。
