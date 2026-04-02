### 十字花科病虫害图谱修复方案（按你已选偏好）

#### Summary
- 先修复“属性丢失”的主因：`text -> LightRAG`链路没有做属性节点转属性字段，后续清洗又把属性类型节点删掉了。
- 多模态按你选择走“仅强化 Prompt”路线，不加规则表格解析器，但会加短描述重试和空结果硬丢弃。
- 去重按你确认的“仅精确去重”执行，不做语义合并；对“鞘翅目叶甲科”这类问题改为源头命名约束来降低新增噪声。

#### Key Changes
- `raganything/kg_quality.py`：
  - 在 `clean_graphml_file()` 增加“属性投影阶段”（在 PPE 过滤节点之前执行）：
    - 识别属性类型节点（`形态特征/危害症状/发病诱因/发生时期/防治要点`）。
    - 将其内容挂到相邻主实体节点的 `attributes`（仅主实体类型，见新配置）。
    - 证据写入 `attribute_evidence`（chunk/file/page）。
    - 投影完成后再删除属性节点，避免信息丢失。
  - 保留“仅多模态保留 description”策略。
  - 新增“空多模态节点剔除”逻辑：`detailed_description` 为空或摘要为空的图/表节点直接不入图。
  - 保留“仅精确去重”策略（仅空白/标点归一，不做语义合并）。

- `raganything/processor.py` + 多模态处理链路：
  - 为图/表/公式主节点强制写入 `content_modality=image|table|equation`（不再依赖名称推断）。
  - 多模态描述增加一次“短描述重试”（仅非二维码、非空结果），防止过短摘要导致后续抽取弱。
  - 批量 merge 前统一兜底关系描述非空，避免 `has no description` 中断。

- `raganything/prompts_zh.py`（同步 `raganything/prompt.py`）：
  - 强化图/表提示词：要求 `detailed_description` 明确包含“实体候选、属性句、关系线索、证据短句”。
  - 对表格明确要求按“对象-字段-值”描述，避免只给一句总述。
  - 生物分类命名约束（源头降噪）：优先输出规范单一分类名，避免复合写法反复出现。

- `raganything/config.py`（公开配置新增）：
  - `kg_attribute_host_types`：默认 `["虫害","病害","病原菌"]`（仅主实体挂属性）。
  - `kg_multimodal_min_desc_chars`：默认 `80`（触发重试阈值）。
  - `kg_drop_empty_multimodal`：默认 `True`（空多模态不入图）。

#### Test Plan（仅单元/模块）
- `tests/test_kg_quality.py` 新增：
  - `clean_graphml_file` 能把属性类型节点正确投影到主实体 `attributes`，并删除属性节点。
  - 空多模态节点会被剔除。
  - 仅精确去重下，语义近似分类名不合并（行为可预期）。
- `tests/test_ppe_extraction.py` 新增：
  - 主实体挂属性约束生效（不挂到部位/生长期）。
  - 短描述重试触发条件与跳过条件（二维码/空结果）正确。
  - `content_modality` 写入正确。
- 模块级验证：
  - `python -m compileall raganything`
  - `pytest tests/test_kg_quality.py tests/test_ppe_extraction.py`

#### Assumptions
- 按你确认：`仅强化 Prompt`、`仅主实体挂属性`、`仅精确去重`、`空多模态直接丢弃`。
- 不做集成测试；本轮只交付单元与模块级可验证改造。
- 在“仅精确去重”前提下，历史语义近似重复不会自动合并，主要靠后续抽取命名约束减少新增。

### 代码实现总结
已按 `plan3` 完成实现，核心改造已落地并通过你要求的模块级验证。

**已实现内容**
- 在 [kg_quality.py](/media/disk2/wzh/RAG-Anything/raganything/kg_quality.py)：
  - 新增 `clean_graphml_file()` 的“属性投影阶段”：先把属性类节点（形态特征/危害症状/发病诱因/发生时期/防治要点等）挂到主实体属性，再做 PPE 节点裁剪，避免属性丢失。
  - 属性挂载限制到 `kg_attribute_host_types`（默认虫害/病害/病原菌）。
  - 新增空多模态节点剔除（`detailed_description` 与 `summary` 均空则丢弃）。
  - 保留“仅精确去重”（未做语义合并）。
  - 重新打通“别名不建边，写入 aliases 属性”。
- 在 [config.py](/media/disk2/wzh/RAG-Anything/raganything/config.py)：
  - 新增配置：
    - `kg_attribute_host_types`
    - `kg_multimodal_min_desc_chars`
    - `kg_drop_empty_multimodal`
- 在 [raganything.py](/media/disk2/wzh/RAG-Anything/raganything/raganything.py)：
  - 将上述新配置注入 `KGQualityManager`。
- 在 [processor.py](/media/disk2/wzh/RAG-Anything/raganything/processor.py)：
  - 多模态增加一次短描述重试（仅非二维码、非空结果）。
  - 空多模态结果在入抽取前直接丢弃。
  - 批量 merge 前统一补齐空关系描述为 `N/A`，避免 `has no description` 中断。
  - 强制为图/表/公式主节点写入 `content_modality`。
- 在 [modalprocessors.py](/media/disk2/wzh/RAG-Anything/raganything/modalprocessors.py)：
  - 回退链路也补齐 `content_modality` 写入。
- 在 [prompts_zh.py](/media/disk2/wzh/RAG-Anything/raganything/prompts_zh.py) 与 [prompt.py](/media/disk2/wzh/RAG-Anything/raganything/prompt.py)：
  - 强化图/表 prompt，要求 `detailed_description` 明确给出“实体候选/属性句/关系线索/证据短句”。
  - 表格强调“对象-字段-值”表达。
  - 增加生物分类命名约束（优先单一规范名，避免复合分类写法反复出现）。
- 在 [env.example](/media/disk2/wzh/RAG-Anything/env.example)：
  - 同步新增配置示例。
- 在测试：
  - [test_kg_quality.py](/media/disk2/wzh/RAG-Anything/tests/test_kg_quality.py)
  - [test_ppe_extraction.py](/media/disk2/wzh/RAG-Anything/tests/test_ppe_extraction.py)
  - 新增了属性投影、空多模态剔除、主实体挂属性约束、重试规则等单测。

**验证结果**
- `python -m compileall raganything tests/test_kg_quality.py tests/test_ppe_extraction.py` 通过
- `/media/disk2/lhy/anaconda/envs/rag_anything/bin/python -m pytest tests/test_kg_quality.py tests/test_ppe_extraction.py -q` 通过（`20 passed`）

如果你需要，我下一步可以直接基于这版代码给你一条“重新跑 `pdf_rag_pipeline.py` 并最小化复用旧缓存”的执行命令清单。