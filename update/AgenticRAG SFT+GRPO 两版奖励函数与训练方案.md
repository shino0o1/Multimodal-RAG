# AgenticRAG SFT+GRPO 两版奖励函数与训练方案

## Summary

当前论坛 QA 数据文件 `qa_merged.jsonl` 抽样结果显示：数据为标准 chat 格式，当前文件约 6961 条，包含 `question/answer/source/metadata`，其中 `metadata` 有 `crop`、`category`、`disease_or_pest`、`images`、`quality_score`、`confidence` 等字段。方案按两版设计：

- **版本1：LLM Judge 版**  
  奖励更接近人工语义判断，适合有联网 API 或本地强模型可用时使用。
- **版本2：纯离线 Reward 版**  
  不依赖 LLM Judge，只用规则、实体匹配、相似度、证据覆盖、格式检查，适合完全离线训练。

两版都采用同一训练路线：**SFT 冷启动 -> GRPO 强化 Agent 轨迹**。先不做 KG 覆盖率分桶，但会为每条样本构造 `pseudo_evidence`，并用证据质量作为软权重，避免教材 KG 与论坛 QA 分布不一致时误伤训练。

## 共同数据与轨迹格式

### 数据准备

原始样本：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "西蓝花烂花，有一小块发黑是什么原因？"},
    {"role": "assistant", "content": "是西蓝花的黑腐病，可用..."}
  ],
  "source": "惠农网",
  "metadata": {
    "crop": "西蓝花",
    "category": "病害防治",
    "disease_or_pest": "黑腐病",
    "images": [],
    "quality_score": 0.9,
    "confidence": 0.95
  }
}
```

转换为训练样本：

```json
{
  "question": "...",
  "gold_answer": "...",
  "metadata": {
    "crop": "...",
    "category": "...",
    "disease_or_pest": "...",
    "source_site": "..."
  },
  "gold_entities": ["西蓝花", "黑腐病"],
  "pseudo_evidence": [
    {
      "chunk_id": "...",
      "quote": "...",
      "score": 0.82
    }
  ],
  "pseudo_evidence_quality": 0.0-1.0
}
```

`gold_entities` 来源：

- `metadata.crop`
- `metadata.disease_or_pest`
- 从 `question + gold_answer` 中匹配 KG 实体词表、药剂词表、作物词表、病虫害词表

`pseudo_evidence` 构造：

1. 用当前 LightRAG 对 `question` 检索 top-k。
2. 用 `gold_answer` 与 chunk 做 embedding 相似度。
3. 保留相似度高、包含 gold_entities、或包含关键药剂/病虫害词的 chunk。
4. 计算 `pseudo_evidence_quality`，作为证据奖励软权重。

### Agent 轨迹输出

SFT 和 GRPO 都要求模型输出统一结构：

```json
{
  "plan": {
    "sub_questions": [],
    "tool_plan": ["kg_search", "vector_search"],
    "reason": ""
  },
  "actions": [
    {
      "tool": "kg_search",
      "query": "",
      "citations": []
    }
  ],
  "referee": {
    "evidence_sufficient": true,
    "missing_aspects": [],
    "risk": "low"
  },
  "answer": "",
  "citations": []
}
```

## 版本1：LLM Judge 奖励函数

### 总奖励

文本样本：

\[
R = 0.25R_{ans} + 0.25R_{faith} + 0.15R_{cov} + 0.20R_{agent} + 0.10R_{format} + 0.05R_{safety}
\]

图像样本：

\[
R = 0.20R_{ans} + 0.20R_{faith} + 0.10R_{cov} + 0.15R_{agent} + 0.15R_{mm} + 0.10R_{format} + 0.10R_{safety}
\]

证据相关项使用软权重：

\[
R_{faith}^{'} = q_e R_{faith} + (1-q_e)R_{no\_contradiction}
\]

\[
R_{cov}^{'} = q_e R_{cov} + (1-q_e)0.5
\]

其中 \(q_e\) 是 `pseudo_evidence_quality`。这样不做硬分桶，但不会强迫论坛答案必须被教材 KG 完全支持。

### 奖励项定义

**1. 答案正确性 \(R_{ans}\)**

\[
R_{ans}=0.3R_{entity}+0.3R_{sim}+0.4R_{judge\_ans}
\]

- `R_entity`：模型答案命中 `gold_entities` 的比例
- `R_sim`：模型答案与 `gold_answer` 的 embedding 相似度
- `R_judge_ans`：LLM Judge 对答案正确性的 1-5 分归一化

Judge 只判断“是否答对”，不判断证据。

**2. 事实一致性 \(R_{faith}\)**

\[
R_{faith}=0.3R_{rule\_support}+0.5R_{judge\_faith}+0.2R_{citation}
\]

- `R_rule_support`：答案中的作物、病虫害、药剂、时期是否出现在 evidence 中
- `R_judge_faith`：LLM Judge 判断答案是否被 evidence 支持
- `R_citation`：引用 chunk 是否真实存在且与答案相关

重点扣分：

- evidence 没有药剂但模型编药剂
- evidence 没有剂量但模型编剂量
- evidence 没有防治时期但模型编时期
- evidence 不支持诊断但模型强行诊断

**3. 证据覆盖率 \(R_{cov}\)**

\[
R_{cov}=\frac{|C_{model}\cap C_{pseudo}|}{|C_{pseudo}|}
\]

- `C_model`：模型引用或工具调用返回的 chunk
- `C_pseudo`：离线构造的 pseudo evidence chunk

若 `pseudo_evidence` 为空，`R_cov` 设为 0.5 中性分，不作为主要优化信号。

**4. Agent 行为质量 \(R_{agent}\)**

\[
R_{agent}=0.35R_{plan}+0.30R_{tool}+0.20R_{referee}+0.15R_{efficiency}
\]

- `R_plan`：LLM Judge 判断子问题是否覆盖原问题意图
- `R_tool`：工具选择是否合理
- `R_referee`：Referee 对证据是否充分的判断是否合理
- `R_efficiency`：工具调用次数是否过多、是否重复无效检索

推荐工具规则：

| 问题类型 | 推荐工具 |
|---|---|
| 病害/虫害诊断 | `kg_search` + `vector_search` |
| 药剂推荐 | `kg_search`，必要时 `cypher_query` |
| 栽培管理/土壤肥料 | `vector_search` |
| 图像问答 | `image_evidence` + `kg_search` |
| 证据不足 | `kg_search` + `referee` |

**5. 格式奖励 \(R_{format}\)**

- JSON 可解析：0.4
- 必填字段完整：0.3
- 工具名合法：0.2
- citations 格式合法：0.1

JSON 不可解析时，直接给总奖励 `-0.5`。

**6. 安全与谨慎性 \(R_{safety}\)**

LLM Judge + 规则共同判断：

- 农药建议是否谨慎
- 是否提示按标签/说明使用
- 是否避免编造剂量
- 证据不足时是否说明不确定

**7. 多模态一致性 \(R_{mm}\)**

只对 `metadata.images` 非空样本启用：

- 诊断结果是否与图片标签/图像描述一致
- 可见症状是否与回答一致
- 未看到图片证据时是否避免过度肯定

### 版本1训练方法

**SFT**

- 用 80%-90% 数据做 SFT。
- 用 LLM Judge 或强模型生成 `plan/actions/referee` 教师轨迹。
- `answer` 必须使用原始 `assistant.content`，不让教师模型重写事实。
- 样本权重：`confidence`、`quality_score` 高的样本权重大。

**GRPO**

- prompt：`question + metadata + 工具说明`
- 每个 prompt 采样 4-8 条轨迹
- 执行或模拟工具检索
- 调用 LLM Judge 计算语义奖励
- 组合规则奖励得到总分
- 先用 500-1000 条 RL prompt 小跑，reward 稳定后扩展到 2000-4000 条

适用场景：

- 有 API 可用
- 或本地能部署强 judge 模型，例如 32B/72B
- 更关注语义质量和复杂规划能力

## 版本2：纯离线奖励函数

### 总奖励

文本样本：

\[
R = 0.30R_{ans} + 0.20R_{faith\_rule} + 0.15R_{cov} + 0.20R_{agent\_rule} + 0.10R_{format} + 0.05R_{safety\_rule}
\]

图像样本：

\[
R = 0.25R_{ans} + 0.15R_{faith\_rule} + 0.10R_{cov} + 0.15R_{agent\_rule} + 0.15R_{mm\_rule} + 0.10R_{format} + 0.10R_{safety\_rule}
\]

同样使用 `pseudo_evidence_quality` 软权重，不做硬分桶。

### 奖励项定义

**1. 答案正确性 \(R_{ans}\)**

\[
R_{ans}=0.45R_{entity}+0.35R_{sim}+0.20R_{keyword}
\]

- `R_entity`：gold_entities 命中率
- `R_sim`：embedding 相似度
- `R_keyword`：关键词 F1

关键词包括：

- 作物名
- 病害/虫害名
- 药剂名
- 症状词
- 防治动作词
- 生长期/时期
- 部位词

**2. 规则事实一致性 \(R_{faith\_rule}\)**

\[
R_{faith\_rule}=1-P_{unsupported}
\]

扣分规则：

- 答案中的病害/虫害不在 `gold_entities` 或 evidence 中：扣 0.4
- 答案中的药剂不在 `gold_answer` 或 evidence 中：扣 0.3
- 答案中的剂量、倍液、浓度不在 `gold_answer` 或 evidence 中：扣 0.3
- 答案中的作物与 `metadata.crop` 明显冲突：扣 0.3
- 答案出现“保证治愈、一定有效、随便用”等绝对化表述：扣 0.2

最终截断到 0-1。

**3. 证据覆盖率 \(R_{cov}\)**

\[
R_{cov}=\frac{|C_{model}\cap C_{pseudo}|}{max(1, |C_{pseudo}|)}
\]

如果 pseudo evidence 质量低：

\[
R_{cov}^{'} = q_e R_{cov} + (1-q_e)0.5
\]

即证据质量越低，证据覆盖奖励越接近中性。

**4. 规则 Agent 质量 \(R_{agent\_rule}\)**

\[
R_{agent\_rule}=0.35R_{intent}+0.30R_{tool}+0.20R_{referee}+0.15R_{efficiency}
\]

- `R_intent`：子问题是否覆盖关键意图词
- `R_tool`：工具选择是否符合 category 规则
- `R_referee`：Referee 是否与证据质量一致
- `R_efficiency`：工具调用次数是否合理

按 `category` 的规则：

| category | 规划必须覆盖 | 推荐工具 |
|---|---|---|
| 病害防治 | 诊断对象、症状、用药/防治 | `kg_search`, `vector_search` |
| 虫害防治 | 虫害对象、危害、时期、防治 | `kg_search`, `vector_search` |
| 农药使用 | 药剂、对象、用法、安全限制 | `kg_search` |
| 栽培管理 | 操作建议、时间、作物阶段 | `vector_search` |
| 土壤肥料 | 肥料类型、施用方式、作物 | `vector_search` |
| 品种选择 | 作物、品种特性、适用场景 | `vector_search` |

**5. 格式奖励 \(R_{format}\)**

纯规则：

- JSON 合法：0.4
- 字段齐全：0.25
- `tool_plan` 是合法列表：0.15
- `actions` 工具名合法：0.1
- `answer` 非空且不超长：0.1

**6. 安全规则 \(R_{safety\_rule}\)**

- 农药问题未提示按标签/说明或咨询当地农技人员：扣 0.1
- 编造剂量/倍液：扣 0.3
- 使用绝对化保证：扣 0.2
- 建议高风险混配但 gold answer 未提及：扣 0.3
- 证据不足仍强答具体药剂：扣 0.3

**7. 多模态规则 \(R_{mm\_rule}\)**

若样本有图片：

- 回答中的病虫害名称命中 `metadata.disease_or_pest`：加分
- 回答中的作物命中 `metadata.crop`：加分
- 若图片问题但未调用 `image_evidence`：扣 0.4
- 若无图像描述却强行说“图片显示”：扣 0.3

### 版本2训练方法

**SFT**

- 不需要 LLM Judge。
- 用规则模板生成弱 Agent 轨迹：
  - 从 `category` 推断工具计划
  - 从 `question` 和 `answer` 抽取子问题
  - 从 `metadata` 构造 referee 结论
  - `answer` 使用原始答案
- 先训练模型稳定输出 Agent JSON 格式。

**GRPO**

- prompt：`question + metadata + 工具说明`
- 每个 prompt 采样 4-8 条轨迹
- 不调用任何 LLM Judge
- 只运行：
  - JSON 解析
  - 实体匹配
  - embedding 相似度
  - pseudo evidence 覆盖率
  - 规则扣分
- 完全离线可复现

推荐先做：

- 1000 条 prompt 验证 reward 是否有区分度
- 检查每组候选 reward 方差
- 若 reward 方差太小，提高格式和实体奖励权重
- 若模型开始投机复述 gold 风格但不检索，提高 `R_agent_rule` 和 `R_cov`

适用场景：

- 训练机器不能联网
- 不希望 reward 依赖 API 成本
- 需要可复现、可解释的训练流程

## 推荐训练顺序

1. **先做版本2**
   - 快速、离线、稳定
   - 能验证 Agent JSON、SFT、GRPO 流程是否跑通

2. **再做版本1**
   - 在版本2基础上加入 LLM Judge
   - 主要提升规划合理性、复杂语义判断和证据支持判断

3. **最终对比**
   - 当前 LightRAG
   - Planner-only AgenticRAG
   - SFT only
   - SFT + GRPO 版本2
   - SFT + GRPO 版本1

## Test Plan

固定测试集：

- 项目现有 `eval_dataset_200`
- 从论坛 QA 中留出 500 条
- 人工构造 50 条复杂病虫害多跳问题

指标：

- 答案正确率
- gold_entities 命中率
- embedding 相似度
- 证据覆盖率
- 格式合法率
- 工具调用成功率
- Referee 判断准确率
- 农药建议安全违规率
- 图像样本诊断一致性

## Assumptions

- 当前不做 KG 覆盖率硬分桶。
- 所有论坛 QA 的原始 `assistant.content` 都视为 gold answer。
- pseudo evidence 可能不完整，因此只作为软奖励，不作为唯一事实标准。
- 第一版统一训练一个 Agent 模型，不拆 Planner/Executor/Referee 多模型。
- 图像样本第一阶段不训练 VLM，只把图片信息转换为文本线索参与 Agent 训练。
