# AgenticRAG 的 SFT+GRPO 强化学习升级方案

## Summary

在 8000 条高质量 QA 和 2×A800 80GB 条件下，方案可以从“可行实验”升级为“完整训练闭环”：

1. **SFT 训练本地 Agent 模型**：先用 8000 条 QA 构造规划、检索、判断、回答轨迹，让模型学会 AgenticRAG 格式和农业病虫害问答能力。
2. **GRPO 强化规划与工具调用**：围绕答案正确性、KG事实一致性、证据覆盖率、多模态诊断一致性设计奖励，重点优化 Planner/Executor/Referee 行为。
3. **本地部署替代 API**：训练后用 vLLM/SGLang 部署本地模型，接入当前项目的 `RAG_MODEL_PLANNER`、`RAG_MODEL_ANSWER`。

推荐目标：训练一个 7B/14B 级别本地 Agent 模型，使其在当前病虫害问答系统中接近或超过大模型 API 的规划与证据约束能力。

## Key Changes

### 1. 模型与训练策略

默认推荐：

- 基座模型：`Qwen2.5-14B-Instruct` 或 `Qwen3-14B`
- 训练方式：LoRA SFT + GRPO
- 硬件：2×A800 80GB
- 训练框架：TRL / LLaMA-Factory / veRL 均可，推荐优先用 **LLaMA-Factory 做 SFT，TRL/veRL 做 GRPO**
- 部署：vLLM 或 SGLang，OpenAI-compatible API

可选更激进方案：

- 若追求更强能力，可尝试 `Qwen2.5-32B-Instruct` LoRA/QLoRA。
- 但第一版不建议直接上 32B，因为 RL 调参成本高，先用 14B 跑通闭环更稳。

### 2. 数据划分

8000 条 QA 建议这样划分：

- SFT 训练集：6800 条
- 验证集：600 条
- RL prompt 集：400 条
- 最终测试集：200 条

项目中的 215 条 `eval_dataset_200` 作为外部固定测试集，不参与训练、不参与奖励调参。

如果 8000 条里有图像样本：

- 文本 QA 和图像 QA 分层抽样；
- 保证测试集中保留足够图像问答；
- 图像样本先转成“图像描述/视觉线索 + QA”的文本训练形式，不直接微调多模态模型。

### 3. 训练样本格式

把原始 QA 字段：

```json
{
  "question": "",
  "answer": "",
  "crop": "",
  "category": "",
  "disease_or_pest": "",
  "confidence": 0.95,
  "question_clarity": 0.95,
  "answer_groundedness": 1.0,
  "image_relevance": null,
  "agricultural_correctness": 0.95,
  "safety": 1.0,
  "duplicate_risk": 0.1
}
```

转换为 Agent 轨迹格式：

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
      "evidence": []
    }
  ],
  "referee": {
    "evidence_sufficient": true,
    "missing_aspects": [],
    "risk": "low"
  },
  "answer": ""
}
```

SFT 阶段可用大模型 API 或本地强模型生成教师轨迹，但最终答案必须以原始 `answer` 为准，不能让教师模型重写事实。

## Reward Design

GRPO 总奖励：

\[
R = 0.30R_{ans} + 0.25R_{faith} + 0.20R_{cov} + 0.15R_{plan} + 0.10R_{format}
\]

多模态样本使用：

\[
R = 0.25R_{ans} + 0.25R_{faith} + 0.15R_{cov} + 0.15R_{mm} + 0.10R_{plan} + 0.10R_{format}
\]

各项定义：

- \(R_{ans}\)：答案正确性  
  用 gold answer 相似度、关键实体命中、LLM judge 综合评分。

- \(R_{faith}\)：KG事实一致性  
  判断答案中的作物、病害、虫害、药剂、症状是否被检索证据支持。证据外新增药剂、剂量、防治时期要扣分。

- \(R_{cov}\)：证据覆盖率  
  检索 evidence 与 gold QA 对应证据或知识库 top evidence 的覆盖程度：

\[
R_{cov} = \frac{|E_{retrieved} \cap E_{gold}|}{|E_{gold}|}
\]

如果训练 QA 没有 gold evidence，则先离线用当前 LightRAG 为每条 QA 建立 pseudo-gold evidence。

- \(R_{plan}\)：规划质量  
  子问题覆盖原问题核心意图，工具调用合理，不过度检索、不遗漏关键子任务。

- \(R_{format}\)：格式合法性  
  JSON 合法、字段完整、工具名合法、引用规范。

- \(R_{mm}\)：多模态诊断一致性  
  只在图像样本启用，判断诊断结果是否与图像标签、图像描述、`image_relevance` 一致。

## Implementation Changes

### 1. 数据准备

新增训练数据构造流程：

1. 读取 8000 条 QA。
2. 过滤 `decision != keep`、`confidence` 低、`safety` 低、`duplicate_risk` 高的样本。
3. 对每条 QA 调用当前 LightRAG 检索，生成 pseudo evidence。
4. 生成 SFT 轨迹：
   - 简单问题：1-2 个子问题；
   - 复杂问题：2-4 个子问题；
   - 防治/药剂问题必须检索药剂与适用对象；
   - 证据不足问题必须训练模型明确说明不足。
5. 生成 `sft_train.jsonl`、`sft_val.jsonl`、`rl_prompts.jsonl`、`test.jsonl`。

### 2. AgenticRAG 工作流升级

当前代码只有 Planner，且 `tool_plan` 固定为 `["kg"]`。需要升级为：

- Planner：输出子问题和工具计划；
- Executor：执行 `kg_search`、`vector_search`、`cypher_query`；
- Referee：判断证据是否充分；
- Answerer：基于证据回答并给出引用。

第一版可以仍用一个模型完成四个角色，但代码层要把轨迹拆开记录，便于 RL 采样和奖励计算。

### 3. SFT

训练目标：

- 学会输出 Agent 轨迹；
- 学会农业病虫害领域回答；
- 学会证据不足时拒答；
- 学会不编造药剂、剂量、时期。

推荐配置：

- 模型：14B
- 方法：LoRA
- epoch：2-3
- max length：4096 或 8192
- learning rate：1e-5 到 2e-5
- eval every 200-500 steps

### 4. GRPO

GRPO 输入只使用问题、可选图像描述、工具说明，不直接给 gold answer。

每个 prompt 采样 4-8 条候选轨迹，执行工具检索后计算 reward。

重点优化：

- 是否拆对问题；
- 是否选对检索工具；
- 是否召回支持答案的证据；
- 是否在证据不足时拒答；
- 是否避免证据外发挥。

推荐先用 400 条 RL prompt 跑小规模 GRPO，确认 reward 有区分度后扩展到 1000-2000 条。

## Test Plan

必须做四组对比：

1. 当前 LightRAG 直接问答；
2. 当前 Planner-only AgenticRAG；
3. SFT 模型；
4. SFT + GRPO 模型。

测试集：

- 8000 QA 中独立留出的 200 条；
- 项目现有 `eval_dataset_200` 的 215 条；
- 人工构造 50 条复杂多跳问题。

核心指标：

- 答案正确率；
- KG事实一致性；
- 检索证据覆盖率；
- 图像诊断一致性；
- 证据不足拒答准确率；
- 平均工具调用次数；
- 平均延迟；
- 与 API 大模型的效果差距。

## Assumptions

- 8000 条 QA 全部准确，但仍使用字段分数做训练权重。
- `confidence`、`answer_groundedness`、`agricultural_correctness` 可作为 SFT sample weight 和 reward 校准信号。
- 当前 215 条项目数据只作为测试集。
- 第一版不训练视觉模型，只训练文本 Agent；图像信息通过图像描述进入模型。
- 先训练统一 Agent 模型，不拆 Planner/Executor/Referee 多个模型。

# QA
## Planner-Executor-Referee是否共用一个模型，是否进行相同的SFT\GRPO？
建议第一版：**三个 Agent 用同一个模型，但不要做完全相同的数据和奖励训练**。

更具体地说：
**1. 是否用同一个模型？**
第一阶段建议用 **同一个本地模型** 承担 Planner、Executor、Referee、Answerer 四类能力。
原因是：
- 你现在的数据规模是 8000 条 QA，不算少，但还不足以稳定训练多个专门 Agent；
- 当前系统代码也还没有完整拆出 Executor 和 Referee 工作流；
- 多模型会增加部署复杂度、显存占用和推理延迟；
- 一个 14B 模型经过 SFT+GRPO 后，先学会统一的 AgenticRAG 轨迹更现实。

也就是训练一个：

```text
Agricultural-AgenticRAG-14B
```

让它根据提示词中的 role 执行不同职责：

```text
你现在是 Planner Agent...
你现在是 Executor Agent...
你现在是 Referee Agent...
你现在是 Answerer...
```

**2. 是否进行相同的 SFT？**

不是“完全相同”，而是做 **统一 SFT，多任务样本混合训练**。

SFT 数据应拆成几类：

| 样本类型 | 训练目标 |
|---|---|
| Planner 样本 | 学会问题拆解、子问题生成、工具计划 |
| Executor 样本 | 学会根据计划选择检索工具、生成检索 query |
| Referee 样本 | 学会判断证据是否充分、是否需要重检索 |
| Answerer 样本 | 学会基于证据回答，不编造 |
| End-to-end 轨迹样本 | 学会完整 plan-retrieve-judge-answer 流程 |

也就是说，模型相同，但训练样本里要明确区分任务角色。

**3. 是否进行相同的 GRPO？**

也不建议“相同 GRPO”。GRPO 应该以 **端到端轨迹奖励为主**，同时包含不同 Agent 行为的子奖励。

比如总奖励：

\[
R = 0.25R_{plan} + 0.20R_{tool} + 0.20R_{faith} + 0.25R_{ans} + 0.10R_{format}
\]

其中：

- \(R_{plan}\)：Planner 是否拆对问题；
- \(R_{tool}\)：Executor 是否选对工具、检索 query 是否有效；
- \(R_{faith}\)：答案是否被 KG 证据支持；
- \(R_{ans}\)：最终答案是否正确；
- \(R_{format}\)：输出格式是否合法。

也就是说，GRPO 不应该分别训练三个互不相关的 Agent，而是训练模型在完整 AgenticRAG 流程中的整体决策能力。

**推荐训练路线**

```text
阶段 1：统一模型 SFT
目标：学会 Planner / Executor / Referee / Answerer 的格式和基本行为

阶段 2：端到端 GRPO
目标：优化完整轨迹的最终效果，尤其是规划、检索和证据一致性

阶段 3：可选角色强化
如果发现某个环节弱，再做专项数据增强
例如 Planner 弱，就增加问题拆解样本
Referee 弱，就增加证据充分性判断样本
```

**什么时候才考虑多个模型？**

后续满足这些条件时，可以拆：

| 拆分方式 | 适用条件 |
|---|---|
| Planner 单独模型 | 规划错误是主要瓶颈 |
| Referee 单独模型 | 需要更强事实校验和拒答能力 |
| Executor 单独模型 | Cypher/工具调用复杂度明显上升 |
| Answerer 单独模型 | 最终表达质量要求很高 |

但当前最稳妥的是：

```text
一个 14B 统一 Agent 模型
+ 角色提示词
+ 多任务 SFT
+ 端到端 GRPO
```

结论：**用同一个模型；SFT 是统一多任务训练；GRPO 是端到端轨迹训练，不是三个 Agent 各训一套完全相同流程。**