# 本地知识库驱动的评测集自动构建 Pipeline - codex计划

## Summary

构建一个命令行 pipeline，根据现有 `rag_storage_*` 知识库自动生成约 `200` 条高质量评测样本，覆盖病虫害诊断、防治建议、药剂推荐、症状识别、图像问答。所有 `gold_answer` 和 `evidence` 必须可追溯到本地知识库，不使用网络或外部知识。图像问答使用你额外提供的图片集 manifest，并通过 VLM 识别结果映射回本地 KG/文本证据。

默认产物：
- `eval_dataset/candidates.jsonl`：全部候选样本
- `eval_dataset/accepted.jsonl`：通过质检的评测集
- `eval_dataset/rejected.jsonl`：被过滤样本及原因
- `eval_dataset/report.json`：覆盖率、分布、长尾、质检统计
- `eval_dataset/review_sheet.csv`：人工审核表

## Key Changes

实现一个命令行脚本，例如：

```bash
python scripts/build_eval_dataset.py \
  --rag-dir rag_storage_whole_book_gemini \
  --image-manifest eval_images/manifest.jsonl \
  --output-dir eval_dataset \
  --target-size 200 \
  --seed 42
```

新增核心能力：
- 读取 `graph_chunk_entity_relation.graphml`、`kv_store_text_chunks.json`、`kv_store_entity_chunks.json`、`kv_store_relation_chunks.json`、`kv_store_multimodal_desc_cache.json`。
- 构建 evidence packs：以实体、关系、chunk、图片标注为中心组织证据。
- 调用大模型生成真实用户风格 query、gold answer、评分字段。
- 用规则、证据匹配、去重、分布约束、LLM judge 做中等强度质检。
- 用分层采样避免样本集中在少数热门实体或单一问题类型。

## Dataset Schema

每条样本使用 JSONL，一行一个对象：

```json
{
  "id": "diag_0001",
  "task_type": "病虫害诊断",
  "modality": "text",
  "question": "甘蓝叶片出现多角形黄褐色病斑，潮湿时叶背有霉层，可能是什么病害？",
  "image_path": null,
  "gold_answer": "可能为十字花科蔬菜霜霉病，应结合叶片多角形病斑、潮湿时灰白色霉层等症状判断。防治上可清除病残体、轮作、深沟高畦、合理密植，并按知识库证据选用登记药剂。",
  "expected_entities": ["霜霉病", "甘蓝", "叶片"],
  "expected_relations": ["危害", "症状", "防治"],
  "evidence": [
    {
      "source_type": "text_chunk",
      "chunk_id": "chunk-xxx",
      "file_path": "十字花科蔬菜病虫害_3.pdf",
      "quote": "病叶表面初期出现..."
    }
  ],
  "must_include": ["霜霉病", "叶片病斑", "防治措施"],
  "must_not_include": ["无法从证据支持的药剂", "确定为非证据病害"],
  "difficulty": "medium",
  "quality": {
    "rule_score": 0.92,
    "evidence_score": 0.88,
    "judge_score": 4,
    "status": "accepted"
  }
}
```

图像 manifest 推荐格式：

```json
{
  "image_path": "eval_images/001.jpg",
  "labels": {
    "crop": "甘蓝",
    "target": "菜青虫",
    "symptoms": ["叶片孔洞", "幼虫取食"]
  },
  "notes": "田间拍摄，可见叶片被取食"
}
```

图像样本生成逻辑：
- 先用 VLM 基于图片和 manifest 生成视觉线索。
- 将 `target/crop/symptoms` 映射到本地 KG 实体和相关 chunks。
- 只有映射到本地证据的图片才进入候选集。
- gold answer 只能引用本地 evidence pack，不允许只依赖图片识别常识。

## Generation Strategy

目标规模默认 `200`，先生成 `260-300` 条候选，再过滤到 accepted。

默认配额：
- 病虫害诊断：45
- 防治建议：45
- 药剂推荐：35
- 症状识别：35
- 图像问答：30
- 证据不足/不确定性问题：10

采样约束：
- 单个核心实体最多占 `8` 条 accepted 样本。
- 单个 task_type 内，同一实体最多占该类型 `20%`。
- 药剂类问题必须包含用药注意事项或适用对象，不生成孤立“推荐某药剂”样本。
- 诊断类问题优先采样“症状 + 作物 + 部位 + 环境/时期”的组合。
- 防治类问题优先覆盖农业防治、物理防治、生物防治、化学防治组合。
- 证据不足类问题要求 gold answer 明确说明“仅凭现有证据无法确定”。

## Quality Checks

全量规则检查：
- JSON schema 完整性。
- `question/gold_answer/evidence` 非空。
- `expected_entities` 至少一个能在 KG 中找到。
- `evidence.chunk_id` 必须存在于 `kv_store_text_chunks.json`。
- `quote` 必须能在对应 chunk 中模糊匹配。
- `gold_answer` 不得包含 evidence 中没有支撑的具体药剂、剂量、病虫害名称。

去重检查：
- 问题文本相似度去重。
- gold answer 相似度去重。
- 同一 evidence pack 生成样本数量设上限。
- 同一图片最多生成 `1-2` 条 accepted 图像问答。

分布检查：
- 输出 task_type、difficulty、entity_type、核心实体、证据 chunk 的分布报告。
- 对过度集中的实体自动降采样。
- 对覆盖不足的类型自动补采候选。

LLM judge：
- 对通过规则检查的候选进行中等强度裁判。
- 裁判只接收 `question + gold_answer + evidence`，不得使用外部知识。
- 输出 `answer_correctness`、`evidence_consistency`、`safety`、`clarity` 四项 1-5 分。
- 默认接受条件：`evidence_consistency >= 4` 且 `safety >= 4` 且总评不低于 `16/20`。
- 低分样本写入 `rejected.jsonl`，保留拒绝原因。

## Implementation Notes

建议新增：
- `scripts/build_eval_dataset.py`：CLI 入口。
- `raganything/eval_dataset/builder.py`：pipeline 编排。
- `raganything/eval_dataset/loaders.py`：读取 KG、chunks、图片 manifest。
- `raganything/eval_dataset/generators.py`：prompt 和 LLM 调用。
- `raganything/eval_dataset/validators.py`：规则、证据、一致性、分布检查。
- `raganything/eval_dataset/reports.py`：统计报告输出。

模型调用复用当前项目风格：
- 使用 `openai_complete_if_cache`。
- 默认从 `RAGAnythingConfig` 和环境变量读取模型名、`OPENAI_API_KEY`、`OPENAI_BASE_URL`。
- 支持 `--generator-model`、`--judge-model`、`--vision-model` 覆盖。
- 默认不调用网络搜索。

## Test Plan

最小可验证测试：
- 用 `rag_storage_whole_book_gemini` 生成 `--target-size 20` 的小样本集。
- 验证输出文件存在且 JSONL 每行可解析。
- 验证 accepted 样本全部有合法 evidence chunk。
- 验证 report 中 task_type 分布、实体分布、拒绝原因统计正常。

功能测试：
- 无 image manifest 时，只生成文本类样本并跳过图像问答。
- image manifest 中图片无法映射到 KG 时，该图片样本进入 rejected。
- 人为构造不存在 chunk_id 的样本，validator 必须拒绝。
- 同一实体过多时，分布控制必须降采样。

人工抽检：
- 从 accepted 中每类抽 `5` 条，看问题是否自然、gold answer 是否准确、证据是否支撑。
- 从 rejected 中抽 `10` 条，确认拒绝原因合理。

## Assumptions

- 默认知识库目录使用 `rag_storage_whole_book_gemini`。
- gold answer 严格只基于本地 KB，不引入模型常识和网络知识。
- 图像问答依赖用户提供 `manifest.jsonl`，每张图片至少包含 `image_path` 和一个可映射到 KG 的 `target/crop/symptom` 标签。
- 默认目标为 `200` 条 accepted 样本，实际会先生成更多候选再筛选。
- 初版只做命令行工具，不接入 Streamlit 前端。

# 评测集构建流程
现在 pipeline 流程是：

1. 读取本地知识库  
从 `rag_storage_whole_book_gemini/` 读取：

```text
graph_chunk_entity_relation.graphml
kv_store_text_chunks.json
kv_store_entity_chunks.json
kv_store_relation_chunks.json
kv_store_multimodal_desc_cache.json
```

2. 构建证据包  
把 KG 里的实体、关系、chunk 组合成 `evidence pack`，例如：

```text
核心实体：霜霉病
相关实体：甘蓝、叶片
关系：危害、症状、防治
证据 chunk：chunk-xxx
证据原文：病叶表面出现...
```

3. 分类型采样  
按任务类型抽样：

```text
病虫害诊断
防治建议
药剂推荐
症状识别
图像问答
证据不足
```

4. 调用大模型生成 QA  
RAGAnythingConfig.model_answer，现在是gemini-2.5-flash

给模型输入：
```text
任务类型 + 核心实体 + 相关实体 + 关系 + 本地证据原文
```

让模型生成：

```text
question
gold_answer
expected_entities
expected_relations
must_include
must_not_include
difficulty
```

要求模型只能基于本地证据回答。

**生成评测集命令：**
python scripts/build_eval_dataset.py \
  --rag-dir rag_storage_whole_book_gemini \
  --image-manifest raganything/eval_dataset/manifest.jsonl \
  --output-dir eval_dataset_test \
  --target-size 20
这里的target-size大约是最终accepted的样本数量，实际生成时会先生成更多候选再过滤。

**eval_dataset_test 常见文件作用：**
candidates.jsonl：通过基础流程得到的候选样本（含质量分）。
accepted.jsonl：最终通过并入选的评测集。
rejected.jsonl：被拒绝样本及原因。
report.json：整体统计（数量、接受率、分布、拒绝原因汇总等）。
review_sheet.csv：给人工抽检/复核用的表格视图。

5. 图像问答  
如果提供 `image-manifest`：

```text
图片 + crop/target/symptoms 标签
```

pipeline 会先把图片标签映射到本地 KG 实体，再用对应本地证据生成图像问答。

6. 质量检查  
自动检查：

```text
字段是否完整
证据 chunk 是否存在
quote 是否能在 chunk 中找到
实体是否在 KG 中
是否重复
是否某个实体占比过高
药剂建议是否有注意事项
```

7. LLM 裁判  
现在是gemini-3.1-pro-preview
再让模型按证据打分：

```text
答案正确性
证据一致性
安全性
表达清晰度
```

不过裁判也只能看本地证据。

接受/拒绝逻辑（核心）：

先做规则校验：字段完整性、evidence chunk 是否存在、quote 能否在 chunk 中匹配、must_include/must_not_include 等。
再做 LLM 裁判打分：answer_correctness / evidence_consistency / safety / clarity。
满足阈值才进入有效候选，再按分数排序取前 target-size；其余进 rejected（含拒绝原因，如 judge_rejected、invalid_chunk、image_target_mismatch、over_target_size 等）。


8. 输出结果  

```text
eval_dataset/
  candidates.jsonl   # 所有候选
  accepted.jsonl     # 通过质检的最终评测集
  rejected.jsonl     # 被过滤的样本
  report.json        # 分布和质量报告
  review_sheet.csv   # 人工审核表
```

一句话：它不是凭空造题，而是先从 KG 和 chunk 里找证据，再围绕证据生成问题、标准答案和评测字段，最后自动过滤低质量和分布不均的样本。