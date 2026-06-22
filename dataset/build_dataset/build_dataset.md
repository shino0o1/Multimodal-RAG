
## 背景
我从几个农业论坛上爬取了农业问答数据集(在：E:\Research\Project\RAG-Anything\dataset\merged_dataset_v2\qa_merged.jsonl)，想用来训练大模型。本来只想爬取十字花科蔬菜的病虫害问答和防治相关的问题的，但是发现数据有这些问题：
1. 一部分问答的主题和我预定的主题不一致，比如爬取的问答数据中有品种选择、气象灾害等和病虫害弱相关的主题
2. 经过筛选过滤后数据集数量不够，目前只有6961条，对于微调LLM来说或许有些少
3. 部分问答数据涉及的作物品种并不属于十字花科蔬菜

我现在想通过简单有效的方式来提高数据集质量和扩充数据集，但是我又不想再去重新爬取这些农业论坛了。
我想或许可以通过大模型造数据的方式来实现我的目标，这里可能需要大模型能够联网搜索获取准确的答案。

## 核心思路
做 **检索增强造数 + 严格过滤**。

1. **先做目标域重筛选**
   用 LLM 或规则把现有 6961 条重新打标签：
   `是否十字花科蔬菜`、`是否病虫害`、`是否防治相关`、`是否可训练`。
   品种选择、气象灾害、泛栽培管理可以先放到 `弱相关/不用作主训练`，不要混进核心 SFT。

2. **不要直接丢弃非十字花科数据**
   非目标作物的数据可以分三类：
   `完全丢弃`、`只保留通用病虫害方法论`、`作为负样本/分类样本`。
   比如玉米田除草剂这种不适合训练十字花科病虫害助手，但可以作为“识别非目标问题并谨慎拒答/转向”的样本。

3. **用“联网检索 + 证据生成”扩充**
   对每个目标作物和病虫害组合生成查询，例如：
   `甘蓝 霜霉病 防治`、`白菜 蚜虫 绿色防控`、`萝卜 黑腐病 发病条件`。
   然后让模型基于检索到的权威资料生成 QA，而不是自由发挥。优先来源：
   农技推广站、植保部门、农科院、农业农村部门、农药标签/登记信息、可信教材资料。

4. **扩充时按“缺口表”造数**
   先建一个覆盖矩阵：
   `作物 × 病害/虫害 × 任务类型`。
   当前任务类型包括：
   病虫害识别、症状识别、发生条件、防治时期、防治方法。
   哪些格子样本少，就只补哪些，避免重复造一堆相似问答。

5. **每条合成数据保留证据字段**
   建议新增 metadata：
   `evidence_urls`、`evidence_snippets`、`generation_method`、`is_synthetic`、`target_crop_ok`、`pest_disease_ok`。
   训练时不一定把证据给模型，但后续审计和过滤会很有用。

6. **质量门槛要比生成更重要**
   每条生成后再过一轮 judge：
   是否目标作物、是否病虫害防治、是否有证据支持、是否含危险用药建议、是否过度确定、是否和来源冲突、是否重复。

7. **数量目标别盲目做太大**
   6961 条如果质量高，其实可以先做 LoRA SFT。扩充可以先做到 **1万到1.5万条高质量样本**，不要为了数量塞入低质合成数据。合成数据占比建议先控制在 **30%-50%**，并单独标记，方便后续 ablation。

建议路线是：
**现有数据重筛选 → 建目标域覆盖表 → 对缺口联网检索权威资料 → 基于证据生成 QA → 二次质量审查 → 去重与分层采样 → 再训练。**

## 具体方案
建议按一条主线做：**先把现有 6961 条变干净，再只针对缺口做检索增强造数，最后单独做图片分支**。不要一开始就同时追求文本、图片、RL、全量扩充。

**实施方案**

1. **统一数据结构**
   给每条样本补齐这些字段：`crop`、`is_cruciferous`、`topic_type`、`is_pest_disease`、`is_control_related`、`evidence_urls`、`is_synthetic`、`image_paths`、`quality_decision`。

2. **现有数据重筛**
   第一轮只按“是否十字花科蔬菜相关”决定保留，不再按病虫害过滤。
   病虫害、栽培、施肥、缺素、除草、安全用药等只作为 `topic_type` 标签，后续训练时按任务选择子集。

3. **建覆盖矩阵**
   做一张表：`作物 × 病虫害 × 问题类型`。  
   作物先限定：白菜、甘蓝、花椰菜、青花菜、萝卜、芥菜、油菜等。  
   问题类型包括：病虫害识别、症状识别、发生条件、防治时期、防治方法。

4. **只对缺口检索资料**
   对覆盖不足的格子生成搜索 query，优先搜农技推广站、植保站、农科院、农业农村部门、农药登记/标签资料。  
   把网页正文、标题、URL、发布时间、图片 URL 缓存下来，形成 `evidence_cache`。

5. **基于证据生成 QA**
   让 LLM 只能根据检索证据生成问答。  
   每条生成样本都保留 `evidence_urls` 和 `evidence_snippets`。  
   生成类型以诊断识别、发生条件、防治时期和防治方法为主；安全用药提醒合并到防治方法答案中，不单独作为任务类型。

6. **严格过滤**
   再跑一轮 Judge，判断：
   是否目标作物、是否病虫害防治、是否证据支持、是否有危险用药、是否幻觉、是否重复、是否回答过度确定。  
   只把 `accept` 进训练集，`review` 单独留给人工抽查，不直接进训练。

7. **图片数据单独做分支**
   图片只从真实网页或原论坛图片来。  
   每张图绑定：`image_path`、`image_url`、`source_url`、`crop`、`disease_or_pest`、`caption/evidence_text`。  
   先生成“图片文字描述 + QA”，后续如果训练多模态模型，再保留真实图片路径。

8. **导出训练集**
   最终产物建议分成：
   `train_text_core.jsonl`：高质量真实 + 合成文本 QA  
   `train_negative.jsonl`：非目标问题拒答/转向样本  
   `train_image_text.jsonl`：图片转文字描述后的 QA  
   `eval_holdout.jsonl`：人工或高置信样本，绝不混入训练

**优先级待办**

P0：
- 修好当前数据文档和路径规范。
- 写现有 6961 条的重筛脚本。
- 明确十字花科蔬菜白名单和别名表。
- 只按十字花科相关性保留样本，病虫害相关性只做标签。

P1：
- 生成覆盖矩阵，统计哪些作物/病虫害/问题类型缺样本。
- 做检索证据缓存，不直接边搜边训练。
- 写“基于证据生成 QA”的 prompt 和输出 schema。
- 写二次 Judge 过滤和去重流程。

P2：
- 扩充到 1 万到 1.5 万条高质量文本样本。
- 加入非目标问题的拒答/转向样本。
- 做训练/验证/测试切分，保留独立评测集。

P3：
- 抽取已有图片样本。
- 从权威图文网页补充真实图片。
- 做图文一致性过滤。
- 生成图片描述版 QA，为后续多模态训练准备。

建议第一步先做 **现有 6961 条的重筛 + 覆盖矩阵**。这一步做完，才能知道到底缺什么，不然造数很容易重复和跑偏。

## 当前落地状态

已新增第一轮离线重筛脚本：

```powershell
python curate_existing_dataset.py `
  --input dataset\merged_dataset_v2\qa_merged.jsonl `
  --output-dir curated
```

脚本只做确定性筛选，不调用外部 LLM，也不联网。当前规则基于：

- 十字花科蔬菜白名单与别名归一化
- 只要识别为十字花科蔬菜相关，就进入 `cruciferous_all`
- 病害、虫害、防治、栽培、施肥、缺素、除草等只作为 `topic_type` 标签
- `quality_score`、`confidence`、高风险用药关键词只写入 `quality_flags`，不再决定是否保留

当前输出：

```text
dataset/build_dataset/curated/
├── all_labeled.jsonl
├── cruciferous_all.jsonl
├── cruciferous_pest_disease.jsonl
├── cruciferous_agronomy.jsonl
├── non_cruciferous.jsonl
├── invalid.jsonl
├── coverage_matrix.csv
└── summary.json
```

当前统计结果：

| 分桶 | 数量 | 用途 |
| --- | ---: | --- |
| `cruciferous_all` | 4763 | 所有十字花科蔬菜相关问答，默认保留 |
| `cruciferous_pest_disease` | 2147 | 十字花科病虫害相关派生子集 |
| `cruciferous_agronomy` | 2616 | 栽培、施肥、缺素、除草等派生子集 |
| `non_cruciferous` | 2198 | 非十字花科或未识别为十字花科的样本 |
| `invalid` | 0 | 空问答或非法 JSON |

十字花科相关样本中带图片的有 1992 条。`coverage_matrix.csv` 用来查看 `作物 × 主题 × 问题类型` 的缺口，后续检索增强造数只补缺口，不做无目标扩充。

## 下一步实施顺序

1. 根据 `coverage_matrix.csv` 生成缺口清单，只保留样本数小于 3 或小于 8 的组合。
2. 为缺口组合生成检索 query，建立 `evidence_cache`。
3. 基于证据生成合成 QA，再用二次 Judge 过滤。
4. 单独抽取 `cruciferous_all` 中带图样本，建立图文 manifest。
5. 最后再用自动报告抽样检查，不把人工抽查作为主流程依赖。

## 缺口查询生成

已新增缺口查询脚本：

```powershell
python build_gap_queries.py `
  --coverage curated\coverage_matrix.csv `
  --output-dir gaps `
  --min-count 3 `
  --top-n 120
```

当前输出：

```text
dataset/build_dataset/gaps/
├── gap_targets.csv
├── search_queries.jsonl
├── priority_gap_targets_top120.csv
├── priority_search_queries_top120.jsonl
└── summary.json
```

当前统计：

| 项目 | 数量 |
| --- | ---: |
| 全量缺口组合 | 759 |
| 完全空缺组合 | 443 |
| 样本不足组合 | 316 |
| 第一批优先检索组合 | 120 |

后续先使用 `priority_search_queries_top120.jsonl` 做小批量联网检索和证据缓存验证，不建议直接跑全量 759 个缺口。

## Target 归一化与复合拆分

`coverage_matrix.csv` 已改为按标准化 target 统计，不再直接使用原始 `metadata.disease_or_pest` 字符串。

归一化逻辑：

- 去掉 target 中的作物名前缀，例如 `白菜霜霉病` 归一为 `霜霉病`。
- 合并常见别名，例如 `吊丝虫`、`吊死虫`、`菜蛾` 归一为 `小菜蛾`；`黄条跳甲`、`跳甲` 归一为 `黄曲条跳甲`；`青菜虫` 归一为 `菜青虫`。
- 对括号内容同时识别，例如 `小菜蛾（吊丝虫）` 和 `吊死虫（小菜蛾）` 都归一为 `小菜蛾`。
- 归一结果写入每条样本的 `metadata.curation.normalized_targets`。

复合 target 拆分逻辑：

- 按 `、`、`,`、`，`、`;`、`；`、`/`、`+`、`和`、`及`、`与`、`或` 等分隔符拆分。
- 一条样本如果包含多个 target，会在覆盖矩阵中分别计数。
- 例如 `菜青虫，小菜蛾` 会同时计入 `菜青虫` 和 `小菜蛾`。
- 原始 `disease_or_pest` 为空时，才回退到 `topic_type`，如 `weak_agronomy`、`pesticide_use`。

这版修正后，覆盖矩阵行数为 2211，缺口组合从 949 降到 759。典型变化：

| 组合 | 当前计数 |
| --- | ---: |
| `大白菜 + 小菜蛾 + symptom_diagnosis` | 4 |
| `大白菜 + 小菜蛾 + chemical_control` | 21 |
| `大白菜 + 菜青虫 + chemical_control` | 41 |
| `甘蓝 + 霜霉病 + symptom_diagnosis` | 4 |
| `大白菜 + 软腐病 + chemical_control` | 142 |

## Evidence Cache 构建

已新增联网证据缓存脚本：

```powershell
python build_evidence_cache.py `
  --input gaps\priority_search_queries_top1000.jsonl `
  --output evidence_cache\search_results.jsonl `
  --max-results 5 `
  --max-snippets 4 `
  --max-images 0 `
  --timeout 20 `
  --delay 8 `
  --fallback-queries 4 `
  --retry-errors `
  --retry-no-results `
  --resume
```

脚本默认使用 DuckDuckGo HTML 搜索，不依赖搜索 API key。当前策略是先提高召回，再用后续质量过滤控制风险：

- 自动移除 `农技推广`、`植保站`、`农科院`、`农业农村` 等过强限定词。
- 自动生成短 query，例如 `作物 + 病虫害 + 任务词`、`作物 + 病虫害`。
- 自动扩展别名，例如 `青花菜 -> 西兰花`、`大白菜 -> 白菜`、`小菜蛾 -> 吊丝虫/吊死虫`。
- `--retry-errors` 会重试已有 `error` 行。
- `--retry-no-results` 会重试已有 `no_results` 行。
- `--resume` 会保留成功行，并替换被重试的失败行，避免重复追加。

每条输入 query 输出一行 JSON，包含：

- `query_id`、`crop`、`target`、`task_type`、`query`
- `search_attempts`，记录实际尝试过的搜索词
- 搜索结果标题、URL、搜索摘要
- 网页抓取状态、页面标题、正文证据片段
- 图片 URL 候选
- `source_score` 和 `source_type`，用于粗略标记更像权威来源的结果

已用 1 条样本 smoke test：

```powershell
python build_evidence_cache.py `
  --input gaps\priority_search_queries_top120.jsonl `
  --output evidence_cache\search_results.jsonl `
  --limit 1 `
  --max-results 2 `
  --max-snippets 2 `
  --max-images 3 `
  --timeout 12 `
  --delay 0
```

后续跑完整 Top 120 时使用 `--resume`，可以跳过已完成的 query。

## Evidence 二分类

已新增二分类脚本，只分 `usable` / `unusable`，不设置中间灰区：

```powershell
python score_evidence_cache.py `
  --input evidence_cache\search_results.jsonl `
  --output-dir evidence_cache\scored
```

策略是“只要不是明显不可用，就判为可用”。明显不可用包括：

- 搜索结果为空
- 没有任何可抓取正文片段
- 正文片段里找不到目标病虫害
- 正文片段太短
- 乱码严重

当前输出：

```text
dataset/build_dataset/evidence_cache/scored/
├── usable_evidence.jsonl
├── unusable_evidence.jsonl
├── all_scored_evidence.jsonl
└── evidence_quality_report.json
```

当前结果：

| 类别 | 数量 |
| --- | ---: |
| `usable` | 115 |
| `unusable` | 5 |
| 可用率 | 95.83% |

后续 LLM 构造 QA 只读取 `usable_evidence.jsonl`。`unusable_evidence.jsonl` 暂时跳过，后面如果数据仍不够，再单独重搜。

## 基于 Evidence 生成 QA

已新增 SiliconFlow 生成脚本：

```powershell
python generate_qa_from_evidence.py `
  --input evidence_cache\scored\usable_evidence.jsonl `
  --output synthetic_qa\generated_qa_raw_v2.jsonl `
  --train-output synthetic_qa\generated_qa_train_v2.jsonl `
  --qa-per-row 3 `
  --timeout 60 `
  --delay 0.5 `
  --workers 8 `
  --rate-limit-workers 2 `
  --temperature 0.4 `
  --dedup-threshold 0.88 `
  --model "deepseek-ai/DeepSeek-V4-Flash" `
  --base-url "https://api.siliconflow.cn/v1"
```

API Key 可二选一：

```powershell
$env:SILICONFLOW_API_KEY="你的 key"
```

也可以临时通过参数传入，但不建议将密钥写进脚本或文档：

```powershell
--api-key "YOUR_SILICONFLOW_API_KEY"
```

正式跑之前先 dry-run 检查 prompt 和证据是否正常：

```powershell
python generate_qa_from_evidence.py `
  --input evidence_cache\scored\usable_evidence.jsonl `
  --output synthetic_qa\generated_qa_dryrun.jsonl `
  --limit 1 `
  --dry-run `
  --delay 0
```

脚本输出两份文件：

- `generated_qa_raw.jsonl`：保留 query、证据和模型输出，便于追踪和后续质检。
- `generated_qa_train.jsonl`：直接可用于 SFT 的 `messages` 格式样本，字段尽量对齐原始数据：`messages`、`source`、`metadata.crop/category/disease_or_pest/images`。

当前任务类型固定为 5 类：`病虫害识别`、`症状识别`、`发生条件`、`防治时期`、`防治方法`。农业防治、生物防治、药剂防治统一归入 `防治方法`，不再单独拆分。

`安全用药` 不再作为独立任务类型生成，也不再单独建立缺口和搜索证据；相关安全提醒合并到 `防治方法` 答案约束中。

生成约束：

- 不再强制按检索任务类型生成问题。模型可以根据证据自由构造病虫害诊断、相似病虫害区分、防治建议、药剂用法和相关种植管理问题。
- 问题可以简洁、口语化、场景化或包含多个相关疑问，但问题中的作物、症状、地点、时间、药剂和剂量等事实必须有证据支持。
- `task_type` 是生成后的标签，只能从 5 类中选择；训练文件的 `metadata.crop/category/disease_or_pest` 按每条 QA 自己的字段写入。
- 同一批次会自动删除完全重复问题，并在相同 `crop + target + category` 内按 `--dedup-threshold` 删除高度相似问题。
- 答案中不使用“根据百度百科/根据资料/证据显示”等来源口吻。
- 文本 QA 的 `metadata.images` 暂时固定为 `[]`，图片数据后续单独清洗。
- 涉及用药时，除非证据明确且来源可靠，否则不输出精确剂量、倍液、浸种时间或安全间隔期，并提醒结合当地植保建议、登记范围和产品标签。

生成规则已经变化，第一次运行应使用新的输出文件且不要加 `--resume`。中断后继续同一批次时再加 `--resume`。

并发生成默认使用 8 个 worker。任一请求遇到 HTTP 429 限流后，脚本会保留正在执行的请求，并将后续并发上限降为 2。文件写入和全局去重仍在主线程中串行执行。

继续当前 v2 批次：

```powershell
python generate_qa_from_evidence.py `
  --input evidence_cache\scored\usable_evidence.jsonl `
  --output synthetic_qa\generated_qa_raw_v2.jsonl `
  --train-output synthetic_qa\generated_qa_train_v2.jsonl `
  --qa-per-row 3 `
  --timeout 60 `
  --delay 0.5 `
  --workers 8 `
  --rate-limit-workers 2 `
  --temperature 0.4 `
  --dedup-threshold 0.88 `
  --model "deepseek-ai/DeepSeek-V4-Flash" `
  --base-url "https://api.siliconflow.cn/v1" `
  --resume
```

python generate_qa_from_evidence.py `
>>   --input evidence_cache\scored\usable_evidence.jsonl `
>>   --output synthetic_qa\generated_qa_raw_v2.jsonl `
>>   --train-output synthetic_qa\generated_qa_train_v2.jsonl `
>>   --qa-per-row 3 `
>>   --timeout 60 `
>>   --delay 0.5 `
>>   --temperature 0.4 `
>>   --dedup-threshold 0.88 `
>>   --workers 8 `
>>   --rate-limit-workers 2 `
>>   --model "deepseek-ai/DeepSeek-V4-Flash" `
>>   --base-url "https://api.siliconflow.cn/v1" `
>>   --api-key "sk-qmzwznhribdoqbwizzthxdwrfmsyatawedxmvjzxbtqhufnk" `
>>   --resume

`--resume` 会跳过 raw 文件中已经成功或已判定为全重复的 `query_id`，重试 error，并继续向 v2 文件追加。若 train 文件因中断少写了数据，脚本会先根据 raw 中的 `training_rows` 自动重建。

已有训练数据可单独自动去重：

```powershell
python deduplicate_qa_dataset.py `
  --input synthetic_qa\generated_qa_train.jsonl `
  --output synthetic_qa\generated_qa_train_dedup.jsonl `
  --rejected-output synthetic_qa\generated_qa_train_duplicates.jsonl `
  --threshold 0.88
```


## QA 修复
不建议继续在现有 Evidence 上直接重写。当前 Evidence 评分策略过松：696条“可用”中有304条仅依赖搜索摘要，且607条存在正文抓取失败，这是错误和信息缺失的主要来源。

建议新建一条**QA修复流水线**：

1. **整理输入**
   - 使用 `curated/cruciferous_all.jsonl` 的4763条。
   - 加入 `generated_qa_train_v2_dedup.jsonl` 的1998条。
   - 不再加入旧版 `generated_qa_train.jsonl`。
   - 共约6761条，每条生成唯一 `qa_id` 并保留原问答。

2. **针对每条QA重新检索**
   - 从问题提取作物、病虫害、症状、时期、药剂等实体。
   - 生成2–4个不同查询。
   - 优先抓取完整正文，搜索摘要只能用于发现来源，不能单独支撑最终答案。
   - 相同主题共享 Evidence 缓存，减少搜索量。

3. **提高高风险信息门槛**
   - 症状、危害方式、发生条件：至少两个一致来源，或一个权威来源。
   - 药剂、剂量、用药时期：必须有登记标签、植保部门或可靠试验资料。
   - 地区性发生时期必须注明适用地区，不能泛化为全国规律。
   - 无法获得可靠证据时直接淘汰，不生成“资料未提供”式答案。

4. **LLM纠偏和补全**
   - 先拆分原答案中的事实主张。
   - 标记为“支持、矛盾、缺失、地区相关”。
   - 必要时针对缺失内容再检索一次。
   - 重写问题和答案，并重新标注 `category`。
   - 复合问题必须逐项回答。

5. **独立质量审查**
   - 使用第二次独立LLM调用检查准确性、完整性、问答匹配和用药安全。
   - 只分 `usable/unusable`。
   - 不通过时允许修复一次，仍不通过就淘汰。

6. **最终去重并导出**
   - 完全重复和近似重复去除。
   - 输出 `accepted.jsonl`、`rejected.jsonl`、证据缓存和质量报告。

实现上建议写一个统一的 `repair_qa_dataset.py`，内部完成“检索→补充检索→纠偏→审核→导出”，支持缓存、并发和 `--resume`。先自动跑200条验证通过率，再完整处理6761条。最终不足6000条时，再从已经验证过的高质量 Evidence 中扩充，而不是保留错误样本凑数量。

## 百炼联网纠偏脚本

> 当前采用该轻量化方案；上文较复杂的 evidence cache / 多阶段审查方案仅作为历史设计参考。

已新增一个轻量化脚本，仅处理已有 QA 数据文件，不再读取缺口 `gap-queries`，也不生成中间 evidence cache。

```bash
python dataset/build_dataset/repair_qa_with_bailian.py \
  --input dataset/build_dataset/synthetic_qa/generated_qa_train.jsonl \
  --input dataset/build_dataset/synthetic_qa/generated_qa_train_v2.jsonl \
  --output dataset/build_dataset/synthetic_qa/bailian_repaired_full_qa.jsonl \
  --change-log dataset/build_dataset/synthetic_qa/bailian_change_log.jsonl \
  --model qwen-plus
```

脚本默认读取环境变量：

```bash
export DASHSCOPE_API_KEY="你的百炼 API Key"
export BAILIAN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export BAILIAN_MODEL="qwen-plus"
```

输出只保留两个文件：

- `bailian_repaired_full_qa.jsonl`：纠偏后的完整 QA 数据集。
- `bailian_change_log.jsonl`：只记录被修改或删除的样本、原答案、新答案和修改原因。

脚本不会向训练样本新增 `evidence_urls`、`evidence_snippets`、`generation_method`、`is_synthetic` 等证据字段。百炼联网模型只在内部用于检查、纠偏和必要删除。纠偏规则重点包括：补足关键信息、修正回答方向不匹配、删除或重写无实质信息回答、防治时期尽量写到生育期或病虫发生期，以及药剂使用和药剂效果必须严格遵循联网搜索到的证据。

小批量验证可先使用：

```bash
python dataset/build_dataset/repair_qa_with_bailian.py \
  --input dataset/build_dataset/synthetic_qa/generated_qa_train.jsonl \
  --output /tmp/bailian_repaired_sample.jsonl \
  --change-log /tmp/bailian_change_sample.jsonl \
  --dry-run \
  --limit 20
```
