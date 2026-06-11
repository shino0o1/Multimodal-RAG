## 需求
给我一版 eval_runner.py（读取 Eval.jsonl、调用当前系统、调用 Judge、输出总表和分组报告），测试时采用并发请求，并发数设为6。
这个问答系统中设置的大模型比较多，如 raganything\config.py 中的model_answer、model_planner(系统会对输入问题先进行规划)
我的想法是：评测时这些系统层面的模型统一用一个小参数模型，在eval_runner.py中设置，暂定为qwen3-8b，裁判模型也才eval_runner.py中设置，暂定为qwen3.5-plus，提示词也在eval_runner.py中写好，写明打分规则。
另外，我后续做一些消融实验，比如测试没有model_planner时系统的指标、去除知识图谱检索\RAG模块时的指标，为了方便做消融实验，我希望评测脚本中可以控制系统主要模块的开关。
有什么不确定的地方可以先问我

## 方案
已实现可直接用的评测脚本：[scripts/eval_runner.py](E:\Research\Project\RAG-Anything\scripts\eval_runner.py)

实现内容：
1. 读取 `Eval.jsonl`（支持文本/图片样本）。
2. 调当前系统（`ui/service.py`）做问答，支持并发，默认并发 `6`。
3. 系统侧模型统一覆盖为一个小模型（默认 `qwen3-8b`，覆盖 answer/planner/vision）。
4. Judge 模型独立配置（默认 `qwen3.5-plus`），并在脚本内内置了评分提示词和打分规则。
5. 输出总表和分组报告：
- `results.jsonl`
- `summary.json`
- `report_overall.csv`
- `report_by_task_type.csv`
- `report_by_modality.csv`
6. 支持消融开关：
- `--disable-planner`：关闭规划模块
- `--ablate-no-kg`：强制 `mode=naive`（近似去KG）
- `--ablate-no-rag`：强制 `mode=bypass`（去检索/RAG）
- --include-evidence：证据构建，--no-include-evidence（显式关闭，提速）

codex已经按我的要求做了一个默认映射假设：`no_kg -> naive`，`no_rag -> bypass`。

可直接运行示例：
```powershell
python .\scripts\eval_runner.py `
  --eval-file .\eval_dataset_200\Eval.jsonl `
  --rag-dir .\rag_storage_whole_book_gemini `
  --output-dir .\eval_results `
  --concurrency 6 `
  --system-model deepseek-ai/DeepSeek-V4-Flash `
  --judge-model deepseek-ai/DeepSeek-V4-Pro `
  --disable-planner `
  --enable-judge `
  --image-timeout-sec 180 `
  --resume `
  --save-every 1
```



消融示例：
```powershell
# 关闭 planner
python .\scripts\eval_runner.py --disable-planner

```
全量跑：
python .\scripts\eval_runner.py `
  --eval-file .\eval_dataset_200\Eval.jsonl `
  --rag-dir .\rag_storage_whole_book_gemini `
  --output-dir .\eval_results `
  --concurrency 4 `
  --system-model Qwen/Qwen3.5-9B `
  --judge-model deepseek-ai/DeepSeek-V4-Pro `
  --disable-planner `
  --enable-judge

随机抽样跑10条：
python .\scripts\eval_runner.py `
  --eval-file .\eval_dataset_200\Eval_rand10_img3.jsonl `
  --rag-dir .\rag_storage_whole_book_gemini `
  --output-dir .\eval_results
  --concurrency 4 `
  --system-model Qwen/Qwen3.5-9B `
  --judge-model deepseek-ai/DeepSeek-V4-Pro `
  --disable-planner `
  --enable-judge

## 评测的流程

1. 读取输入
- 读评测集：`eval_dataset_200/Eval.jsonl`
- 注册本地知识库：`rag_storage_whole_book_gemini`
- 输出目录：`eval_results`

2. 默认运行参数
- `mode=hybrid`
- `concurrency=6`
- `system_model=qwen3-8b`
- `judge_model=qwen3.5-plus`
- `planner=开启`（没传 `--disable-planner`）
- `enable_judge=关闭`（默认不用 Judge模型评分，这个一定要开）
- `disable_llm_cache=开启`（默认禁用查询缓存）

3. 初始化服务与模型路由
- 通过环境变量把系统侧模型（answer/planner/vision/image_desc）统一指向 `system_model`
- 创建 `RAGUIService`，注册已有 KB
- 因为默认 `enable_judge=false`，不会初始化 Judge 调用链

4. 执行评测（逐条样本）
- 先串行跑第1条做 warm-up
- 剩余样本并发执行（6线程）
- 每条样本调用系统问答：
  - 文本样本：`service.query(...)`
  - 图片样本：`service.query_with_image(...)`
- 如果想加快评测，评测时可设置 `include_evidence=False`，即**不做回答后的二次 context/citation 构建**
- Judge prompt 里会传 `retrieved_evidence`，但当前只传 chunk_id 列表，不传检索片段正文，见 [eval_runner.py](/e:/Research/Project/RAG-Anything/scripts/eval_runner.py:324)。
- 记录每条的：
  - `answer`
  - `latency_sec`
  - `query_time_sec`
  - 检索对齐指标（基于样本 evidence 的 chunk_id 与返回 citations 对比；若 citations 为空则通常为0）

5. 错误处理
- 某条失败不会中断全局，写入该条 `error`
- 所有失败样本汇总到：`eval_results/eval_errors.log`

6. 输出结果文件
- `eval_results/results.jsonl`：逐条结果
- `eval_results/summary.json`：总体统计与配置
- `eval_results/report_overall.csv`
- `eval_results/report_by_task_type.csv`
- `eval_results/report_by_modality.csv`
- `eval_results/eval_errors.log`（有错误时）



## 指标
**评测指标**

当前 [summary.json](/e:/Research/Project/RAG-Anything/eval_results/summary.json:42) 里主要有四类指标：

1. `timing`
   - `total_wall_time_sec`：整轮评测总耗时。
   - `avg_latency_sec`：成功样本平均端到端耗时。
   - `p95_latency_sec`：延迟 P95。

2. `counts`
   - `total`：总样本数。
   - `success`：无 `error` 的样本数。
   - `errors`：失败样本数。
   - `pass` / `pass_rate`：Judge 开启时才统计。通过条件是 `correctness >= 4`、`evidence_consistency >= 4`、`safety >= 4`。

3. `scores`
   - `avg_correctness`
   - `avg_evidence_consistency`
   - `avg_completeness`
   - `avg_clarity`
   - `avg_safety`
   - `avg_total_score`

    1) correctness：系统答案是否回答了问题且事实基本正确
    2) evidence_consistency：系统答案是否与给定证据一致，是否出现证据外硬性结论
    3) completeness：关键要点覆盖是否充分
    4) clarity：表达是否清晰、结构是否可读
    5) safety：是否存在明显误导或高风险建议

`total_score = correctness + evidence_consistency + completeness + clarity + safety`，见 [eval_runner.py](/e:/Research/Project/RAG-Anything/scripts/eval_runner.py:435) 。

  满分是 25 分。`avg_total_score` 是所有成功样本 `total_score` 的平均值，见 [eval_runner.py](/e:/Research/Project/RAG-Anything/scripts/eval_runner.py:675)。

4. `retrieval`
   - `avg_recall`
   - `avg_precision`
   - `avg_f1`

   单条 retrieval 指标在 [eval_runner.py](/e:/Research/Project/RAG-Anything/scripts/eval_runner.py:299) 算：
   - `gold_chunk_ids`：来自评测集样本里的 `evidence[].chunk_id`
   - `pred_chunk_ids`：来自系统返回的 `response["citations"][].chunk_id`
   - `hit = gold_ids ∩ pred_ids`
   - `recall = hit / len(gold_ids)`
   - `precision = hit / len(pred_ids)`
   - `f1 = 2PR/(P+R)`


## 历史问题

1. `retrieval` 三项全是 `0.0`
 `retrieval` 三项全是 `0.0`，直接原因是 `results.jsonl` 里的 `pred_chunk_ids` 为空。`pred_chunk_ids` 来自 `response["citations"]`，而 `citations` 是 `ui/service.py` 里通过额外拉取 context、提取 chunk_id、再和 `kv_store_text_chunks.json` 校验得到的；如果 context 里没有可识别 chunk_id，或提取失败，就会为空，见 [service.py](/e:/Research/Project/RAG-Anything/ui/service.py:1180) 和 [service.py](/e:/Research/Project/RAG-Anything/ui/service.py:1393)。