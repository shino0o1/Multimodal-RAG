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

我按你的要求做了一个默认映射假设：`no_kg -> naive`，`no_rag -> bypass`。如果你想改成别的映射（比如 no_kg 用 `mix`），我可以再改一版。

可直接运行示例：
```powershell
python .\scripts\eval_runner.py `
  --eval-file .\eval_dataset_200\Eval.jsonl `
  --rag-dir .\rag_storage_whole_book_gemini `
  --output-dir .\eval_results `
  --concurrency 6 `
  --system-model qwen3-8b `
  --judge-model qwen3.5-plus
```

消融示例：
```powershell
# 关闭 planner
python .\scripts\eval_runner.py --disable-planner

# 去 KG
python .\scripts\eval_runner.py --ablate-no-kg

# 去 RAG
python .\scripts\eval_runner.py --ablate-no-rag
```

python .\scripts\eval_runner.py `
  --eval-file .\eval_dataset_200\Eval.jsonl `
  --rag-dir .\rag_storage_whole_book_gemini `
  --output-dir .\eval_results `
  --concurrency 4 `
  --system-model qwen3.5-35b-a3b `
  --judge-model qwen3.5-plus `
  --disable-planner `
  --enable-judge