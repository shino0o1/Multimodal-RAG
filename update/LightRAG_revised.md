LightRAG库中进行知识抽取、关系抽取和知识图谱构建的prompt位置：

1. 知识抽取/实体抽取 + 关系抽取的 prompt 定义在  
[**prompt.py**](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/prompt.py)：
- `PROMPTS["entity_extraction_system_prompt"]`（实体+关系抽取总规则）  
  [prompt.py:11](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/prompt.py:11)
- `PROMPTS["entity_extraction_user_prompt"]`（首轮抽取 user prompt）  
  [prompt.py:63](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/prompt.py:63)
- `PROMPTS["entity_continue_extraction_user_prompt"]`（补抽/纠错 prompt）  
  [prompt.py:84](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/prompt.py:84)
- 例子在 `PROMPTS["entity_extraction_examples"]`  
  [prompt.py:102](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/prompt.py:102)

2. 这些抽取 prompt 的实际调用位置在  
[**operate.py**](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/operate.py) 的 `extract_entities()`：
- 组装并调用 system/user prompt  
  [operate.py:2813](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/operate.py:2813)  
  [operate.py:2881](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/operate.py:2881)  
  [operate.py:2892](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/operate.py:2892)

3. 知识图谱“构建/入库”本身基本不是 prompt 驱动，而是程序化 merge + upsert：
- 主流程：`merge_nodes_and_edges()`  
  [operate.py:2443](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/operate.py:2443)
- 节点合并入图：`_merge_nodes_then_upsert()` → `upsert_node`  
  [operate.py:1613](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/operate.py:1613)  
  [operate.py:1891](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/operate.py:1891)
- 关系合并入图：`_merge_edges_then_upsert()` → `upsert_edge`  
  [operate.py:1918](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/operate.py:1918)  
  [operate.py:2380](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/operate.py:2380)

补充一点：图谱构建阶段唯一明显用到的 LLM prompt 是“描述合并总结”：
- `PROMPTS["summarize_entity_descriptions"]`  
  [prompt.py:184](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/prompt.py:184)
- 在 `_summarize_descriptions()` 中调用  
  [operate.py:297](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/operate.py:297)


  当前 `pdf_rag_pipeline.py` 构图阶段主要用了两类 prompt：

1. 多模态解析 prompt（`set_prompt_language("zh")` 后走中文模板）
- 图像：`IMAGE_ANALYSIS_SYSTEM` + `vision_prompt/vision_prompt_with_context`  
  [prompts_zh.py:17](/media/disk2/wzh/RAG-Anything/raganything/prompts_zh.py#L17)  
  [modalprocessors.py:907](/media/disk2/wzh/RAG-Anything/raganything/modalprocessors.py#L907)
- 表格：`TABLE_ANALYSIS_SYSTEM` + `table_prompt/table_prompt_with_context`  
  [prompts_zh.py](/media/disk2/wzh/RAG-Anything/raganything/prompts_zh.py)  
  [modalprocessors.py:1106](/media/disk2/wzh/RAG-Anything/raganything/modalprocessors.py#L1106)
- 公式：`EQUATION_ANALYSIS_SYSTEM` + `equation_prompt/equation_prompt_with_context`  
  [modalprocessors.py:1299](/media/disk2/wzh/RAG-Anything/raganything/modalprocessors.py#L1299)

2. 实体关系抽取 prompt（LightRAG）
- `entity_extraction_system_prompt`
- `entity_extraction_user_prompt`
- `entity_continue_extraction_user_prompt`  
  [operate.py:2894](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/operate.py#L2894)  
  [operate.py:2898](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/operate.py#L2898)  
  [operate.py:2901](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/operate.py#L2901)  
  Prompt定义在：  
  [lightrag/prompt.py:11](/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/prompt.py#L11)

你说的“大猿叶甲成虫”和“大猿叶甲”并存，根因是：
- 当前规范化只清理 `(image)` 这类后缀，不会把“成虫/幼虫/卵/蛹”等生命周期词从虫害名中拆出来。  
  [_clean_suffix in kg_quality.py:115](/media/disk2/wzh/RAG-Anything/raganything/kg_quality.py#L115)
- 所以模型一旦把“成虫”并到实体名里，就会被当成另一个虫害节点。

建议这样解决（效果最好）：
1. 在 `entity_extraction_*` prompt 增加硬约束：生命周期词不能并入虫害主实体名；应拆成“虫害实体 + 生长期实体”，并用 `发生于/生命周期` 关系连接。  
2. 在 `kg_quality.py` 增加规则归一化：若 `entity_type=="虫害"` 且名称以 `成虫|幼虫|若虫|卵|蛹` 结尾，canonical 名去后缀，原名进 `aliases`。  
3. 对已生成图谱执行一次清洗重写（`clean_graphml_file`），把旧图中的这类节点并回主实体。  

如果你要，我可以直接给你把第 1+2 点代码改好（含中文阶段词表和正则），然后你重跑一次构图即可。