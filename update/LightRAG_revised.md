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