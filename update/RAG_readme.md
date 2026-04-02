# 当前 RAG 流程说明（基于本仓库当前代码）

本文档说明你现在这套 `RAG-Anything + LightRAG` 代码在查询阶段到底怎么做检索、检索哪些字段、以及如何定位到目标节点/关系。

## 1. 总体流程（查询时）

查询入口是 `rag.aquery(query, mode=...)`，默认模式是 `mix`。

主链路如下：

1. 用户问题进入 `RAGAnything.aquery()`。
2. 调用 `LightRAG.aquery_llm()`。
3. 对 query 做关键词抽取，得到：
   - 高层关键词 `hl_keywords`
   - 低层关键词 `ll_keywords`
4. 根据模式进入 KG 检索：`kg_query()`。
5. KG 检索分 4 阶段：
   - 阶段1：检索（实体/关系/向量块）
   - 阶段2：实体与关系 token 截断
   - 阶段3：关联 chunk 合并
   - 阶段4：构造最终上下文并喂给 LLM
6. 输出回答（或在 `only_need_context/only_need_prompt` 时只返回上下文/提示词）。

关键代码：
- `raganything/query.py` 的 `aquery()`
- `lightrag/lightrag.py` 的 `aquery_llm()`
- `lightrag/operate.py` 的 `kg_query()` / `_build_query_context()`

## 2. 检索哪些字段（你关心的 attributes / aliases 在哪里生效）

### 2.1 实体向量库 `entities_vdb` 的召回文本（最关键）

当前代码已经改为：实体向量召回时使用结构化 `content`，内容包含：

- `entity_id/entity_name`
- `entity_type`
- `description`
- `aliases`
- `attributes`
- `content_modality`（有则写）

也就是说，你说的 `attributes`、`aliases`、`entity_id` 现在会进入实体 embedding 文本，参与相似度召回。

实现位置：
- `/media/disk2/lhy/anaconda/envs/rag_anything/lib/python3.12/site-packages/lightrag/operate.py`
  - `_build_entity_vector_content(...)`
- `raganything/processor.py`
  - `_build_multimodal_entity_vector_content(...)`

### 2.2 关系向量库 `relationships_vdb` 的召回文本

关系向量 `content` 仍是如下结构：

- `keywords`
- `src_id`
- `tgt_id`
- `description`

形式大致是：
`keywords + src + tgt + description`

所以关系召回主要依赖关系关键词与描述。

### 2.3 文本块向量库 `chunks_vdb`

按 chunk 的 `content` 做向量召回：

- 文本 chunk：正文文本
- 多模态 chunk：模板化文本（图/表/公式 + 增强描述）

在 `mix` 模式中，chunk 向量召回会作为并行来源加入最终上下文。

## 3. 如何定位“目标节点/关系”

这里分为局部（实体）和全局（关系）两条路径：

### 3.1 实体定位（local 分支）

1. 用 `ll_keywords` 查询 `entities_vdb.query(...)`。
2. 从召回结果中拿 `entity_name`（这是节点主键）。
3. 用 `knowledge_graph.get_nodes_batch(entity_name列表)` 回图数据库取节点完整属性。
4. 再用这些节点扩展出相关边（`get_nodes_edges_batch` + `get_edges_batch`）。

结论：
- 向量侧决定“先命中哪些实体名”。
- 图侧再把这些实体名对应的节点属性取全。

### 3.2 关系定位（global 分支）

1. 用 `hl_keywords` 查询 `relationships_vdb.query(...)`。
2. 得到 `(src_id, tgt_id)` 对。
3. 用 `get_edges_batch` 回图数据库取边属性。
4. 再根据这些边反查相关实体节点。

结论：
- 先命中边，再反推出节点。

### 3.3 hybrid / mix 的合并方式

- `hybrid`：local + global 轮转合并实体/关系。
- `mix`：在 `hybrid` 基础上，再加入 `chunks_vdb` 的向量 chunk 召回结果。

## 4. 从节点/关系到证据 chunk 的过程

当实体和关系确定后，会进一步走 `source_id` 找原始 chunk：

1. 从实体/关系的 `source_id` 拆出 chunk_id。
2. 两种策略选 chunk（配置 `kg_chunk_pick_method`）：
   - `VECTOR`：按 query 与 chunk 向量相似度选
   - `WEIGHT`：按出现频次加权轮询
3. 与 mix 模式下的“纯向量 chunk召回”做轮转合并去重。
4. 如果启用 rerank，再重排。
5. 最后按 token 预算截断。

## 5. 最终送给 LLM 的上下文长什么样

最终上下文由三部分组成：

- Entities（实体列表）
- Relationships（关系列表）
- Text Chunks（证据文本）

并附带 references（引用文件列表）。

注意一个重要点：
- 虽然 `attributes/aliases` 已进入实体向量召回文本，
- 但最终给 LLM 的实体结构目前仍主要是 `entity/type/description/...` 这套字段；`attributes/aliases` 不是单独固定栏目输出（除非它们已经被写进 `description` 或你后续扩展了上下文构造）。

## 6. 如何调试“为什么命中这个节点/关系”

推荐直接用结构化接口而不是只看最终自然语言回答：

1. 用 `query_data` / `aquery_data`（`only_need_context=True` 也可）查看：
   - `data.entities`
   - `data.relationships`
   - `data.chunks`
   - `metadata.keywords`
2. 重点看：
   - 命中的 `entity_name` / `src_id~tgt_id`
   - 对应 `source_id` 指向了哪些 chunk
   - chunk 最终来自 `E`（实体）、`R`（关系）、`C`（纯向量）哪一路

这样可以准确回答“为什么这次命中的是它”。

## 7. 查询模式速查

- `local`：偏实体（ll_keywords -> entities_vdb）
- `global`：偏关系（hl_keywords -> relationships_vdb）
- `hybrid`：实体+关系
- `mix`：实体+关系+chunk向量（默认推荐）
- `naive`：只做 chunk 向量检索
- `bypass`：不检索，直接问 LLM

## 8. 与你当前改造相关的结论

1. 你要求的“把 `attributes/entity_id/aliases/description` 作为召回特征”已经在实体向量 `content` 侧生效。  
2. 你要求“非多模态节点与边的 description 也保留”，已生效（不再在质量层清空）。  
3. 如果你希望 `attributes` 直接出现在最终 LLM 实体上下文中，需要下一步改 `_apply_token_truncation()` / `_build_context_str()` 的实体序列化逻辑（当前是召回用了 attributes，但上下文展示不完整）。

