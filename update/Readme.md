## 启动方式
环境变量要求
export RAG_UI_EMBED_MODEL="Qwen/Qwen3-Embedding-4B"
export EMBEDDING_DIM=2560

**启动命令**

nohup python -u pdf_rag_pipeline.py > bulid_graph_whole_book.log 2>&1 &
python -m streamlit run ui/app.py


### 测试
Q1: 图中的害虫是什么？ 种植什么蔬菜时应该注意这种虫害？对于这种害虫在幼虫期防治时该如何用药？
Q2：我种植的青菜上面有好多这种虫，这是什么虫？  我该打什么农药啊？
Q3：我种的菜上面生虫了，被咬出很多孔，这是什么虫？ 该打什么药？


### 配置
- VLM-enhanced:额外尝试用检索到的库内图片增强最终回答
  配置：QUERY_VLM_ENHANCED
- top_K: 检索时返回的top_K条数据
  配置：QUERY_TOP_K
- enable_rerank 与 rerank model: 是否启用重排序功能，启用后会使用rerank model对检索结果进行重排序
  配置：QUERY_ENABLE_RERANK, RAG_MODEL_RERANK
- model_answer、model_planner、model_vision、model_image_description、model_embedding: 分别配置回答模型、规划模型、视觉模型、图片描述模型和向量化模型
  配置：RAG_MODEL_ANSWER(也用于知识图谱抽取), RAG_MODEL_PLANNER, RAG_MODEL_VISION, RAG_MODEL_IMAGE_DESCRIPTION, RAG_MODEL_EMBEDDING