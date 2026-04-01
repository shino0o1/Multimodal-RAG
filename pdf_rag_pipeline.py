import asyncio
import os
from raganything import RAGAnything, RAGAnythingConfig, set_prompt_language
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.prompt import PROMPTS as LIGHTRAG_PROMPTS


KG_ENTITY_TYPES = [
    "虫害",
    "病害",
    "作物",
    "病原菌",
    "药剂",
    "生长期",
    "生物分类",
    "部位",
    "时间",
    "形态特征",
    "危害症状",
    "发病诱因",
    "发生时期",
    "防治要点",
]

KG_RELATION_ENUM = [
    "致病",
    "发生于",
    "使用药剂",
    "防治",
    "症状",
    "位于",
    "影响",
    "生命周期",
    "属于",
]


def patch_lightrag_extraction_prompts() -> None:
    """Patch LightRAG extraction prompts so outputs align with KG ontology."""
    relation_text = "、".join(KG_RELATION_ENUM)
    extraction_prompt = f"""---Task---
从输入文本中抽取实体与关系。

---Hard Constraints---
1. 实体类型只能使用 <Entity_types> 中给出的类型。
2. 关系关键词只能使用以下枚举：{relation_text}
3. 关系关键词可多选，多个关键词用英文逗号分隔。
4. 严格输出格式，不要输出解释说明：
   - entity<|#|>entity_name<|#|>entity_type<|#|>entity_description
   - relation<|#|>source_entity<|#|>target_entity<|#|>relationship_keywords<|#|>relationship_description
5. 最后一行必须输出 <|COMPLETE|>。

---Data to be Processed---
<Entity_types>
{{entity_types}}

<Input Text>
{{input_text}}

<Output>"""

    continue_prompt = f"""---Task---
基于上一轮结果，补充遗漏或格式错误的实体与关系。

---Hard Constraints---
1. 仅输出新增或修正项，不要重复已正确项。
2. 实体类型仍只能使用 <Entity_types>。
3. 关系关键词仍只能使用：{relation_text}
4. 严格输出格式：
   - entity<|#|>entity_name<|#|>entity_type<|#|>entity_description
   - relation<|#|>source_entity<|#|>target_entity<|#|>relationship_keywords<|#|>relationship_description
5. 最后一行必须输出 <|COMPLETE|>。

---Data to be Processed---
<Entity_types>
{{entity_types}}

<Input Text>
{{input_text}}

<Output>"""

    loop_prompt = """请判断是否仍有遗漏的实体或关系。
如果有遗漏，仅输出 yes；
如果没有遗漏，仅输出 no。"""

    LIGHTRAG_PROMPTS["entity_extraction"] = extraction_prompt
    LIGHTRAG_PROMPTS["entity_continue_extraction"] = continue_prompt
    LIGHTRAG_PROMPTS["entity_if_loop_extraction"] = loop_prompt

async def main():
    # ==========================================
    # 1. 环境与模型配置
    # ==========================================
    api_key = "sk-MwcAPesgu8ol4F0ePPNP0hkGiseYaNEbfoLv4phN03ldl3AV" # 替换为您的 API Key
    base_url = "https://yunwu.ai/v1" # 如果使用代理可以修改此处
    set_prompt_language("zh")
    patch_lightrag_extraction_prompts()
    os.environ["SUMMARY_LANGUAGE"] = "Chinese"

    # 核心配置：指定解析器并开启多模态开关
    config = RAGAnythingConfig(
        working_dir="./rag_storage_test2",  # 知识图谱和向量库存储路径, 每次新的PDF用独立的
        parser="mineru",              # 使用 MinerU 进行专业级 PDF 解析
        parse_method="auto",           # 强制使用 OCR 方法识别 PDF（也可以填 "auto"）
        enable_image_processing=True, # 开启图像识别与图谱节点构建
        enable_table_processing=True, # 开启表格解析
        enable_equation_processing=True, # 开启公式解析
        kg_quality_enabled=True,      # 开启知识图谱质量治理
        kg_extraction_mode="ppe",     # PPE三轮抽取模式（可切换 legacy 回退）
        kg_canonical_language="zh",   # 统一中文主实体
        kg_relation_schema="fixed",   # 固定关系枚举
        kg_core_entity_types=["虫害", "病害", "作物", "病原菌", "药剂", "生长期", "生物分类"],
        kg_anchor_node_types=["部位", "时间"],
        kg_attribute_fields=["形态特征", "危害症状", "发病诱因", "发生时期", "防治要点", "生活习性", "发生规律"],
        kg_noise_drop_types=["header", "page_number"],
        kg_noise_drop_patterns=[
            r"^\s*None\s*$",
            r"^\s*\d+\s*$",
            r"蔬菜病虫害诊断[与于]防治原色图谱",
            r"(?:QR|qr|二维码|QR码|扫码)",
            r"^(?:[A-Za-z]:)?[/\\].+\.(?:jpg|jpeg|png|bmp|gif|webp|tiff?)\s*$",
        ],
        kg_description_policy="multimodal_only",  # 仅多模态元素保留description
        kg_ontology_profile="cruciferous_pest_disease",  # 十字花科病虫害本体（可扩展到多作物）
        kg_enforce_ontology=True,     # 强制关系主宾类型校验
        kg_merge_threshold=0.85,      # 语义合并阈值
    )

    # 定义文本大模型回调（用于文本知识抽取和最终回答）
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gemini-2.5-flash",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    # 定义视觉大模型回调（用于图片 OCR 后的语义理解和多模态抽取）
    def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
        if messages:
            return openai_complete_if_cache("gpt-4o", "", system_prompt=None, history_messages=[], messages=messages, api_key=api_key, base_url=base_url, **kwargs)
        elif image_data:
            return openai_complete_if_cache(
                "gemini-2.5-flash",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                        ],
                    }
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # 定义 Embedding 模型回调
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed.func(
            texts, model="text-embedding-3-large", api_key=api_key, base_url=base_url
        ),
    )

    # 初始化系统
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs={
            "addon_params": {
                "entity_types": KG_ENTITY_TYPES,
            }
        },
    )

    # ==========================================
    # 2. PDF OCR 识别与多模态知识图谱构建
    # ==========================================
    print("🚀 开始解析 PDF 并构建知识图谱（此过程可能较长，请耐心等待）...")
    await rag.process_document_complete(
        file_path="docs/test3_single.pdf",  # 替换为您的 PDF 路径
        output_dir="./output",
        parse_method="auto"
    )
    print("✅ 知识图谱构建完成！数据已保存在 ./rag_storage 目录。")

    # ==========================================
    # 3. 多模态 RAG 测试查询
    # ==========================================
    print("\n🔍 开始进行 RAG 查询测试...")
    
    # 测试A：常规查询（系统会自动检索图表描述、文本并利用 VLM 增强分析）
    question_1 = "小猿叶甲幼虫的形态特征是什么？"
    print(f"问：{question_1}")
    text_result = await rag.aquery(question_1, mode="hybrid")
    print(f"答：\n{text_result}\n")

    # # 测试B：携带特定模态片段进行联合提问（高级用法）
    # question_2 = "结合文档内容，详细解释这个公式的含义"
    # print(f"问：{question_2}")
    # multimodal_result = await rag.aquery_with_multimodal(
    #     question_2,
    #     multimodal_content=[{
    #         "type": "equation",
    #         "latex": "E = mc^2",  # 这里可以换成你在 PDF 中关注的任意公式或表格
    #         "equation_caption": "质能等价公式"
    #     }],
    #     mode="hybrid"
    # )
    # print(f"答：\n{multimodal_result}\n")

if __name__ == "__main__":
    asyncio.run(main())
