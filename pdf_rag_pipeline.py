import asyncio
import os
import time
from pathlib import Path
from raganything import RAGAnything, RAGAnythingConfig, set_prompt_language
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

BUILD_GLOBAL_TIMEOUT_SECONDS = 12 * 60 * 60  # 6 hours
BUILD_STALL_TIMEOUT_SECONDS = 100 * 60        # no file write for 110 minutes
BUILD_WATCH_INTERVAL_SECONDS = 30            # check every 30s


def _latest_mtime_in_dir(target_dir: str) -> float:
    root = Path(target_dir)
    if not root.exists():
        return 0.0
    latest = root.stat().st_mtime
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > latest:
            latest = mtime
    return latest


async def _watch_build_progress(
    process_task: asyncio.Task,
    working_dir: str,
    stall_timeout_seconds: int,
    watch_interval_seconds: int,
) -> str:
    last_mtime = _latest_mtime_in_dir(working_dir)
    last_progress_ts = time.monotonic()
    while not process_task.done():
        await asyncio.sleep(watch_interval_seconds)
        current_mtime = _latest_mtime_in_dir(working_dir)
        if current_mtime > last_mtime:
            last_mtime = current_mtime
            last_progress_ts = time.monotonic()
            print(f"[watchdog] 检测到工作目录更新：{working_dir}")
            continue
        idle_seconds = time.monotonic() - last_progress_ts
        if idle_seconds >= stall_timeout_seconds:
            return (
                f"构建流程疑似卡住：{working_dir} 已 {int(idle_seconds)}s 无文件写入"
            )
    return ""


async def _run_build_with_guard(
    rag: RAGAnything,
    file_path: str,
    output_dir: str,
    parse_method: str,
) -> None:
    process_task = asyncio.create_task(
        rag.process_document_complete(
            file_path=file_path,
            output_dir=output_dir,
            parse_method=parse_method,
        )
    )
    watchdog_task = asyncio.create_task(
        _watch_build_progress(
            process_task=process_task,
            working_dir=rag.config.working_dir,
            stall_timeout_seconds=BUILD_STALL_TIMEOUT_SECONDS,
            watch_interval_seconds=BUILD_WATCH_INTERVAL_SECONDS,
        )
    )

    done, pending = await asyncio.wait(
        {process_task, watchdog_task},
        timeout=BUILD_GLOBAL_TIMEOUT_SECONDS,
        return_when=asyncio.FIRST_COMPLETED,
    )

    if not done:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise TimeoutError(
            f"构建超时：超过 {BUILD_GLOBAL_TIMEOUT_SECONDS}s 仍未完成"
        )

    if process_task in done:
        # process finished, cancel watchdog
        watchdog_task.cancel()
        await asyncio.gather(watchdog_task, return_exceptions=True)
        await process_task
        return

    # watchdog finished first
    stall_reason = watchdog_task.result()
    process_task.cancel()
    await asyncio.gather(process_task, return_exceptions=True)
    raise TimeoutError(stall_reason or "构建疑似卡住，被 watchdog 终止")


async def main():
    # ==========================================
    # 1. 环境与模型配置
    # ==========================================
    # api_key = "sk-uwqgblktuqsujrieppjbujsrdkwtirmxtlkfjuwsmwcaloag" # 替换为您的 API Key
    api_key = ""  # Prefer config.llm_api_key
    # base_url = "https://api.siliconflow.cn/v1" # 如果使用代理可以修改此处
    base_url = ""  # Prefer config.llm_base_url
    set_prompt_language("zh")
    os.environ["SUMMARY_LANGUAGE"] = "Chinese"
    # LLM 单次调用超时（秒）
    os.environ.setdefault("LLM_TIMEOUT", "200")

    # 核心配置：指定解析器并开启多模态开关
    config = RAGAnythingConfig(
        # working_dir="./rag_storage_test6",  # 知识图谱和向量库存储路径, 每次新的PDF用独立的
        working_dir="./rag_storage_whole_book_gemini",  # 知识图谱和向量库存储路径, 每次新的PDF用独立的
        parser="mineru",              # 使用 MinerU 进行专业级 PDF 解析
        parse_method="auto",           # 强制使用 OCR 方法识别 PDF（也可以填 "auto"）
        enable_image_processing=True, # 开启图像识别与图谱节点构建
        enable_table_processing=True, # 开启表格解析
        enable_equation_processing=False, # 开启公式解析
        kg_quality_enabled=True,      # 开启知识图谱质量治理
        kg_extraction_mode="ppe",     # PPE三轮抽取模式（可切换 legacy 回退）
        kg_canonical_language="zh",   # 统一中文主实体
        kg_relation_schema="fixed",   # 固定关系枚举
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
        kg_merge_threshold=0.85,      # 向量主导去重的灰区阈值（严格阈值在质量层内自动推导）
        kg_llm_semantic_merge_enabled=False,  # 默认关闭LLM灰区仲裁（避免卡顿）
        kg_llm_semantic_merge_types=["作物", "生物分类", "病原菌", "药剂", "病害", "虫害"],
        kg_llm_semantic_name_sim_threshold=0.75,  # 候选分组阈值（规则预筛）
        kg_llm_semantic_merge_min_confidence=0.90,  # LLM输出最小置信度
        kg_llm_semantic_merge_max_group_size=20,
        kg_llm_semantic_merge_max_groups=80,
        kg_llm_timeout_seconds=90,
    )

    api_key = config.llm_api_key.strip() or api_key or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing API key: set config.llm_api_key in pdf_rag_pipeline.py "
            "or export OPENAI_API_KEY."
        )
    base_url = config.llm_base_url.strip() or base_url or os.getenv("OPENAI_BASE_URL", "").strip() or None

    answer_reasoning_effort = config.get_reasoning_effort("answer")
    planner_reasoning_effort = config.get_reasoning_effort("planner")
    vision_reasoning_effort = config.get_reasoning_effort("vision")
    image_desc_reasoning_effort = config.get_reasoning_effort("image_description")

    def text_model_func(model_name, prompt, system_prompt=None, history_messages=[], **kwargs):
        if "reasoning_effort" not in kwargs:
            if model_name == config.model_planner and planner_reasoning_effort:
                kwargs["reasoning_effort"] = planner_reasoning_effort
            elif answer_reasoning_effort:
                kwargs["reasoning_effort"] = answer_reasoning_effort
        return openai_complete_if_cache(
            model_name,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    # 答案生成模型（Answer）
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        if "reasoning_effort" not in kwargs and answer_reasoning_effort:
            kwargs["reasoning_effort"] = answer_reasoning_effort
        return text_model_func(
            config.model_answer,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    # 规划模型（Planner）
    def planner_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        if "reasoning_effort" not in kwargs and planner_reasoning_effort:
            kwargs["reasoning_effort"] = planner_reasoning_effort
        return text_model_func(
            config.model_planner,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    # 视觉问答模型（Vision）
    def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
        if "reasoning_effort" not in kwargs and vision_reasoning_effort:
            kwargs["reasoning_effort"] = vision_reasoning_effort
        if messages:
            return openai_complete_if_cache(config.model_vision, "", system_prompt=None, history_messages=[], messages=messages, api_key=api_key, base_url=base_url, **kwargs)
        elif image_data:
            return openai_complete_if_cache(
                config.model_vision,
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

    # 图片描述模型（Image Description）
    def image_description_model_func(
        prompt,
        system_prompt=None,
        history_messages=[],
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if "reasoning_effort" not in kwargs and image_desc_reasoning_effort:
            kwargs["reasoning_effort"] = image_desc_reasoning_effort
        if messages:
            return openai_complete_if_cache(
                config.model_image_description,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        elif image_data:
            return openai_complete_if_cache(
                config.model_image_description,
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
                    },
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        else:
            return planner_model_func(prompt, system_prompt, history_messages, **kwargs)

    # 定义 Embedding 模型回调
    embedding_func = EmbeddingFunc(
        embedding_dim=config.embedding_dim,
        max_token_size=8192,
        func=lambda texts: openai_embed.func(
            texts, model=config.model_embedding, api_key=api_key, base_url=base_url
        ),
    )

    # 初始化系统
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        planner_model_func=planner_model_func,
        vision_model_func=vision_model_func,
        image_description_model_func=image_description_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs={
            # 关闭二次抽取（gleaning），减少超时与格式噪声
            "entity_extract_max_gleaning": 0,
            # 限制抽取阶段输入上限，控制提示词长度
            "max_extract_input_tokens": 5120,
            # 控制并发，避免整书构建时触发上游 TPM 限流和 worker 超时
            "llm_model_max_async": 4,
            "embedding_func_max_async": 4,
            "max_parallel_insert": 4,
        },
    )

    # ==========================================
    # 2. PDF OCR 识别与多模态知识图谱构建
    # ==========================================
    print("🚀 开始解析 PDF 并构建知识图谱（此过程可能较长，请耐心等待）...")
    rag_for_query = rag
    await _run_build_with_guard(
        rag=rag,
        file_path="docs/十字花科蔬菜病虫害_3.pdf",
        # file_path="docs/test4.pdf",
        output_dir="./output",
        parse_method="auto",
    )
    print(f"✅ 知识图谱构建完成！数据已保存在 {config.working_dir} 目录。")

    # ==========================================
    # 3. 多模态 RAG 测试查询
    # ==========================================
    print("\n🔍 开始进行 RAG 查询测试...")
    
    # 测试A：常规查询（系统会自动检索图表描述、文本并利用 VLM 增强分析）
    question_1 = "小猿叶甲幼虫的形态特征是什么？"
    print(f"问：{question_1}")
    text_result = await rag_for_query.aquery(question_1, mode=config.query_mode)
    print(f"答：\n{text_result}\n")

    # 测试A-2：先规划再检索（Graph-RFT 简版）
    # - Planner先拆解问题，再基于本地KG检索与回答
    question_plan = "最近有哪些关于十字花科害虫绿色防控的新建议？"
    print(f"问（计划检索）：{question_plan}")
    plan_result = await rag_for_query.aquery_plan_then_retrieve(
        question_plan,
        mode=config.query_mode,
    )
    print(f"答（计划检索）：\n{plan_result}\n")

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
