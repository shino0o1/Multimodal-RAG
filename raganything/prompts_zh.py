"""
Chinese (中文) prompt templates for multimodal content processing.

Provides Chinese-language prompt templates as an alternative to the default
English templates.  Users can activate these at process level by calling
``set_prompt_language("zh")`` from :mod:`raganything.prompt_manager`.

Addresses GitHub issue #85 — prompt language support.
"""

from __future__ import annotations
from typing import Any

PROMPTS_ZH: dict[str, Any] = {}

# System prompts for different analysis types
PROMPTS_ZH["IMAGE_ANALYSIS_SYSTEM"] = (
    "你是一位农业病虫害知识抽取专家。请只提取对病虫害知识图谱有价值的信息，并严格按JSON格式输出。"
)
PROMPTS_ZH["IMAGE_ANALYSIS_FALLBACK_SYSTEM"] = (
    "你是一位专业的图像分析专家。请根据现有信息提供详细分析。"
)
PROMPTS_ZH["TABLE_ANALYSIS_SYSTEM"] = (
    "你是一位农业病虫害数据分析专家。请只提取对病虫害知识图谱有价值的表格信息，并严格按JSON格式输出。"
)
PROMPTS_ZH["EQUATION_ANALYSIS_SYSTEM"] = "你是一位数学专家。请提供详细的数学分析。"
PROMPTS_ZH["GENERIC_ANALYSIS_SYSTEM"] = "你是一位专注于{content_type}内容的专业分析师。"

# Image analysis prompt template
PROMPTS_ZH["vision_prompt"] = """请面向“农业病虫害知识抽取”分析这张图片，并以以下JSON结构提供回答：

{{
    "detailed_description": "只保留对农业病虫害知识图谱有价值的信息，遵循以下指导：
    - 优先识别：作物、病害、虫害、病原菌、药剂、生长期、时间
    - 优先提取：形态特征、危害症状、发生时期/生长期、防治要点
    - 生物分类命名使用单一规范名（如“鞘翅目”或“叶甲科”），避免“鞘翅目叶甲科”这类复合写法
    - detailed_description 按四段输出：[实体候选]、[属性句]、[关系线索]、[证据短句]
    - 保留可用于关系抽取的语义线索（如：致病、影响、发生于、使用药剂、防治、地理位置、属类隶属、生命周期）
    - 若图片包含文字，提取与病虫害相关的关键术语与短句
    - 忽略与病虫害无关的信息（版式装饰、摄影风格、无关人物/背景等）
    - 使用明确实体名，不使用代词",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "病虫害知识抽取摘要（不超过100字）"
    }}
}}

硬约束：
1. 如果图片主体是二维码，仅返回“空结果JSON”：
   {{
     "detailed_description": "",
     "entity_info": {{
       "entity_name": "{entity_name}",
       "entity_type": "image",
       "summary": ""
     }}
   }}
2. 不要输出任何额外解释文字，只输出JSON。

附加信息：
- 图片路径：{image_path}
- 标注：{captions}
- 脚注：{footnotes}

请专注于提升后续实体与关系抽取质量。"""

# Image analysis prompt with context support
PROMPTS_ZH[
    "vision_prompt_with_context"
] = """请结合上下文，面向“农业病虫害知识抽取”分析这张图片，并以以下JSON结构提供回答：

{{
    "detailed_description": "只保留对农业病虫害知识图谱有价值的信息，遵循以下指导：
    - 优先识别：作物、病害、虫害、病原菌、药剂、生长期、时间
    - 优先提取：形态特征、危害症状、发生时期/生长期、防治要点
    - 生物分类命名使用单一规范名（如“鞘翅目”或“叶甲科”），避免“鞘翅目叶甲科”这类复合写法
    - detailed_description 按四段输出：[实体候选]、[属性句]、[关系线索]、[证据短句]
    - 结合上下文补全实体指代，避免无意义碎片描述
    - 保留可用于关系抽取的语义线索（如：致病、影响、发生于、使用药剂、防治、地理位置、属类隶属、生命周期）
    - 若图片包含文字，提取与病虫害相关的关键术语与短句
    - 忽略与病虫害无关的信息（版式装饰、摄影风格、无关人物/背景等）
    - 使用明确实体名，不使用代词",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "病虫害知识抽取摘要（含上下文，不超过100字）"
    }}
}}

硬约束：
1. 如果图片主体是二维码/条形码/支付码/跳转码，仅返回“空结果JSON”：
   {{
     "detailed_description": "",
     "entity_info": {{
       "entity_name": "{entity_name}",
       "entity_type": "image",
       "summary": ""
     }}
   }}
2. 不要输出任何额外解释文字，只输出JSON。

周围内容上下文：
{context}

图片详细信息：
- 图片路径：{image_path}
- 标注：{captions}
- 脚注：{footnotes}

请专注于提升后续实体与关系抽取质量。"""

# Image analysis prompt with text fallback
PROMPTS_ZH["text_prompt"] = """根据以下图片信息提供分析：

图片路径：{image_path}
标注：{captions}
脚注：{footnotes}

{vision_prompt}"""

# Table analysis prompt template
PROMPTS_ZH["table_prompt"] = """请面向“农业病虫害知识抽取”分析此表格内容，并以以下JSON结构提供回答：

{{
    "detailed_description": "只保留对农业病虫害知识图谱有价值的表格信息，包括：
    - 与作物、病害、虫害、病原菌、药剂、生长期、时间相关的列与字段
    - 形态特征、危害症状、发生时期/生长期、防治要点等关键内容
    - 生物分类命名使用单一规范名，避免复合分类写法
    - 优先按“对象-字段-值”组织句子（如：对象=小猿叶甲，字段=形态特征，值=...）
    - detailed_description 按四段输出：[实体候选]、[属性句]、[关系线索]、[证据短句]
    - 可用于关系抽取的语义线索（如：致病、影响、发生于、使用药剂、防治、地理位置、属类隶属、生命周期）
    - 关键数值、剂量、时间范围、频次、条件等结构化信息
    - 忽略无关列、装饰性说明和与病虫害无关信息
    始终使用具体实体名、字段名和数值。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "病虫害知识抽取摘要（不超过100字）"
    }}
}}

硬约束：
1. 仅输出JSON，不要附加说明文字。
2. 若表格与农业病虫害无关，返回空结果JSON（字段保留、内容为空）。

表格信息：
图片路径：{table_img_path}
标题：{table_caption}
内容：{table_body}
脚注：{table_footnote}

请专注于提升后续实体与关系抽取质量。"""

# Table analysis prompt with context support
PROMPTS_ZH[
    "table_prompt_with_context"
] = """请结合上下文，面向“农业病虫害知识抽取”分析此表格内容，并以以下JSON结构提供回答：

{{
    "detailed_description": "只保留对农业病虫害知识图谱有价值的表格信息，包括：
    - 与作物、病害、虫害、病原菌、药剂、生长期、时间相关的列与字段
    - 形态特征、危害症状、发生时期/生长期、防治要点等关键内容
    - 生物分类命名使用单一规范名，避免复合分类写法
    - 优先按“对象-字段-值”组织句子（如：对象=小猿叶甲，字段=形态特征，值=...）
    - detailed_description 按四段输出：[实体候选]、[属性句]、[关系线索]、[证据短句]
    - 结合上下文补全实体指代与语义关系
    - 可用于关系抽取的语义线索（如：致病、影响、发生于、使用药剂、防治、地理位置、属类隶属）
    - 关键数值、剂量、时间范围、频次、条件等结构化信息
    - 忽略无关列、装饰性说明和与病虫害无关信息
    始终使用具体实体名、字段名和数值。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "病虫害知识抽取摘要（含上下文，不超过100字）"
    }}
}}

硬约束：
1. 仅输出JSON，不要附加说明文字。
2. 若表格与农业病虫害无关，返回空结果JSON（字段保留、内容为空）。

周围内容上下文：
{context}

表格信息：
图片路径：{table_img_path}
标题：{table_caption}
内容：{table_body}
脚注：{table_footnote}

请专注于提升后续实体与关系抽取质量。"""

# Equation analysis prompt template
PROMPTS_ZH["equation_prompt"] = """请分析此数学公式，并以以下JSON结构提供回答：

{{
    "detailed_description": "对公式的全面分析，包括：
    - 数学含义和解释
    - 变量及其定义
    - 使用的数学运算和函数
    - 应用领域和背景
    - 物理或理论意义
    - 与其他数学概念的关系
    - 实际应用或用例
    始终使用准确的数学术语。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "公式目的和重要性的简明摘要（不超过100字）"
    }}
}}

公式信息：
公式：{equation_text}
格式：{equation_format}

请专注于提供数学洞察和解释公式的重要性。"""

# Equation analysis prompt with context support
PROMPTS_ZH[
    "equation_prompt_with_context"
] = """请结合上下文分析此数学公式，并以以下JSON结构提供回答：

{{
    "detailed_description": "对公式的全面分析，包括：
    - 数学含义和解释
    - 在上下文中变量的定义
    - 使用的数学运算和函数
    - 基于周围材料的应用领域和背景
    - 物理或理论意义
    - 与上下文中提到的其他数学概念的关系
    - 实际应用或用例
    - 公式如何与更广泛的讨论或框架相关联
    始终使用准确的数学术语。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "公式目的、重要性及在上下文中作用的简明摘要（不超过100字）"
    }}
}}

周围内容上下文：
{context}

公式信息：
公式：{equation_text}
格式：{equation_format}

请专注于在更广泛的上下文中提供数学洞察和解释公式的重要性。"""

# Generic content analysis prompt template
PROMPTS_ZH["generic_prompt"] = """请分析此{content_type}内容，并以以下JSON结构提供回答：

{{
    "detailed_description": "对内容的全面分析，包括：
    - 内容结构和组织
    - 关键信息和元素
    - 组件之间的关系
    - 背景和重要性
    - 与知识检索相关的细节
    始终使用适合{content_type}内容的专业术语。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "内容目的和要点的简明摘要（不超过100字）"
    }}
}}

内容：{content}

请专注于提取对知识检索有用的有意义信息。"""

# Generic content analysis prompt with context support
PROMPTS_ZH[
    "generic_prompt_with_context"
] = """请结合上下文分析此{content_type}内容，并以以下JSON结构提供回答：

{{
    "detailed_description": "对内容的全面分析，包括：
    - 内容结构和组织
    - 关键信息和元素
    - 组件之间的关系
    - 与周围内容相关的背景和重要性
    - 此内容如何与更广泛的讨论相联系或支持
    - 与知识检索相关的细节
    始终使用适合{content_type}内容的专业术语。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "内容目的、要点及与周围上下文关系的简明摘要（不超过100字）"
    }}
}}

周围内容上下文：
{context}

内容：{content}

请专注于提取对知识检索有用的信息，并理解内容在更广泛上下文中的作用。"""

# Modal chunk templates
PROMPTS_ZH["image_chunk"] = """
图片内容分析：
图片路径：{image_path}
标注：{captions}
脚注：{footnotes}

视觉分析：{enhanced_caption}"""

PROMPTS_ZH["table_chunk"] = """表格分析：
图片路径：{table_img_path}
标题：{table_caption}
结构：{table_body}
脚注：{table_footnote}

分析：{enhanced_caption}"""

PROMPTS_ZH["equation_chunk"] = """数学公式分析：
公式：{equation_text}
格式：{equation_format}

数学分析：{enhanced_caption}"""

PROMPTS_ZH["generic_chunk"] = """{content_type}内容分析：
内容：{content}

分析：{enhanced_caption}"""

# Query-related prompts
PROMPTS_ZH["QUERY_IMAGE_DESCRIPTION"] = (
    "请简要描述这张图片的主要内容、关键元素和重要信息。"
)

PROMPTS_ZH["QUERY_IMAGE_ANALYST_SYSTEM"] = (
    "你是一位能准确描述图片内容的专业图像分析师。"
)

PROMPTS_ZH["QUERY_TABLE_ANALYSIS"] = """请分析以下表格数据的主要内容、结构和关键信息：

表格数据：
{table_data}

表格标题：{table_caption}

请简要总结表格的主要内容、数据特征和重要发现。"""

PROMPTS_ZH["QUERY_TABLE_ANALYST_SYSTEM"] = (
    "你是一位能准确分析表格数据的专业数据分析师。"
)

PROMPTS_ZH["QUERY_EQUATION_ANALYSIS"] = """请解释以下数学公式的含义和用途：

LaTeX公式：{latex}
公式标题：{equation_caption}

请简要说明这个公式的数学意义、应用场景和重要性。"""

PROMPTS_ZH["QUERY_EQUATION_ANALYST_SYSTEM"] = "你是一位能清晰解释数学公式的数学专家。"

PROMPTS_ZH[
    "QUERY_GENERIC_ANALYSIS"
] = """请分析以下{content_type}类型内容并提取其主要信息和关键特征：

内容：{content_str}

请简要总结此内容的主要特征和重要信息。"""

PROMPTS_ZH["QUERY_GENERIC_ANALYST_SYSTEM"] = (
    "你是一位能准确分析{content_type}类型内容的专业内容分析师。"
)

PROMPTS_ZH["QUERY_ENHANCEMENT_SUFFIX"] = (
    "\n\n请基于用户查询和提供的多模态内容信息，提供全面的回答。"
)
