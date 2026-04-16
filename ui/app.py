"""Streamlit UI for RAG-Anything with structured citations and graph focus."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

try:
    from streamlit_echarts import st_echarts
except Exception:  # pragma: no cover - dependency handled at runtime
    st_echarts = None

try:
    # Preferred import when running as package/module from project root.
    from ui.service import RAGUIService
except ModuleNotFoundError:
    # Fallback for `streamlit run ui/app.py` when cwd/sys.path points to `ui/`.
    from service import RAGUIService


@st.cache_resource(show_spinner=False)
def get_service() -> RAGUIService:
    return RAGUIService()


def _init_state() -> None:
    defaults = {
        "selected_kb_id": "",
        "last_query_result": None,
        "focus_chunk_ids": [],
        "graph_full": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _build_graph_option(payload: Dict[str, Any]) -> Dict[str, Any]:
    nodes = payload.get("nodes", [])
    links = payload.get("links", [])
    highlight_nodes = set(payload.get("highlight_node_ids", []))
    highlight_edges = set(payload.get("highlight_edge_ids", []))

    categories = sorted({n.get("category", "Unknown") for n in nodes})
    category_index = {name: idx for idx, name in enumerate(categories)}

    graph_nodes: List[Dict[str, Any]] = []
    for node in nodes:
        node_id = node.get("id", "")
        category_name = node.get("category", "Unknown")
        item_style = dict(node.get("itemStyle", {}))

        if node_id in highlight_nodes:
            item_style.update(
                {
                    "borderColor": "#ff6b35",
                    "borderWidth": 2,
                    "shadowBlur": 8,
                    "shadowColor": "rgba(255, 107, 53, 0.6)",
                }
            )

        graph_nodes.append(
            {
                "id": node_id,
                "name": node.get("name", node_id),
                "value": node.get("value", ""),
                "category": category_index.get(category_name, 0),
                "symbolSize": node.get("symbolSize", 16),
                "itemStyle": item_style,
                "label": {"show": True, "fontSize": 11},
            }
        )

    graph_links: List[Dict[str, Any]] = []
    for edge in links:
        edge_id = edge.get("id", "")
        base_style = dict(edge.get("lineStyle", {}))
        if edge_id in highlight_edges:
            base_style.update(
                {
                    "color": "#ff6b35",
                    "width": max(2, base_style.get("width", 1)),
                    "opacity": 0.9,
                }
            )
        graph_links.append(
            {
                "source": edge.get("source", ""),
                "target": edge.get("target", ""),
                "name": edge.get("name", ""),
                "value": edge.get("value", ""),
                "lineStyle": base_style,
            }
        )

    return {
        "tooltip": {"trigger": "item"},
        "legend": [{"data": categories, "top": 8}],
        "series": [
            {
                "name": "KnowledgeGraph",
                "type": "graph",
                "layout": "force",
                "roam": True,
                "draggable": True,
                "focusNodeAdjacency": True,
                "force": {
                    "repulsion": 180,
                    "gravity": 0.08,
                    "edgeLength": [70, 220],
                },
                "lineStyle": {
                    "curveness": 0.18,
                    "opacity": 0.65,
                },
                "label": {"position": "right"},
                "edgeLabel": {"show": False},
                "categories": [{"name": c} for c in categories],
                "data": graph_nodes,
                "links": graph_links,
                "emphasis": {
                    "focus": "adjacency",
                    "lineStyle": {"width": 2},
                },
            }
        ],
    }


def _render_kb_panel(service: RAGUIService) -> str:
    st.subheader("1) 知识库管理")

    upload_types = service.supported_upload_types()
    uploaded_files = st.file_uploader(
        "上传文档（PDF/图片）并建库",
        type=upload_types,
        accept_multiple_files=True,
        key="uploader_doc",
    )
    st.caption(f"支持类型：{', '.join(upload_types)}")
    if st.button("开始建库", type="primary", use_container_width=True):
        valid_files = [f for f in (uploaded_files or []) if f is not None]
        if not valid_files:
            st.warning("请先选择至少一个待建库文件（PDF 或图片）")
        else:
            try:
                result = service.create_kb(valid_files)
                st.session_state["selected_kb_id"] = result["kb_id"]
                st.success(f"建库任务已创建：{result['job_id']}（共 {len(valid_files)} 个文件）")
            except Exception as exc:
                st.error(f"建库失败：{exc}")

    with st.expander("接入本地已有知识库（working_dir）", expanded=False):
        existing_working_dir = st.text_input(
            "已有知识库存储路径",
            value="./rag_storage_test2",
            key="existing_working_dir_input",
        )
        existing_output_dir = st.text_input(
            "对应 output 路径（可选）",
            value="./output",
            key="existing_output_dir_input",
            help="用于图片安全目录校验，默认 ./output",
        )
        if st.button("接入已有库", use_container_width=True):
            try:
                result = service.register_existing_kb(
                    working_dir=existing_working_dir.strip(),
                    output_dir=existing_output_dir.strip() or None,
                )
                st.session_state["selected_kb_id"] = result["kb_id"]
                if result.get("existed"):
                    st.success(f"已存在并切换：{result['kb_id']}")
                else:
                    st.success(f"接入成功：{result['kb_id']}")
            except Exception as exc:
                st.error(f"接入失败：{exc}")

    kb_items = service.list_kbs()
    kb_options = [item.get("kb_id", "") for item in kb_items]

    selected = st.session_state.get("selected_kb_id", "")
    if kb_options:
        if selected not in kb_options:
            selected = kb_options[0]
        selected = st.selectbox(
            "选择知识库",
            kb_options,
            index=kb_options.index(selected),
        )
        st.session_state["selected_kb_id"] = selected

        meta = service.get_kb_meta(selected)
        if meta:
            st.caption(f"文件: {meta.get('file_name', '')}")
            st.caption(f"状态: {meta.get('status', '')} / 阶段: {meta.get('stage', '')}")
            st.progress(int(meta.get("progress", 0)))

            job_id = meta.get("job_id", "")
            if job_id:
                job = service.get_job(job_id)
                if job.get("error"):
                    st.error(job["error"])
                with st.expander("查看任务事件", expanded=False):
                    events = job.get("events", [])
                    for event in events[-20:]:
                        st.write(event)

    return st.session_state.get("selected_kb_id", "")


def _render_qa_panel(service: RAGUIService, kb_id: str) -> None:
    st.subheader("2) 问答与引用")

    if not kb_id:
        st.info("先创建或选择知识库")
        return

    question = st.text_area("输入问题", height=110, placeholder="例如：小猿叶甲幼虫的形态特征是什么？")
    query_image = st.file_uploader(
        "可选：上传查询图片（用于识别/检索增强）",
        type=service.supported_query_image_types(),
        key="query_image_uploader",
    )
    if query_image is not None:
        st.image(query_image, caption=f"查询图片：{getattr(query_image, 'name', '')}", use_container_width=True)
    mode = st.selectbox("检索模式", ["hybrid", "mix", "local", "global", "naive", "bypass"], index=0)
    debug = st.checkbox("返回调试上下文", value=False)

    if st.button("生成回答", use_container_width=True):
        if not question.strip() and query_image is None:
            st.warning("请输入问题或上传查询图片")
        else:
            with st.spinner("检索与生成中..."):
                try:
                    if query_image is not None:
                        result = service.query_with_image(
                            kb_id,
                            question.strip(),
                            query_image,
                            mode=mode,
                            debug=debug,
                        )
                    else:
                        result = service.query(kb_id, question.strip(), mode=mode, debug=debug)
                    st.session_state["last_query_result"] = result
                    st.session_state["focus_chunk_ids"] = result.get("graph_focus", {}).get("chunk_ids", [])
                    st.success("回答已生成")
                except Exception as exc:
                    st.error(f"查询失败：{exc}")

    result = st.session_state.get("last_query_result")
    if not result:
        return

    st.markdown("**回答**")
    st.write(result.get("answer", ""))

    graph_focus = result.get("graph_focus", {})
    if graph_focus.get("chunk_ids"):
        if st.button("按本次回答定位图谱", key="focus_answer_graph"):
            st.session_state["focus_chunk_ids"] = graph_focus.get("chunk_ids", [])
            st.rerun()

    citations = result.get("citations", [])
    st.markdown(f"**引用片段（{len(citations)}）**")

    for idx, citation in enumerate(citations, start=1):
        title = f"[{idx}] {citation.get('modality', 'text')} | chunk: {citation.get('chunk_id', '') or 'N/A'}"
        with st.expander(title, expanded=False):
            st.caption(
                f"file={citation.get('file_path', '')}, page_idx={citation.get('page_idx', -1)}, source={citation.get('source_type', '')}"
            )
            st.write(citation.get("snippet", ""))

            modality = citation.get("modality", "text")
            asset_ref = citation.get("asset_ref", {}) or {}

            if modality == "image":
                img_path = asset_ref.get("image_path", "")
                if img_path and Path(img_path).exists():
                    st.image(img_path, caption=img_path)
                elif img_path:
                    st.code(img_path)

            elif modality == "table":
                table_html = asset_ref.get("table_html", "")
                if table_html:
                    try:
                        frames = pd.read_html(StringIO(table_html))
                        if frames:
                            st.dataframe(frames[0], use_container_width=True)
                    except Exception:
                        st.code(table_html)

                img_path = asset_ref.get("image_path", "")
                if img_path and Path(img_path).exists():
                    st.image(img_path, caption=img_path)

            elif modality == "equation":
                st.code(asset_ref.get("equation_text", ""))

            chunk_id = citation.get("chunk_id", "")
            if chunk_id and st.button("定位该引用到图谱", key=f"focus_chunk_{idx}"):
                st.session_state["focus_chunk_ids"] = [chunk_id]
                st.rerun()

    if debug and result.get("debug", {}).get("context_raw"):
        with st.expander("Debug Context", expanded=False):
            st.code(result["debug"]["context_raw"])


def _render_graph_panel(service: RAGUIService, kb_id: str) -> None:
    st.subheader("3) 知识图谱")

    if not kb_id:
        st.info("先创建或选择知识库")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        full_graph = st.toggle("显示全图", value=st.session_state.get("graph_full", False))
    with col2:
        if st.button("清除定位", use_container_width=True):
            st.session_state["focus_chunk_ids"] = []
            st.rerun()

    st.session_state["graph_full"] = full_graph

    focus_chunk_ids = st.session_state.get("focus_chunk_ids", [])
    payload = service.get_graph(kb_id, focus_chunk_ids=focus_chunk_ids, full=full_graph)

    if payload.get("error"):
        st.warning(payload["error"])
        return

    st.caption(
        f"nodes={len(payload.get('nodes', []))}, links={len(payload.get('links', []))}, highlight_nodes={len(payload.get('highlight_node_ids', []))}, highlight_edges={len(payload.get('highlight_edge_ids', []))}"
    )

    if st_echarts is None:
        st.error("缺少依赖 streamlit-echarts，请先安装后运行 UI")
        return

    option = _build_graph_option(payload)
    st_echarts(option, height="720px", key=f"kg_{kb_id}_{full_graph}")


def main() -> None:
    st.set_page_config(
        page_title="RAG-Anything UI",
        page_icon="📚",
        layout="wide",
    )

    _init_state()
    service = get_service()

    st.title("RAG-Anything 前端（Streamlit）")

    kb_id = _render_kb_panel(service)
    st.divider()
    _render_qa_panel(service, kb_id)
    st.divider()
    _render_graph_panel(service, kb_id)


if __name__ == "__main__":
    main()
