# Streamlit UI

This project now includes a Streamlit frontend in `ui/app.py`.

## Install

```bash
pip install -r requirements.txt
# or: pip install -e ".[ui]"
```

## Required env

At minimum:

```bash
export OPENAI_API_KEY=your_api_key
# optional
export OPENAI_BASE_URL=https://api.openai.com/v1
```

## Run

```bash
streamlit run ui/app.py
```

## Features (v1)

- Upload multiple PDFs/images (`png/jpg/jpeg/bmp/gif/webp/tif/tiff`) in one batch and build one isolated KB (`rag_storage_ui/{kb_id}`)
- Ask with text only or text + query image (image is analyzed and fused into retrieval/answer generation)
- Connect existing local KB storage directory (for example `./rag_storage_test2`)
- Async ingestion with stage progress and job events
- Query with structured citations (`answer + citations + graph_focus`)
- Multimodal citation rendering (image/table)
- Knowledge graph visualization with answer-focused subgraph and full graph toggle
