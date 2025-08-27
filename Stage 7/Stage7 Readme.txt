1) Install (one-time)

Run this in your terminal (PowerShell/CMD) inside my-agent:
pip install streamlit ollama ddgs requests beautifulsoup4 readability-lxml lxml python-docx PyPDF2 sentence-transformers faiss-cpu numpy

If ddgs isn’t available on your mirror, try: pip install duckduckgo-search (the code handles both).



2) Start the web app
streamlit run agent.py
or
python agent.py gradio


It’ll open in your browser (usually at http://localhost:8501).
Upload .txt / .pdf / .docx files in the sidebar to add them into RAG automatically.

3) How to use it:
Open the app → Sidebar: upload .txt / .pdf / .docx
It will auto rebuild the RAG index (or click “Rebuild RAG Index”)
In the chat box:
Type search: <query> for web results
Type rag: <question> to answer from your docs
Or toggle “Use docs (RAG)” and ask normally
Chat history persists to history_web.json
FAISS index persists (faiss_index.index + doc_texts.json)