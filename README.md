🤖IntelliChat

This project is a multi-stage AI Agent that evolves step by step into a professional chatbot similar to ChatGPT.
It supports document understanding, web search, file uploads, and a modern web UI using Streamlit or Gradio.

📌 Features (Final Stage)

✅ Conversational AI  with context-aware memory
✅ Web Search Tool for real-time answers
✅ RAG (Retrieval-Augmented Generation) for private documents
✅ File Uploads (PDF, TXT, DOCX, CSV) → instantly usable in chat
✅ Dual UI Options: CLI (Terminal), Streamlit, or Gradio
✅ Attractive & Professional Web UI with chat history
✅ Accurate answers (not just links)
✅ Expandable with more tools (APIs, databases, etc.)



🛠️ Installation

# Create virtual environment
python -m venv venv
source venv/bin/activate    # (Linux/Mac)
venv\Scripts\activate       # (Windows)

# Install dependencies
pip install -r requirements.txt

requirements.txt
openai
streamlit
gradio
duckduckgo-search
langchain
langchain-community
langchain-openai
sentence-transformers
faiss-cpu
pypdf
python-docx

🚀 Stages Overview

Stage 1 — Simple CLI Chatbot
A basic chatbot using an LLM (OpenAI or HuggingFace).
Run:
python agent.py

Stage 2 — Memory & Context
Added conversation memory, so the agent remembers previous messages.
Still CLI-based.

Stage 3 — Web Search Tool
Integrated DuckDuckGo Search API for real-time internet queries:
search: cheapest flights from mumbai to dubai

Stage 4 — Document Support
Added RAG (Retrieval-Augmented Generation):
Upload and query PDF, TXT, DOCX files.
Uses FAISS + Sentence Transformers for semantic search.

Stage 5 — File Uploads
Now supports multiple file uploads at once (TXT, PDF, DOCX, CSV).
Documents are embedded into vector DB and searchable in chat.

Stage 6 — Combined Agent
Unified Chat + Search + RAG into one intelligent agent.
CLI tool works like ChatGPT with access to documents + internet search.

Stage 7 — Web App (Final Stage 🎉)
Turned the agent into a modern web app with:
✅ Streamlit Chat UI (app.py)
✅ Gradio Alternative UI (app.py --gradio)
✅ File uploader
✅ Persistent chat history

Run:
# Streamlit
streamlit run app.py

🎯 Usage
Type questions directly:
What is AI?

Search the web:
search: current gold price in dubai


Upload files:
Drag & drop PDF/DOCX/TXT/CSV → chat with them instantly

📊 Example Queries

Knowledge: "Explain Quantum Computing in simple terms"
Web Search: "search: weather in Riyadh today"
Docs: "Summarize the uploaded financial report"
Multi-Source: "Compare my CV with job description and suggest improvements"

🔮 Future Roadmap

 Add voice input/output
 Add multimodal (images + text) support
 Integrate with Google Calendar / Gmail API
 Deploy on cloud (AWS/GCP/Azure)
 Add user authentication
