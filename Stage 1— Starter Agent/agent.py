#!/usr/bin/env python3
"""
Model: GPT-5 Thinking — Pro Reasoning Mode | Date: 2025-08-20
Free Local AI Agent (single file)

Features
- Local LLM via Ollama (default: llama3.1:8b)
- ReAct-style tool use (no paid APIs)
- Tools: web_search, browse_url, calculator, filesystem read/write
- Long-term memory (SQLite)
- Lightweight RAG over local docs/notes (sentence-transformers on CPU)
- Simple CLI chat loop

Prereqs (install once)
  1) Install Python 3.10+
  2) Install Ollama: https://ollama.com (Mac/Win/Linux)
     Then pull a model, e.g.:  `ollama pull llama3.1:8b`
  3) pip install -U duckduckgo-search requests beautifulsoup4 lxml readability-lxml
                         sentence-transformers faiss-cpu numpy pandas rich

Run
  python agent.py

Folders created
  data/agent.db        (SQLite memory)
  data/vector.index    (FAISS index)
  docs/                (Put PDFs/TXT/MD here for RAG)
  notes/               (Agent notes; also included in RAG)

Tip
- First run will build an index from docs/ and notes/; keep it small to start.
"""

import os, sys, re, json, math, time, sqlite3, uuid, textwrap
from pathlib import Path
from typing import Dict, Any, List, Tuple

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from readability import Document

import numpy as np

# Embeddings & FAISS
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table

console = Console()
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DOCS = BASE / "docs"
NOTES = BASE / "notes"
DATA.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)
NOTES.mkdir(exist_ok=True)
DB_PATH = DATA / "agent.db"
INDEX_PATH = DATA / "vector.index"
MODEL_NAME = os.environ.get("AGENT_MODEL", "llama3.1:8b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
EMB_MODEL_NAME = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# Utility: SQLite memory store
# ----------------------------
class Memory:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(path)
        self._init()
    def _init(self):
        c = self.conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id TEXT PRIMARY KEY,
            topic TEXT,
            content TEXT,
            ts REAL
        )
        """)
        self.conn.commit()
    def remember(self, topic: str, content: str):
        c = self.conn.cursor()
        c.execute("INSERT INTO memory VALUES (?,?,?,?)",
                  (str(uuid.uuid4()), topic, content, time.time()))
        self.conn.commit()
    def recall(self, topic: str, limit: int = 5) -> List[Tuple[str,str,float]]:
        c = self.conn.cursor()
        c.execute("SELECT topic, content, ts FROM memory WHERE topic LIKE ? ORDER BY ts DESC LIMIT ?",
                  (f"%{topic}%", limit))
        return c.fetchall()

MEM = Memory(DB_PATH)

# ----------------------------
# Embedding + FAISS Index
# ----------------------------
class RAG:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.model = None
        self.index = None
        self.docs: List[Tuple[str,str]] = []  # (doc_id, text)
        if SentenceTransformer is None:
            console.print("[yellow]sentence-transformers not installed. RAG disabled.[/yellow]")
            return
        self.model = SentenceTransformer(EMB_MODEL_NAME)
        if faiss is None:
            console.print("[yellow]faiss not installed. Using naive cosine search.[/yellow]")
            self.index = None
        self._load_or_build()
    def _iter_texts(self) -> List[Tuple[str,str]]:
        items: List[Tuple[str,str]] = []
        for folder in [DOCS, NOTES]:
            for p in folder.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".txt",".md",".markdown"}:
                    try:
                        items.append((str(p), p.read_text(encoding="utf-8", errors="ignore")))
                    except Exception:
                        pass
        return items
    def _load_or_build(self):
        self.docs = self._iter_texts()
        texts = [t for _, t in self.docs]
        if not texts:
            return
        emb = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        if faiss is not None:
            dim = emb.shape[1]
            index = faiss.IndexFlatIP(dim)
            # Normalize for cosine similarity
            emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
            index.add(emb_norm.astype('float32'))
            self.index = (index, emb_norm)
        else:
            self.index = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        # Persist (optional: could store vectors to disk)
    def refresh(self):
        self._load_or_build()
    def search(self, query: str, k: int = 5) -> List[Tuple[str,str,float]]:
        if not self.docs:
            return []
        q = self.model.encode([query], convert_to_numpy=True)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        if faiss is not None and isinstance(self.index, tuple):
            index, _ = self.index
            D, I = index.search(q.astype('float32'), k)
            out = []
            for score, idx in zip(D[0], I[0]):
                doc_id, text = self.docs[int(idx)]
                out.append((doc_id, text[:1500], float(score)))
            return out
        else:
            # naive cosine
            sims = (self.index @ q.T).ravel()
            topk = np.argsort(-sims)[:k]
            out = []
            for idx in topk:
                doc_id, text = self.docs[int(idx)]
                out.append((doc_id, text[:1500], float(sims[idx])))
            return out

RAGGER = RAG(INDEX_PATH)

# ----------------------------
# Tools
# ----------------------------

def tool_web_search(query: str, max_results: int = 5) -> str:
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({"title": r.get("title"), "href": r.get("href"), "body": r.get("body")})
    except Exception as e:
        return f"ERROR web_search: {e}"
    return json.dumps(results, ensure_ascii=False)


def _extract_main_text(html: str, url: str) -> str:
    try:
        doc = Document(html)
        summary_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(summary_html, "lxml")
        text = soup.get_text("\n")
        return f"[URL] {url}\n\n{text[:5000]}"
    except Exception:
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text("\n")
        return f"[URL] {url}\n\n{text[:5000]}"


def tool_browse_url(url: str, timeout: int = 10) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        return _extract_main_text(r.text, url)
    except Exception as e:
        return f"ERROR browse_url: {e}"


def tool_calculator(expr: str) -> str:
    try:
        # Safe eval using limited namespaces
        allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed_names.update({"__builtins__": {}})
        result = eval(expr, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"ERROR calculator: {e}"


def tool_read_file(path: str, max_chars: int = 4000) -> str:
    p = (BASE / path).resolve()
    if not str(p).startswith(str(BASE)):
        return "ERROR read_file: path traversal blocked"
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
        return text[:max_chars]
    except Exception as e:
        return f"ERROR read_file: {e}"


def tool_write_note(topic: str, content: str) -> str:
    MEM.remember(topic, content)
    note_path = NOTES / f"{int(time.time())}_{re.sub(r'[^a-zA-Z0-9_-]+','_',topic)}.md"
    try:
        note_path.write_text(content, encoding="utf-8")
    except Exception:
        pass
    # refresh RAG so note becomes searchable
    if RAGGER.model is not None:
        RAGGER.refresh()
    return f"Saved note on '{topic}'."


def tool_recall(topic: str) -> str:
    rows = MEM.recall(topic, limit=5)
    if not rows:
        return "No memory found."
    lines = [f"- {time.strftime('%Y-%m-%d %H:%M', time.localtime(ts))} — {content}" for (topic, content, ts) in rows]
    return "\n".join(lines)


def tool_rag_search(query: str) -> str:
    if RAGGER.model is None:
        return "RAG disabled (missing sentence-transformers)."
    hits = RAGGER.search(query, k=5)
    if not hits:
        return "No docs indexed. Put .txt/.md files in docs/ or notes/."
    out = []
    for doc_id, text, score in hits:
        out.append({"doc": doc_id, "score": round(float(score), 3), "snippet": text[:500]})
    return json.dumps(out, ensure_ascii=False)

TOOLS = {
    "web_search": {"fn": tool_web_search, "doc": "Search the web via DuckDuckGo. Args: query, max_results"},
    "browse_url": {"fn": tool_browse_url, "doc": "Fetch and summarize a URL. Args: url"},
    "calculator": {"fn": tool_calculator, "doc": "Evaluate a math expression. Args: expr"},
    "read_file":  {"fn": tool_read_file,  "doc": "Read a local file (relative path). Args: path, max_chars?"},
    "write_note": {"fn": tool_write_note, "doc": "Save a note to memory + notes/. Args: topic, content"},
    "recall":     {"fn": tool_recall,     "doc": "Recall recent notes by topic. Args: topic"},
    "rag_search": {"fn": tool_rag_search,  "doc": "Semantic search over docs/ + notes/. Args: query"},
}

TOOL_DESCRIPTIONS = "\n".join([f"- {name}: {meta['doc']}" for name, meta in TOOLS.items()])

# ----------------------------
# LLM via Ollama chat API
# ----------------------------

def ollama_chat(messages: List[Dict[str, Any]], stream: bool=False) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        # system prompt nudges the model to emit ReAct steps
        "options": {"temperature": 0.6}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "")

SYSTEM_PROMPT = f"""
You are a helpful AI agent that can use tools when needed. Follow this protocol:
- Think step-by-step but ONLY output in the following schema.
- When you need a tool, output exactly:
  ACTION: <tool_name>
  ACTION_INPUT: <JSON arguments>
- After receiving the observation, continue reasoning. When you are ready to answer the user, output only:
  FINAL: <your answer for the user>

Available tools:\n{TOOL_DESCRIPTIONS}

Rules:
- Prefer web_search+browse_url for factual queries.
- Use rag_search for anything about local docs/notes.
- Use write_note to save long-term info the user asks you to remember.
- Keep ACTION_INPUT JSON minimal and valid.
- Never invent tool names.
""".strip()

ACTION_RE = re.compile(r"^ACTION:\s*(?P<name>[a-z_]+)\s*\nACTION_INPUT:\s*(?P<json>\{[\s\S]*?\})\s*\Z", re.MULTILINE)
FINAL_RE = re.compile(r"^FINAL:\s*(?P<final>[\s\S]+)$", re.MULTILINE)

# ----------------------------
# Agent loop
# ----------------------------

def agent_reply(history: List[Dict[str,str]], user_msg: str) -> str:
    messages = [
        {"role":"system", "content": SYSTEM_PROMPT},
        *history,
        {"role":"user", "content": user_msg}
    ]
    reply = ollama_chat(messages)

    # Check for ACTION or FINAL
    m_final = FINAL_RE.search(reply)
    if m_final:
        return m_final.group("final").strip()

    m_act = ACTION_RE.search(reply)
    if m_act:
        name = m_act.group("name").strip()
        try:
            args = json.loads(m_act.group("json"))
        except json.JSONDecodeError:
            return "Model produced invalid ACTION_INPUT JSON."
        if name not in TOOLS:
            return f"Unknown tool: {name}"
        obs = TOOLS[name]["fn"](**args)
        # Feed observation back for finalization
        messages.append({"role":"assistant", "content": reply})
        messages.append({"role":"user", "content": f"OBSERVATION: {obs[:4000]}"})
        reply2 = ollama_chat(messages)
        m_final2 = FINAL_RE.search(reply2)
        if m_final2:
            return m_final2.group("final").strip()
        else:
            # Fallback: return raw text
            return reply2
    else:
        # If no action or final, just return the text
        return reply

# ----------------------------
# CLI
# ----------------------------

def main():
    console.print(Panel.fit(f"Free Local AI Agent — Model: [bold]{MODEL_NAME}[/bold]", subtitle="Type 'exit' to quit"))
    history: List[Dict[str,str]] = []
    # Warm tip
    console.print("[dim]Tip: Ask me anything. I can search the web, do math, and read your local docs (docs/).\n[/dim]")

    while True:
        try:
            user_msg = Prompt.ask("[bold cyan]You")
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break
        if not user_msg:
            continue
        if user_msg.strip().lower() in {"exit","quit"}:
            console.print("Goodbye!")
            break
        if user_msg.strip().lower() == "/help":
            console.print(Markdown("**Tools available**\n\n" + TOOL_DESCRIPTIONS))
            continue
        if user_msg.strip().lower() == "/reindex":
            if RAGGER.model is not None:
                RAGGER.refresh()
                console.print("[green]Rebuilt vector index.[/green]")
            else:
                console.print("[yellow]RAG disabled (sentence-transformers missing).[/yellow]")
            continue

        with console.status("Thinking..."):
            out = agent_reply(history, user_msg)
        history.append({"role":"user","content": user_msg})
        history.append({"role":"assistant","content": out})
        console.print(Markdown(out))

if __name__ == "__main__":
    main()
