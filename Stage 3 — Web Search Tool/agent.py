import ollama
import os
import json
from rich.console import Console
from rich.markdown import Markdown
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from readability import Document

console = Console()
HISTORY_FILE = "history.json"

# ------------------------
# Memory functions
# ------------------------
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

# ------------------------
# Web search function
# ------------------------
def web_search(query, max_results=3):
    """Search DuckDuckGo and return top results as readable summaries"""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            url = r.get("href")
            if not url:
                continue
            try:
                resp = requests.get(url, timeout=10)
                doc = Document(resp.text)
                title = doc.title()
                summary = doc.summary()
                text = BeautifulSoup(summary, "lxml").get_text()
                results.append(f"{title}\n{text[:500]}... [source: {url}]")
            except Exception as e:
                results.append(f"Could not fetch {url} ({e})")
    return results

# ------------------------
# Agent reply function
# ------------------------
def agent_reply(history, user_msg):
    user_msg_lower = user_msg.lower().strip()

    # Web search trigger
    if user_msg_lower.startswith("search:"):
        query = user_msg.split("search:",1)[1].strip()
        results = web_search(query)
        return "\n\n".join(results) if results else "No results found."

    # Normal chat with Ollama
    history.append({"role": "user", "content": user_msg})
    response = ollama.chat(
        model="llama3.1:8b",
        messages=history
    )
    reply = response["message"]["content"]
    history.append({"role": "assistant", "content": reply})
    return reply

# ------------------------
# Main loop
# ------------------------
def main():
    console.print("[bold green]AI Agent with Memory & Web Search![/bold green]")
    console.print("Type 'exit' to quit. For web search, type: search: <your query>\n")

    history = load_history()

    while True:
        user_msg = input("\nYou: ").strip()
        if not user_msg:
            continue

        if user_msg.lower() in ["exit", "quit", "bye"]:
            console.print("[red]Goodbye![/red]")
            save_history(history)
            break

        with console.status("Thinking..."):
            out = agent_reply(history, user_msg)

        console.print(Markdown(out))
        save_history(history)

if __name__ == "__main__":
    main()
