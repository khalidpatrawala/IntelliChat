import ollama
import os
import json
from rich.console import Console
from rich.markdown import Markdown

console = Console()
HISTORY_FILE = "history.json"

# Load history if available
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# Save history to file
def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

# Function to get AI reply
def agent_reply(history, user_msg):
    # Add user message to history
    history.append({"role": "user", "content": user_msg})

    # Call Ollama with full conversation
    response = ollama.chat(
        model="llama3.1:8b",
        messages=history
    )

    # Extract assistant reply
    reply = response["message"]["content"]

    # Add assistant reply to history
    history.append({"role": "assistant", "content": reply})

    return reply

def main():
    console.print("[bold green]AI Agent Ready with Memory![/bold green]")
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
