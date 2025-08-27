It includes everything from Stage 5 (memory + web search + persistent RAG) plus tools:

Calculator (calc: <math>)

Python code runner (run: <python code>)

Save notes to notes/ (note:<filename>:<content>)

Reminders/To-Dos (reminder:<text> / list reminders)

Save this as agent.py inside your my-agent folder and run with python agent.py.

✅ How the full Stage 5+6 agent works
Command	What it does
rag: <query>	Answers using your documents intelligently (RAG)
search: <query>	Searches DuckDuckGo and summarizes top results
docs	Dumps your document contents (basic)
calc: <math>	Evaluates math expressions safely
run: <python code>	Executes small Python snippets
note:<filename>:<content>	Saves a note to notes/ folder
reminder:<text>	Adds a reminder
list reminders	Shows all reminders

Install (once):
pip install ollama rich duckduckgo-search requests beautifulsoup4 readability-lxml lxml python-docx PyPDF2 sentence-transformers faiss-cpu numpy

TO RUN:
python agent.py


Commands you can use:
help — list commands
search: cheapest flights from mumbai to dubai — web search
rag: summarize docs — answers using your docs
docs — preview documents in the docs/ folder
calc: 2*(5+3)**2 — calculator
run: print(2+2) — run a tiny Python snippet
note:todo.txt:buy milk at 6pm — save a note in notes/
reminder: Pay electricity bill — add a reminder
list reminders — show reminders