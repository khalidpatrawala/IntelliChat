Perfect! Let’s create a full working agent with Stage 3 (Web Search Tool) added.
This will include:

Stage 2 memory (chat history saved to history.json)

Web search using DuckDuckGo + scraping for readable text

Basic chat using Ollama

You can run this as a standalone agent.py.
✅ How to use

Save this as agent.py inside C:\Users\admin\Desktop\my-agent\.

Install dependencies if you haven’t already:

pip install ollama rich duckduckgo-search requests beautifulsoup4 readability-lxml lxml


Run the agent:

python agent.py


Example commands:

You: Hi
You: search: latest AI news
You: How does Python work?
You: exit


Web search results are triggered with search: <query>

Chat messages are remembered in history.json