# Single-cell deploy of the 4-agent pipeline using LangGraph + ChatOpenAI
import os
import time
import json
import uuid
import requests
from typing import Any, Dict, List
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import BaseTool
from dotenv import load_dotenv
# --- (optional) re-create model if needed - comment out if model already exists in session ---
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o", temperature=0.6)

# --- small utilities ---
def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

# --- Tools -----------------------------------------------------------------
def web_search_tool(query: str) -> str:
    """
    Tool used by TopicResearcher to gather 'search snippets'.
    If SERPAPI_API_KEY env var is present, it attempts a live SERPAPI request.
    Otherwise returns a deterministic, readable mock of search snippets.
    """
    serp_key = os.getenv("SERPAPI_API_KEY")
    if serp_key:
        try:
            params = {"engine": "google", "q": query, "api_key": serp_key, "num": 8}
            r = requests.get("https://serpapi.com/search.json", params=params, timeout=8)
            r.raise_for_status()
            data = r.json()
            items = []
            for item in data.get("organic_results", [])[:8]:
                t = item.get("title") or ""
                s = item.get("snippet") or item.get("description") or ""
                items.append(f"- {t} — {s}")
            return "\n".join(items) if items else _mock_snippets(query)
        except Exception as e:
            # fallback to mock if any error
            return _mock_snippets(query)
    else:
        return _mock_snippets(query)

def _mock_snippets(query: str) -> str:
    base = [
        f"{query} — Industry forecasts and ROI studies.",
        f"How {query} impacts business operations and decision making.",
        f"{query}: Policy, regulation and governance considerations.",
        f"Tools and platforms enabling {query} adoption.",
        f"Case studies: companies using {query} for efficiency.",
        f"Predictions for {query} in 2026 and beyond."
    ]
    return "\n".join(f"- {s}" for s in base)

class WriteJSONTool(BaseTool):
    """
    A LangChain-style tool used by the OutputWriter agent.
    It expects a single JSON string (or a short description) and writes it to /mnt/data.
    """
    name: str = "write_json"
    description: str = "Write provided JSON string to a file and return the saved path."

    def _run(self, tool_input: str, **kwargs: Any) -> str:
        # tool_input should be a JSON string or pretty JSON; attempt to parse, otherwise wrap it.
        try:
            obj = json.loads(tool_input)
        except Exception:
            # if parsing fails, store as message under 'content'
            obj = {"content": tool_input}
        fname = f"multi_agent_output_{uuid.uuid4().hex[:8]}.json"
        with open(fname, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
        return f"wrote:{fname}"

    async def _arun(self, tool_input: str, **kwargs: Any) -> str:
        return self._run(tool_input, **kwargs)

write_json_tool = WriteJSONTool()

# --- Agent prompts and creation ------------------------------------------
# TopicResearcher: call web_search_tool(query) and return a JSON block of keywords
topic_researcher_prompt = """
You are TopicResearcher. Objective:
1) Use the provided web_search tool to fetch search snippets for the user's topic.
2) From those snippets, produce a JSON object with the following structure (ONLY return the JSON object inside a fenced block or plain JSON):
{
  "topic": "<topic>",
  "search_summary": "<one-line summary of findings>",
  "seed_keywords": [
    {"keyword": "phrase1", "intent": "Informational|Commercial|Transactional|Navigational", "note": "one-line rationale"},
    ...
  ]
}
Make sure seed_keywords has 6-12 items and each item has keyword, intent, and a one-line note.
"""

topic_researcher = create_react_agent(
    model,
    tools=[web_search_tool],
    prompt=topic_researcher_prompt,
)

# ContentWriter: reads previous JSON (TopicResearcher output) and produces a draft JSON
content_writer_prompt = """
You are ContentWriter. Read the conversation history to find the TopicResearcher JSON output
(there will be a JSON object containing 'seed_keywords'). Using that data, produce a content draft JSON in this exact structure:

{
  "title": "<article title>",
  "intro": "<one paragraph intro>",
  "sections": [
    {"heading": "<heading1>", "body": "<paragraph body>"},
    ...
  ],
  "conclusion": "<one-paragraph conclusion>"
}

Prefer about 3-5 sections. Return only the JSON object (no commentary).
"""

content_writer = create_react_agent(
    model,
    tools=[],  # no external tools needed — content is generated by the LLM
    prompt=content_writer_prompt,
)

# SEOOptimizer: reads the draft JSON and add seo metadata, return JSON with 'seo' block
seo_optimizer_prompt = """
You are SEOOptimizer. Read the conversation history to find the content draft JSON produced by ContentWriter.
Produce a JSON object with these fields:

{
  "title": "<possibly improved title>",
  "meta_description": "<short meta, 140-155 chars>",
  "slug": "<url-friendly-slug>",
  "tags": ["tag1","tag2", ...],
  "optimized_sections": [
    {"heading": "<heading>", "body": "<body (may include keyword mentions)>"},
    ...
  ],
  "keyword_density_hints": {
    "keyword1": "recommended_count",
    ...
  }
}

Return only the JSON object.
"""

seo_optimizer = create_react_agent(
    model,
    tools=[],
    prompt=seo_optimizer_prompt,
)

# OutputWriter: expects the combined content + seo in conversation history and must call write_json tool
output_writer_prompt = """
You are OutputWriter. Read the conversation history to find:
- TopicResearcher JSON (seed_keywords)
- ContentWriter JSON (draft)
- SEOOptimizer JSON (seo)

Combine these into a final JSON publish package with this structure:

{
  "topic": "<topic>",
  "research": { ... },      // the TopicResearcher JSON
  "draft": { ... },         // the ContentWriter JSON
  "seo": { ... },           // the SEOOptimizer JSON
  "generated_at": "<timestamp>"
}

Once the final JSON object is prepared, CALL the provided tool 'write_json' with the final JSON as a string (no extra commentary). The tool will return the file path. Finally, return a short acknowledgement message (one line) that includes the saved file path.
"""

output_writer = create_react_agent(
    model,
    tools=[write_json_tool],
    prompt=output_writer_prompt,
)

# --- State / graph wiring -------------------------------------------------
class MultiAgentState(MessagesState):
    last_active_agent: str

def call_topic_researcher(state: MultiAgentState) -> Command:
    response = topic_researcher.invoke(state)
    update = {**response, "last_active_agent": "topic_researcher"}
    return Command(update=update, goto="content_writer")

def call_content_writer(state: MultiAgentState) -> Command:
    response = content_writer.invoke(state)
    update = {**response, "last_active_agent": "content_writer"}
    return Command(update=update, goto="seo_optimizer")

def call_seo_optimizer(state: MultiAgentState) -> Command:
    response = seo_optimizer.invoke(state)
    update = {**response, "last_active_agent": "seo_optimizer"}
    return Command(update=update, goto="output_writer")

def call_output_writer(state: MultiAgentState) -> Command:
    response = output_writer.invoke(state)
    update = {**response, "last_active_agent": "output_writer"}
    # After output_writer runs the tool, it will have called write_json and return a message that includes the write_json response.
    # Send user control back to 'human' for any further actions.
    return Command(update=update, goto="human")

def human_node(state: MultiAgentState, config) -> Command:
    # In our demo we will just stop or allow for further inputs; the graph requires a human node.
    user_input = interrupt(value="Pipeline complete. Ready for next action.")
    active_agent = state["last_active_agent"]
    return Command(
        update={"messages": [{"role": "human", "content": user_input}]},
        goto=active_agent,
    )

# Build graph
builder = StateGraph(MultiAgentState)
builder.add_node("topic_researcher", call_topic_researcher)
builder.add_node("content_writer", call_content_writer)
builder.add_node("seo_optimizer", call_seo_optimizer)
builder.add_node("output_writer", call_output_writer)
builder.add_node("human", human_node)
builder.add_edge(START, "topic_researcher")

# compile with a simple memory saver
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# --- Demo run --------------------------------------------------------------
# Provide initial user message prompting the pipeline. Topic is "Importance of AI in 2026"
initial_user_message = {"messages": [{"role": "user", "content": "Topic: Importance of AI in 2026. Please research, write a draft, optimize for SEO, and save final JSON."}]}
thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

print("\n=== Running pipeline: TopicResearcher -> ContentWriter -> SEOOptimizer -> OutputWriter ===\n")

# Stream updates; LangGraph yields incremental updates for each node run.
for update in graph.stream(initial_user_message, config=thread_config, stream_mode="updates"):
    # update is a dict mapping node_id -> node_state
    for node_id, val in update.items():
        if isinstance(val, dict) and val.get("messages"):
            # print the last AI message content for tracing
            last_msg = val["messages"][-1]
            try:
                # last_msg could be a LangGraph message object; access .content if present, otherwise treat as string
                content = last_msg.content if hasattr(last_msg, "content") else last_msg
            except Exception:
                content = str(last_msg)
            print(f"[{node_id}] -> {str(content)[:1000]}\n")  # print a slice to avoid massive dumps

print("\n=== Pipeline finished. Check for saved output JSON files ===\n")
