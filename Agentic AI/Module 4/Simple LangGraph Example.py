import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import gradio as gr

# Try to import LangGraph pieces — if unavailable, we'll fallback gracefully.
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except Exception as e:
    # If import fails, still allow the rest of the script to run.
    StateGraph = None
    END = None
    MemorySaver = None
    LANGGRAPH_AVAILABLE = False
    print("LangGraph import failed (falling back). Error:", e)

# Load environment variables
load_dotenv()

# -----------------------------
# 1️⃣ Define AgentState schema
# -----------------------------
class AgentState:
    def __init__(self, user_id: str, thread_id: str):
        self.user_id = user_id
        self.thread_id = thread_id
        self.messages: List[str] = []        # short-term memory
        self.user_history: List[Dict[str, Any]] = []  # long-term memory

    def append_message(self, message: str):
        self.messages.append(message)
        # Trim context to last 5 messages
        self.messages = self.messages[-5:]

# ---------------------------------
# 2️⃣ Implement Memory Management
# ---------------------------------
class PersistentMemory:
    def __init__(self, filename="memory_store.json"):
        self.filename = filename
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                json.dump({}, f)

    def load(self, user_id: str):
        with open(self.filename, "r") as f:
            data = json.load(f)
        return data.get(user_id, [])

    def save(self, user_id: str, history: List[Dict[str, Any]]):
        with open(self.filename, "r") as f:
            data = json.load(f)
        data[user_id] = history
        with open(self.filename, "w") as f:
            json.dump(data, f, indent=4)

# ---------------------------------
# 3️⃣ Mock LLM Response
# ---------------------------------
def mock_llm_response(query: str) -> str:
    faq = {
        "reset password": "You can reset your password by clicking 'Forgot Password' on the login page.",
        "change password": "Go to settings → account → change password.",
        "billing": "Our billing team can be reached at billing@techtrend.com."
    }
    for key in faq:
        if key in query.lower():
            return faq[key]
    return "I'm escalating this to a human agent for further review."

# ---------------------------------
# 4️⃣ Define Query Processing Node (business logic)
# ---------------------------------
def process_query(state: AgentState, query: str, memory: PersistentMemory):
    """
    Core node logic: append short-term message, call (mock) LLM,
    store to long-term memory and return a response string.
    """
    state.append_message(query)
    response = mock_llm_response(query)

    # Log to long-term memory
    entry = {"query": query, "response": response}
    state.user_history.append(entry)
    memory.save(state.user_id, state.user_history)

    return response

# ---------------------------------
# 5️⃣ LangGraph: create a simple graph and add a node
# ---------------------------------
graph: Optional["StateGraph"] = None  # type: ignore

if LANGGRAPH_AVAILABLE:
    try:
        # Create a StateGraph instance
        graph = StateGraph(name="techtrend_support_agent")

        # Many LangGraph installs expose methods like `add_node` or `add_action`.
        # We attach a node called "process_query_node" that calls our `process_query` function.
        # NOTE: the exact API for StateGraph may differ between versions. This demonstrates the
        # usual pattern (node registration => graph.run/execute), and includes a fallback below.
        try:
            # preferred API: add node by name with a callable
            graph.add_node(
                name="process_query_node",
                fn=lambda state, inp: process_query(state, inp, memory)  # capture `memory`
            )
            # set graph start and end nodes if API expects it
            graph.set_start("process_query_node")
            graph.set_end("process_query_node")  # a single-node graph — end is same node
        except Exception:
            # alternative API names
            try:
                graph.register_node("process_query_node", lambda state, inp: process_query(state, inp, memory))
                graph.start_node = "process_query_node"
            except Exception:
                # As last resort just attach a property so we can call it later
                graph._process_query_node = lambda state, inp: process_query(state, inp, memory)
    except Exception as e:
        print("Warning: failed to fully configure StateGraph. Continuing without graph runtime. Error:", e)
        graph = None
else:
    print("LangGraph not available — skipping graph configuration.")

# ---------------------------------
# 6️⃣ Gradio Interface (UI)
# ---------------------------------
memory = PersistentMemory()

def chat(user_id, message, history):
    # Build a fresh state per session (you can instead persist between calls by user_id)
    state = AgentState(user_id=user_id, thread_id="T1")
    state.user_history = memory.load(user_id)

    # If we have a usable LangGraph instance, route input through the graph's node.
    if graph is not None:
        try:
            # Try common run/execute API names
            if hasattr(graph, "run"):
                # many graph libs use run(start_node, input=...)
                # we pass state and message as parameters in whatever form the graph expects.
                # Here we attempt a few plausible signatures.
                try:
                    result = graph.run(start_node="process_query_node", state=state, input=message)
                except TypeError:
                    # try different signature
                    result = graph.run(state=state, input=message)
                # If result is a tuple or object, try to extract the textual response
                if isinstance(result, tuple) and len(result) > 0:
                    response = result[0]
                elif isinstance(result, str):
                    response = result
                else:
                    # fallback to direct call if run didn't return text
                    if hasattr(graph, "_process_query_node"):
                        response = graph._process_query_node(state, message)
                    else:
                        response = process_query(state, message, memory)
            elif hasattr(graph, "_process_query_node"):
                response = graph._process_query_node(state, message)
            else:
                # final fallback: call business logic directly
                response = process_query(state, message, memory)
        except Exception as e:
            # If graph execution fails, fallback and keep the app running.
            print("Graph execution failed, falling back to direct processing. Error:", e)
            response = process_query(state, message, memory)
    else:
        # No graph installed — call logic directly
        response = process_query(state, message, memory)

    history = history or []
    history.append((message, response))
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 TechTrend Customer Support Agent (LangGraph node integrated)")
    user_id = gr.Textbox(label="User ID", value="user123")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your message here...")
    clear = gr.Button("Clear Chat")

    msg.submit(chat, [user_id, msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
