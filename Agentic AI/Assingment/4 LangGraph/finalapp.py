import os
import time
import json
from uuid import uuid4
import gradio as gr
from typing import Annotated, Dict, Any, TypedDict, List
from langgraph.graph import StateGraph, START
from operator import add

try:
    from langgraph.store.memory import InMemoryStore  # optional
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

# -------------------------
# 1) Message / State schema
# -------------------------
class Message(TypedDict, total=False):
    """Represents a single message in the chat history."""
    role: str          # "user" | "assistant" | "system"
    content: str
    filtered: bool     # whether this message was filtered (e.g., greeting)

class State(TypedDict):
    messages: Annotated[List[Message], add]
    user_id: str
    thread_id: str
    escalate: bool  # used here as "filtered" flag for last user msg

# --------------------------------
# 2) Persist conversation to a file
# --------------------------------

def save_to_external_memory(state: Dict[str, Any], filename: str = "memory_store.json"):
    user_id = state.get("user_id", "anonymous")
    messages: List[Message] = state.get("messages", [])

    if not os.path.exists(filename):
        with open(filename, "w") as f:
            json.dump({}, f, indent=4)

    # Load existing
    with open(filename, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {}

    # Prepare entries
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    convo_entry = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        filtered = bool(msg.get("filtered", False))
        convo_entry.append({
            "role": role,
            "content": content,
            "filtered": filtered,
            "timestamp": ts
        })

    data.setdefault(user_id, []).extend(convo_entry)

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    return {}  # node convention

# ------------------------------
# 3) Filter node (greeting etc.)
# ------------------------------
def filter_node(state: State):
    last = state["messages"][-1] if state.get("messages") else {"role": "user", "content": ""}
    # last is a dict per our schema
    text = str(last.get("content", "")).strip().lower()

    greetings = {"hi", "hello", "hey", "thanks", "thank you"}
    is_greeting = text in greetings

    # Mark that this should be treated as filtered downstream
    return {"escalate": is_greeting}

# --------------------------
# 4) Mock LLM + LLM node
# --------------------------
def mock_llm_response(query: str) -> str:
    faq = {
        "reset password": "You can reset your password by clicking 'Forgot Password' on the login page.",
        "change password": "Go to settings → account → change password.",
        "billing": "Our billing team can be reached at billing@techtrend.com."
    }
    ql = (query or "").lower()
    for key, val in faq.items():
        if key in ql:
            return val
    return "I'm escalating this to a human agent for further review."

def llm_node(state: State):
    msgs: List[Message] = state.get("messages", [])
    user_msg = msgs[-1] if msgs else {"role": "user", "content": ""}

    user_text = str(user_msg.get("content", ""))
    reply_text = mock_llm_response(user_text)

    filtered_flag = bool(state.get("escalate", False))
    if filtered_flag:
        reply_text = reply_text + " --Filtered"

    assistant_message: Message = {
        "role": "assistant",
        "content": reply_text,
        "filtered": filtered_flag
    }

    new_messages = [assistant_message]

    # Clear the flag after using it (so it doesn't leak into future turns)
    return {"messages": new_messages, "escalate": False}

# --------------------------
# 5) Memory trimmer node
# --------------------------
def memory_trimmer(state: State):
    """
    Keep only last 4 messages to manage context length.
    """
    messages = state.get("messages", [])
    if len(messages) > 4:
        return {"messages": messages[-4:]}
    return {}

# --------------------------
# 6) Build and compile graph
# --------------------------
builder = StateGraph(State)
builder.add_node(filter_node)
builder.add_node(llm_node)
builder.add_node(memory_trimmer)
builder.add_node(save_to_external_memory)

builder.add_edge(START, "filter_node")
builder.add_edge("filter_node", "llm_node")
builder.add_edge("llm_node", "memory_trimmer")
builder.add_edge("memory_trimmer", "save_to_external_memory")

graph = builder.set_entry_point("filter_node").compile()

# --------------------------
# 7) Gradio UI
# --------------------------
with gr.Blocks() as demo:
    gr.Markdown("# LangGraph Customer Support (Gradio + LangGraph)")
    user_id_input = gr.Textbox(label="User ID", value="user_123")
    session_id_input = gr.Textbox(label="Session ID", value=str(uuid4()))
    chatbox = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your message here...")
    send = gr.Button("Send")
    send.click(lambda: "", None, msg)

    def send_click(session_id, user_id, message, chat_history):
    
        if not session_id:
            session_id = str(uuid4())
        user_id = user_id or "anonymous"
        chat_history = chat_history or []

        message = (message or "").strip()
        if not message:
            return session_id, chat_history

        input_state: State = {
            "messages": [{"role": "user", "content": message}],
            "user_id": user_id,
            "thread_id": session_id,
            "escalate": False
        }

        try:
            result = graph.invoke(input_state)
        except Exception as e:
            fallback = "Sorry — an internal error occurred while processing your message."
            chat_history.append((message, fallback))
            return session_id, chat_history

        if not isinstance(result, dict):
            ai_reply = "Sorry — received an unexpected response from the workflow."
            chat_history.append((message, ai_reply))
            return session_id, chat_history

        msgs = result.get("messages", []) or []

        # Extract the LAST assistant reply safely from dict-style messages
        assistant_texts = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]
        ai_reply = assistant_texts[-1] if assistant_texts else "Sorry — I couldn't generate a reply."

        chat_history.append((message, ai_reply))
        return session_id, chat_history

    send.click(send_click, inputs=[session_id_input, user_id_input, msg, chatbox],
               outputs=[session_id_input, chatbox])

if __name__ == "__main__":
    demo.launch()
