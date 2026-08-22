import streamlit as st
import autogen
from autogen import AssistantAgent, UserProxyAgent
from io import BytesIO
from docx import Document
import csv
from fpdf import FPDF
import io

st.set_page_config(page_title="MediSyn Labs Research Agent", layout="centered")

# --- AgentState Structure and Initialization ---

# Define a class for AgentState (Optional, but good practice for structure)
class ResearchAgentState:
    def __init__(self, researcher_id, project_id, disease_focus):
        self.researcher_id = researcher_id
        self.project_id = project_id
        self.disease_focus = disease_focus
        # Short-term memory: Active session queries and responses
        self.short_term_memory = []
        # Long-term memory: Stored summaries (Placeholder in session state)
        self.long_term_memory = {
            "summaries": [],
            "comparative_findings": [],
            "case_history": []
        }

# Initialize session state for the AgentState object and autogen messages
if "agent_state" not in st.session_state:
    # Initialize with default/dummy metadata
    st.session_state.agent_state = ResearchAgentState("R_001", "P_MEDISYN_023", "Alzheimer's Disease")

# A separate list for assistant messages to keep the existing display logic simple
if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = []

# --- Configuration Constants for Memory Management ---
STM_MAX_SIZE = 7  # Maximum number of clinically relevant entries in Short-Term Memory
VAGUE_QUERY_THRESHOLD = 15 # Minimum character length for a query to be considered 'informational'


# --- Helper Function for Filtering Vague Queries ---
def is_clinically_relevant(text: str) -> bool:
    """
    Checks if a query is relevant/informational based on length and simple keywords.
    This is a basic filter; a more robust filter would use an LLM or NLU service.
    """
    text = text.strip().lower()
    # 1. Check length
    if len(text) < VAGUE_QUERY_THRESHOLD:
        return False

    # 2. Simple keyword check for common non-informational inputs
    non_info_keywords = ["hello", "hi", "thanks", "ok", "yes", "no", "bye", "good job"]
    if any(keyword in text for keyword in non_info_keywords):
        return False

    return True


# --- Agent Configuration (Existing) ---

# Load Gemini LLM config
# NOTE: Ensure 'model_config.json' is available and correctly formatted for your LLM setup.
try:
    config_list_gemini = autogen.config_list_from_json("model_config.json")
except FileNotFoundError:
    st.error("model_config.json not found. Please create it for AutoGen configuration.")
    config_list_gemini = [{"model": "gemini-2.5-flash", "api_key": "dummy"}] # Fallback

# Define the assistant agent
assistant = AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list_gemini, "seed": 42},
    max_consecutive_auto_reply=3,
    system_message="You are a helpful research assistant for MediSyn Labs."
)

# Define user proxy agent with custom receive function to collect messages in Streamlit state
user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    human_input_mode="ALWAYS",
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
)

def custom_receive(self, message, sender, request_reply, silent):
    content = message.get("content", "") if isinstance(message, dict) else message
    st.session_state.assistant_messages.append(content)

    # --- Short-Term Memory Update and Trimming ---

    # 1. Store the new response
    new_entry = {
        "sender": sender.name,
        "content": content,
        "timestamp": st.session_state.agent_state.short_term_memory.__len__() # Simple index as timestamp
    }
    st.session_state.agent_state.short_term_memory.append(new_entry)

    # 2. Trim Short-Term Memory to the limit
    # This keeps only the 'STM_MAX_SIZE' most recent entries.
    if len(st.session_state.agent_state.short_term_memory) > STM_MAX_SIZE:
        st.session_state.agent_state.short_term_memory = st.session_state.agent_state.short_term_memory[-STM_MAX_SIZE:]
        # Optional: Add a debug message to show trimming
        # st.sidebar.info(f"STM trimmed to {STM_MAX_SIZE} entries.")

user_proxy.receive = custom_receive.__get__(user_proxy)


# --- Streamlit UI and Logic ---

st.title("Gemini Research Agent 🔬")

# Display and allow editing of Metadata fields
st.sidebar.header("Agent Metadata")
st.session_state.agent_state.researcher_id = st.sidebar.text_input(
    "Researcher ID", st.session_state.agent_state.researcher_id
)
st.session_state.agent_state.project_id = st.sidebar.text_input(
    "Project ID", st.session_state.agent_state.project_id
)
st.session_state.agent_state.disease_focus = st.sidebar.text_input(
    "Disease Focus", st.session_state.agent_state.disease_focus
)

# Input for topic or question
topic = st.text_input("Enter your question or topic:")

# Button to ask question
if st.button("Ask"):
    if not topic.strip():
        st.warning("Please enter a question or topic.")
    else:
        # --- Filtering Logic Applied to User Input ---
        if not is_clinically_relevant(topic):
            st.error("Input filtered: Please enter a more detailed, clinically relevant query.")
        else:
            # Log the initial query to short-term memory before chat initiation
            st.session_state.agent_state.short_term_memory.append({
                "sender": user_proxy.name,
                "content": topic,
                "timestamp": st.session_state.agent_state.short_term_memory.__len__()
            })
            
            # Trim memory immediately after adding user's query
            if len(st.session_state.agent_state.short_term_memory) > STM_MAX_SIZE:
                 st.session_state.agent_state.short_term_memory = st.session_state.agent_state.short_term_memory[-STM_MAX_SIZE:]

            st.session_state.assistant_messages = []
            user_proxy.initiate_chat(assistant, message=topic)

# Button to generate subtopics for exploration
if st.button("Generate Subtopics"):
    if not st.session_state.assistant_messages:
        st.warning("Please ask a question first.")
    else:
        last_response = st.session_state.assistant_messages[-1]
        subtopic_prompt = (
            "Please generate a list of subtopics or themes based on the following text.\n\n"
            f"{last_response}\n\n"
            "List them in bullet points."
        )
        st.session_state.assistant_messages = []
        user_proxy.initiate_chat(assistant, message=subtopic_prompt)

# Button to summarize the last response
if st.button("Summarise"):
    if not st.session_state.assistant_messages:
        st.warning("Please ask a question first.")
    else:
        last_response = st.session_state.assistant_messages[-1]
        summarise_prompt = f"Please summarise the following text concisely:\n\n{last_response}"
        st.session_state.assistant_messages = []
        user_proxy.initiate_chat(assistant, message=summarise_prompt)


# --- Display and Download Options ---

if st.session_state.assistant_messages:
    st.markdown("### Assistant Response:")
    for msg in st.session_state.assistant_messages:
        st.markdown(msg)

    st.markdown("---")
    st.subheader("Download Options")

    # ... (Download code for Word, CSV, PDF remains largely the same) ...

    # Download as Word document
    doc = Document()
    for msg in st.session_state.assistant_messages:
        doc.add_paragraph(msg)
    word_io = BytesIO()
    doc.save(word_io)
    word_io.seek(0)
    st.download_button(
        label="Download Word",
        data=word_io,
        file_name="assistant_response.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    # Download as CSV
    csv_str_io = io.StringIO()
    writer = csv.writer(csv_str_io)
    writer.writerow(["Agent", "Response Content"])
    # Write from the session's short-term memory for a richer CSV
    for entry in st.session_state.agent_state.short_term_memory:
        writer.writerow([entry['sender'], entry['content']])
    csv_str = csv_str_io.getvalue()
    csv_bytes = csv_str.encode("utf-8")
    csv_io = BytesIO(csv_bytes)
    st.download_button(
        label="Download CSV",
        data=csv_io,
        file_name="session_chat_history.csv",
        mime="text/csv",
    )

    # Download as PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    # Add metadata to PDF
    pdf.multi_cell(0, 10, txt=f"Project ID: {st.session_state.agent_state.project_id}")
    pdf.multi_cell(0, 10, txt=f"Researcher ID: {st.session_state.agent_state.researcher_id}")
    pdf.multi_cell(0, 10, txt=f"Disease Focus: {st.session_state.agent_state.disease_focus}\n")
    pdf.multi_cell(0, 10, txt="--- Session Responses ---\n")
    for msg in st.session_state.assistant_messages:
        try:
            msg_encoded = msg.encode('latin-1', 'replace').decode('latin-1')
        except:
            msg_encoded = "Content encoding error."
        pdf.multi_cell(0, 10, txt=msg_encoded)

    pdf_s = pdf.output(dest="S")
    if isinstance(pdf_s, bytes):
        pdf_bytes = pdf_s
    else:
        pdf_bytes = pdf_s.encode("latin-1")
    pdf_io = BytesIO(pdf_bytes)
    st.download_button(
        label="Download PDF",
        data=pdf_io,
        file_name="assistant_response.pdf",
        mime="application/pdf",
    )

# --- Display Short-term and Long-term Memory for debugging/visualization ---
st.sidebar.markdown("---")
st.sidebar.subheader("Memory Status (Debug)")

# Short-term Memory (STM)
with st.sidebar.expander(f"Short-Term Memory (Session Log - Max {STM_MAX_SIZE})"):
    for item in st.session_state.agent_state.short_term_memory:
        st.sidebar.markdown(f"**{item['sender']}** ({item['timestamp']}): *{item['content'][:30]}...*")

# Long-term Memory (LTM) Placeholder
with st.sidebar.expander("Long-Term Memory (LTM)"):
    st.sidebar.json(st.session_state.agent_state.long_term_memory)
    # Button to add the last summary to LTM (example action)
    if st.button("Store Last Response in LTM (Summary)"):
        if st.session_state.assistant_messages:
            last_response = st.session_state.assistant_messages[-1]
            st.session_state.agent_state.long_term_memory["summaries"].append({
                "source_query": st.session_state.agent_state.short_term_memory[-2]['content'] if len(st.session_state.agent_state.short_term_memory) >= 2 else "N/A",
                "summary": last_response,
                "project": st.session_state.agent_state.project_id
            })
            st.sidebar.success("Stored as LTM Summary! 💾")
        else:
            st.sidebar.warning("No response to store.")