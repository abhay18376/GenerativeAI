import streamlit as st
import autogen
from autogen import AssistantAgent, UserProxyAgent
from io import BytesIO
from docx import Document
import csv
from fpdf import FPDF

st.set_page_config(page_title="Gemini Research Agent", layout="centered")

# Initialize session state for assistant messages
if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = []

# Load Gemini LLM config
config_list_gemini = autogen.config_list_from_json("model_config.json")

# Define the assistant agent
assistant = AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list_gemini, "seed": 42},
    max_consecutive_auto_reply=3,
    system_message="You are a helpful assistant."
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

user_proxy.receive = custom_receive.__get__(user_proxy)

st.title("Gemini Research Agent")

# Input for topic or question
topic = st.text_input("Enter your question or topic:")

# Button to ask question
if st.button("Ask"):
    if not topic.strip():
        st.warning("Please enter a question or topic.")
    else:
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

# Display assistant responses and download options
if st.session_state.assistant_messages:
    st.markdown("### Assistant Response:")
    for msg in st.session_state.assistant_messages:
        st.markdown(msg)

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
    import io
    csv_str_io = io.StringIO()
    writer = csv.writer(csv_str_io)
    writer.writerow(["Assistant Responses"])
    for msg in st.session_state.assistant_messages:
        writer.writerow([msg])
    csv_str = csv_str_io.getvalue()
    csv_bytes = csv_str.encode("utf-8")
    csv_io = BytesIO(csv_bytes)
    st.download_button(
        label="Download CSV",
        data=csv_io,
        file_name="assistant_response.csv",
        mime="text/csv",
    )

    # Download as PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for msg in st.session_state.assistant_messages:
        pdf.multi_cell(0, 10, txt=msg)
    pdf_bytes = pdf.output(dest="S").encode("utf-8")
    pdf_io = BytesIO(pdf_bytes)
    st.download_button(
        label="Download PDF",
        data=pdf_io,
        file_name="assistant_response.pdf",
        mime="application/pdf",
    )
