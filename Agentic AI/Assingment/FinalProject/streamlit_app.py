import streamlit as st
import os
import csv
import json
import uuid
import sys
from io import StringIO
from typing import Any, List, Dict, Optional
from datetime import datetime

# --- CrewAI and LLM Imports (Assume all are installed) ---
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- GLOBAL STATIC DATA STORES ---
# Used for shared state and content storage before final disk write
READER_DATA_STORE: List[Dict[str, Any]] = []
WRITER_OUTPUT_STORE: Dict[str, str] = {}
PROCESSING_LOG: List[Dict[str, Any]] = []

# --- LLM Setup ---
# Initialize LLM only if GEMINI_API_KEY is available
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    llm = LLM(
        model="gemini/gemini-2.5-flash", 
        api_key=GEMINI_API_KEY,
        temperature=0.7
    )
else:
    st.error("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
    llm = None # Assign None or a mock LLM for the app to load

# --- Helper for Logging (Capturing CrewAI's verbose output) ---
class StreamlitLogCapture:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.log_content = ""
        self.original_stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout

    def write(self, data):
        self.log_content += data
        self.placeholder.code(self.log_content, language="text") # Update the UI on every write
        self.original_stdout.write(data)

    def flush(self):
        pass # Required for file-like object

# --- Tool Implementations (Modified to use fixed files for simplicity, 
# as the original code was hardcoded to read from disk) ---

class CSVReaderTool(BaseTool):
    name: str = "CSV Reader Tool"
    description: str = "Reads and loads support_emails.csv and app_store_reviews.csv, merges them, and stores the unified data in READER_DATA_STORE. Expects a single string of file paths (e.g., 'file1.csv, file2.csv')."

    def _read_csv(self, content: str, source_type: str) -> List[Dict[str, Any]]:
        """Helper function to read a single CSV string and add a 'source_type' tag."""
        data = []
        try:
            reader = csv.DictReader(StringIO(content.strip()))

            for row in reader:
                # Standardize row keys for unified data
                standardized_row = {
                    'source_id': row.get('review_id') or row.get('email_id'),
                    'source_type': source_type,
                    'content_raw': row.get('review_text') or row.get('body'),
                    'platform': row.get('platform', 'N/A'),
                    'rating': row.get('rating', 'N/A'),
                    'subject': row.get('subject', 'N/A'),
                    **row
                }
                data.append(standardized_row)
        except Exception as e:
            print(f"Error reading {source_type} data: {e}") 
        return data

    def _run(self, file_paths: str) -> str:
        """Reads dual CSV content, populates the global READER_DATA_STORE, and returns a summary."""
        global READER_DATA_STORE
        
        try:
            # Read from uploaded files stored in session state
            app_store_content = st.session_state['app_reviews_content'].getvalue().decode('utf-8')
            support_email_content = st.session_state['support_emails_content'].getvalue().decode('utf-8')

            app_reviews = self._read_csv(app_store_content, 'review')
            support_emails = self._read_csv(support_email_content, 'email')
            
            READER_DATA_STORE.clear()
            READER_DATA_STORE.extend(app_reviews)
            READER_DATA_STORE.extend(support_emails)
            
            total_count = len(READER_DATA_STORE)
            
            if total_count == 0:
                return "ERROR: Successfully read, but no data rows were found."

            return (
                f"SUCCESS: Loaded {len(app_reviews)} reviews and {len(support_emails)} emails, "
                f"totaling {total_count} records. The unified data is ready for processing."
            )
        except Exception as e:
            return f"CRITICAL ERROR in CSVReaderTool: Could not access uploaded file content. Details: {e}"


class CSVWriterTool(BaseTool):
    name: str = "CSV Writer Tool"
    description: str = "Writes a list of dictionaries to a specified CSV file, ensuring content is logged and saved. Arguments: data (list of dicts), output_file (str)."

    def _run(self, data: Any, output_file: str) -> str:
        """Converts a list of dictionaries into CSV format and logs the result to WRITER_OUTPUT_STORE."""
        global WRITER_OUTPUT_STORE

        if not data or not isinstance(data, list) or not all(isinstance(i, dict) for i in data):
            return f"ERROR: Input data for '{output_file}' is empty or not a list of dictionaries."
        
        try:
            all_keys = set()
            for row in data:
                all_keys.update(row.keys())
            fieldnames = list(all_keys)

            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
            
            writer.writeheader()
            writer.writerows(data)
            
            content_to_write = output.getvalue()
            
            # Store content globally for later download
            WRITER_OUTPUT_STORE[output_file] = content_to_write
                
            return f"SUCCESS: Logged {len(data)} records to {output_file}. Content is ready for final disk write."
        
        except Exception as e:
            return f"ERROR: Failed to write data to {output_file}. Details: {e}"

# Initialize tools
csv_reader = CSVReaderTool()
csv_writer = CSVWriterTool()

# --- Agents and Tasks (Unchanged from appfinal.py) ---
# ... (Include all your Agent and Task definitions here) ...
# To keep this response brief, the Agent/Task definitions are assumed to be copied over
# from the 'appfinal.py' file.

# --- Agent and Task Definitions from appfinal.py ---

csv_reader_agent = Agent(
    role='CSV Reader Agent',
    goal='Ingest and merge data from app_store_reviews.csv and support_emails.csv, providing a unified list of dictionaries.',
    backstory=("You are a data ingestion specialist, responsible for standardizing and combining data from multiple platforms into a single, clean source."),
    tools=[csv_reader],
    verbose=True, allow_delegation=False, llm=llm
)

feedback_classifier_agent = Agent(
    role='Feedback Classifier Agent',
    goal='Accurately categorize every record in the unified list into one of the following: Bug, Feature Request, Praise, Complaint, or Spam.',
    backstory=("You are a master of NLP and sentiment analysis, ensuring every piece of feedback is correctly tagged for the downstream processes."),
    llm=llm
)

bug_analysis_agent = Agent(
    role='Bug Analysis Agent',
    goal='For every Bug and Complaint, extract all technical details (device, os, repro_steps, platform) and assign an initial priority (Critical, High, Medium, Low).',
    backstory=("You are a technical analyst who can read unstructured text and pull out key diagnostic information needed for development."),
    llm=llm
)

feature_extractor_agent = Agent(
    role='Feature Extractor Agent',
    goal='For all Feature Request records, assign a feature_impact_score (1-10) and identify key business keywords.',
    backstory=("You are a product manager focused on roadmap development, translating user requests into actionable priorities."),
    llm=llm
)

ticket_creator_agent = Agent(
    role='Ticket Creator Agent',
    goal='Filter out all Spam and Praise records. For the remainder, generate a final, structured list of ticket dictionaries.',
    backstory=("You are an expert ticketing system engineer, adhering to a strict schema to prepare data for final logging."),
    llm=llm
)

quality_critic_agent = Agent(
    role='Quality Critic Agent',
    goal='Review the final ticket data for completeness, required fields, and logical consistency. Log the final tickets, a processing log, and metrics to three separate CSV files.',
    backstory=("You are a meticulous QA specialist who guarantees data quality and compliance, responsible for the final logging action."),
    tools=[csv_writer],
    llm=llm
)

# --- Tasks ---

ingestion_task = Task(
    description=(
        "1. Use the CSV Reader Tool with the file paths 'app_store_reviews.csv, support_emails.csv' "
        "as a single comma-separated string argument to the tool. "
        "2. The tool will read and merge data from both files. "
        "3. Output the final contents of the global READER_DATA_STORE list of dictionaries as the result."
    ),
    expected_output="A python list of dictionaries, unified from both CSV files, with standardized keys like 'source_id', 'source_type', and 'content_raw'.",
    agent=csv_reader_agent,
)

classification_task = Task(
    description=(
        "Analyze the list of unified feedback dictionaries. For each record, add the 'category' key with one of: "
        "'Bug', 'Feature Request', 'Praise', 'Complaint', or 'Spam'. The output must be the complete, updated list of dictionaries."
    ),
    expected_output="A python list of dictionaries, where every item now includes a 'category' key.",
    agent=feedback_classifier_agent,
)

bug_analysis_task = Task(
    description=(
        "Review the categorized list. For records tagged 'Bug' or 'Complaint', extract the following details (use 'N/A' if missing): "
        "'device', 'os', 'repro_steps'. Also, assign an initial 'priority' key with 'Critical', 'High', 'Medium', or 'Low' based on severity. Output the complete, updated list."
    ),
    expected_output="The updated python list of dictionaries, with Bug/Complaint entries containing 'device', 'os', 'repro_steps', and 'priority'.",
    agent=bug_analysis_agent,
)

feature_extraction_task = Task(
    description=(
        "Review the processed list. For 'Feature Request' entries, add a 'feature_impact_score' (1-10) and a 'keywords' list (e.g., ['dark mode', 'UI']). Skip other categories. Output the complete, updated list of dictionaries."
    ),
    expected_output="The updated python list of dictionaries, with Feature Requests including 'feature_impact_score' (int) and 'keywords' (list).",
    agent=feature_extractor_agent,
)

ticket_generation_task = Task(
    description=(
        "Filter out all 'Spam' and 'Praise' records. Generate the final structured ticket data for the remainder. "
        "Schema MUST be: [ticket_id (unique UUID), category, priority, summary (short title), description (full body), source_id, source_type, device, os, feature_impact_score (null if not Feature Request)]. "
        "The final output must be a clean, parsable Python list of dictionaries."
    ),
    expected_output="A clean, structured python list of ticket dictionaries, following the required schema, with no Spam or Praise entries.",
    agent=ticket_creator_agent,
)

qa_and_logging_task = Task(
    description=(
        "1. **QA Review**: Critically review the ticket dictionaries from the previous step for completeness and schema adherence. "
        "2. **Log Tickets**: Call the **CSV Writer Tool** with **`data=final_tickets_list`** and **`output_file='generated_tickets.csv'`**. "
        "3. **Log Processing**: Construct a 'processing_log' list of dictionaries (one row per original record) and call the **CSV Writer Tool** with **`data=processing_log_list`** and **`output_file='processing_log.csv'`**. "
        "4. **Log Metrics**: Construct a 'metrics' list (e.g., [{'metric_name': 'Total Tickets', 'value': N}]) and call the **CSV Writer Tool** with **`data=metrics_list`** and **`output_file='metrics.csv'`**. "
        "5. **Final Confirmation**: Confirm successful logging of all three files to the global store."
    ),
    expected_output="A final confirmation message that all three output files (generated_tickets.csv, processing_log.csv, metrics.csv) have been successfully logged and are ready for final disk write.",
    agent=quality_critic_agent,
    context=[ticket_generation_task],
)

# --- The Crew ---
customer_support_crew = Crew(
    agents=[csv_reader_agent, feedback_classifier_agent, bug_analysis_agent, feature_extractor_agent, ticket_creator_agent, quality_critic_agent],
    tasks=[ingestion_task, classification_task, bug_analysis_task, feature_extraction_task, ticket_generation_task, qa_and_logging_task],
    process=Process.sequential, 
    verbose=True, 
)

# --- Streamlit UI Components ---

st.set_page_config(page_title="CrewAI Support Ticket System", layout="wide")
st.title("🚀 Automated Support Ticket System (CrewAI)")
st.caption("Upload CSVs and kick off the 6-Agent data processing pipeline.")

# File Upload Section
st.header("1️⃣ Upload Source Files")
col1, col2 = st.columns(2)

with col1:
    uploaded_app_reviews = st.file_uploader(
        "Upload **App Store Reviews** CSV (`app_store_reviews.csv` format)",
        type="csv",
        key="app_reviews_content"
    )

with col2:
    uploaded_support_emails = st.file_uploader(
        "Upload **Support Emails** CSV (`support_emails.csv` format)",
        type="csv",
        key="support_emails_content"
    )

# Execution and Logging Section
st.header("2️⃣ Run CrewAI Process")
start_button = st.button("▶️ Start Ticket Generation Crew", disabled=not (uploaded_app_reviews and uploaded_support_emails))

st.header("3️⃣ Execution Logs")
log_placeholder = st.empty()
final_result_placeholder = st.empty()

# Download Section
st.header("4️⃣ Download Results")
download_section = st.empty()

# --- Execution Logic ---
if start_button:
    if llm is None:
        st.error("Cannot run the crew. LLM setup failed (missing API key).")
    else:
        # Clear previous run data
        READER_DATA_STORE.clear()
        WRITER_OUTPUT_STORE.clear()
        
        # Use st.spinner and the custom log capture for live logging
        with st.spinner('Crew is working...'):
            with StreamlitLogCapture(log_placeholder) as log_capture:
                # Kickoff the Crew
                crew_result = customer_support_crew.kickoff()
            
            # Display final result and logs
            final_result_placeholder.success("✅ Crew Execution Finished Successfully!")
            log_placeholder.code(log_capture.log_content, language="text", height=400) # Final consolidated log

            # Display the result (which is the QA Agent's final confirmation message)
            st.subheader("Final Crew Result")
            st.info(crew_result)

            # Generate Download Buttons
            with download_section.container():
                st.subheader("Generated Output Files")
                for filename, content in WRITER_OUTPUT_STORE.items():
                    st.download_button(
                        label=f"⬇️ Download {filename}",
                        data=content,
                        file_name=filename,
                        mime="text/csv",
                        key=f"download_{filename}"
                    )
                
                if not WRITER_OUTPUT_STORE:
                    st.warning("No output files were generated. Check the execution logs for errors.")