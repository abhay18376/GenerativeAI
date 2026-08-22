import gradio as gr
import os
import csv
import json
import sys
from io import StringIO
from typing import Any, List, Dict
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from dotenv import load_dotenv
import uuid
from datetime import datetime
from contextlib import redirect_stdout, contextmanager

# Load environment variables from .env file
load_dotenv()

# --- GLOBAL STATIC DATA STORES ---
# Used for shared state and content storage before final disk write
READER_DATA_STORE: List[Dict[str, Any]] = []
WRITER_OUTPUT_STORE: Dict[str, str] = {}
PROCESSING_LOG: List[Dict[str, Any]] = []

# --- LLM Setup ---
# Initialize LLM here, assuming GEMINI_API_KEY is in .env
try:
    llm = LLM(
        model="gemini/gemini-2.5-flash",
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=0.7
    )
except KeyError:
    print("FATAL: GEMINI_API_KEY not found in environment variables.")
    llm = None # Set to None to handle later gracefully

# --- Tool Implementations ---

class CSVReaderTool(BaseTool):
    name: str = "CSV Reader Tool"
    description: str = "Reads and loads support_emails.csv and app_store_reviews.csv, merges them, and stores the unified data in READER_DATA_STORE. Expects a single string of file paths (e.g., 'file1.csv, file2.csv')."
    
    # Store the actual file paths set by the Gradio function
    app_store_path: str = "" 
    support_email_path: str = ""

    def _read_csv(self, content: str, source_type: str) -> List[Dict[str, Any]]:
        """Helper function to read a single CSV string and add a 'source_type' tag."""
        data = []
        try:
            reader = csv.DictReader(StringIO(content.strip()))

            for row in reader:
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
        
        # Reset store for each run
        READER_DATA_STORE.clear()

        if not self.app_store_path or not self.support_email_path:
             return "ERROR: File paths for App Store Reviews and Support Emails were not set correctly."


        try:
            with open(self.app_store_path, mode='r', newline='', encoding='utf-8') as file:
                APP_STORE_CONTENT = file.read()
            app_reviews = self._read_csv(APP_STORE_CONTENT, 'review')
        except Exception as e:
            return f"ERROR: Failed to read App Store Reviews file at {self.app_store_path}. Details: {e}"

        try:
            with open(self.support_email_path, mode='r', newline='', encoding='utf-8') as file:
                SUPPORT_EMAIL_CONTENT = file.read()
            support_emails = self._read_csv(SUPPORT_EMAIL_CONTENT, 'email')
        except Exception as e:
             return f"ERROR: Failed to read Support Emails file at {self.support_email_path}. Details: {e}"

        
        READER_DATA_STORE = app_reviews + support_emails
        total_count = len(READER_DATA_STORE)
        
        if total_count == 0:
            return "ERROR: Successfully read, but no data rows were found."

        return (
            f"SUCCESS: Loaded {len(app_reviews)} reviews and {len(support_emails)} emails, "
            f"totaling {total_count} records. The unified data is ready for processing."
        )


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
            
            # Store content globally for later disk write
            WRITER_OUTPUT_STORE[output_file] = content_to_write
                
            return f"SUCCESS: Logged {len(data)} records to {output_file}. Content is ready for final disk write."
        
        except Exception as e:
            return f"ERROR: Failed to write data to {output_file}. Details: {e}"


# --- Context Manager for Logging (Standard Python approach for I/O redirection) ---
@contextmanager
def captured_output():
    """Context manager to capture stdout and store it in a StringIO object."""
    new_stdout = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = new_stdout
        yield new_stdout
    finally:
        sys.stdout = old_stdout


# --- Main Crew Execution Function ---

def run_crew(app_reviews_file, support_emails_file):
    """Executes the CrewAI process using uploaded files and captures logs."""
    
    global READER_DATA_STORE, WRITER_OUTPUT_STORE, PROCESSING_LOG, llm
    
    if llm is None:
        return "", "", "❌ FATAL ERROR: GEMINI_API_KEY not configured. Please set it in your .env file.", None, None, None
        
    # --- Clear global state for fresh run ---
    READER_DATA_STORE.clear()
    WRITER_OUTPUT_STORE.clear()
    PROCESSING_LOG.clear()
    
    # --- Handle file uploads and set paths ---
    if not app_reviews_file or not support_emails_file:
        return "", "", "❌ ERROR: Please upload both 'App Store Reviews' and 'Support Emails' CSV files.", None, None, None

    # Gradio uploads files to a temp path, we use those paths directly
    app_store_path = app_reviews_file.name
    support_email_path = support_emails_file.name
    
    # Initialize tools and inject the file paths
    csv_reader = CSVReaderTool(app_store_path=app_store_path, support_email_path=support_email_path)
    csv_writer = CSVWriterTool()
    
    # --- Agents (Six-Agent Architecture) ---
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

    # --- Kickoff and Capture Logs ---
    final_result = ""
    full_log = ""
    
    with captured_output() as log_capture:
        print("🚀 Starting the Automated Support Ticket System Crew (6-Agent Architecture)...")
        try:
            final_result = customer_support_crew.kickoff()
        except Exception as e:
            final_result = f"Crew Execution Failed: {e}"
        
        print("\n\n################################")
        print("## Crew Execution Finished ##")
        print("################################\n")
        print(final_result)
        
        # This section simulates the original script's final file write but uses Gradio's File object
        print("\n--- Preparing Final Files for Download ---")
        full_log = log_capture.getvalue()

    # --- Prepare files for Gradio output ---
    output_files = {}
    for filename, content in WRITER_OUTPUT_STORE.items():
        try:
            # Create a temporary file path for Gradio to serve
            temp_path = f"temp_output_{filename}"
            with open(temp_path, 'w', newline='', encoding='utf-8') as f:
                f.write(content)
            
            output_files[filename] = gr.File(value=temp_path, visible=True)
            print(f"✅ Download link generated for {filename}.")
        except Exception as e:
            print(f"❌ ERROR preparing {filename} for download: {e}")

    # Return values in the order of Gradio components
    return (
        final_result, 
        full_log, 
        "✅ **Success!** Ticket processing complete. Download the generated files below.", 
        output_files.get('generated_tickets.csv'), 
        output_files.get('processing_log.csv'), 
        output_files.get('metrics.csv')
    )


# --- Gradio Interface Definition ---

with gr.Blocks(title="CrewAI Automated Support Ticket System") as demo:
    gr.Markdown(
        """
        # 🤖 CrewAI Automated Support Ticket System
        This interface runs a 6-Agent CrewAI system to process and categorize customer feedback from two CSV files.
        
        **Instructions:**
        1.  Upload your `app_store_reviews.csv` file.
        2.  Upload your `support_emails.csv` file.
        3.  Click **Run CrewAI Process**.
        4.  View the live logs and download the generated tickets, logs, and metrics when complete.
        
        ***Note: Requires a `GEMINI_API_KEY` in a local `.env` file.***
        """
    )
    
    with gr.Row():
        app_reviews_file = gr.File(label="1. Upload App Store Reviews CSV", file_types=[".csv"])
        support_emails_file = gr.File(label="2. Upload Support Emails CSV", file_types=[".csv"])
    
    run_button = gr.Button("🚀 Run CrewAI Process", variant="primary")
    
    # --- Status and Final Results ---
    status_message = gr.Markdown("Ready to start...", elem_id="status_msg")
    
    with gr.Row():
        final_output_text = gr.Textbox(label="Final Crew Result", lines=5, interactive=False, container=True)
        
    with gr.Column():
        gr.Markdown("### Generated Output Files (Downloadable)")
        with gr.Row():
            tickets_file = gr.File(label="Generated Tickets", type="filepath", interactive=False)
            processing_log_file = gr.File(label="Processing Log", type="filepath", interactive=False)
            metrics_file = gr.File(label="Metrics Report", type="filepath", interactive=False)
    
    # --- Live Log ---
    gr.Markdown("---")
    gr.Markdown("### 📜 Crew Execution Logs (Verbose Output)")
    full_log_textbox = gr.Textbox(label="Execution Log", lines=20, interactive=False, autoscroll=True)

    # --- Button Action Mapping ---
    run_button.click(
        fn=run_crew,
        inputs=[app_reviews_file, support_emails_file],
        outputs=[final_output_text, full_log_textbox, status_message, tickets_file, processing_log_file, metrics_file]
    )

if __name__ == "__main__":
    demo.launch()