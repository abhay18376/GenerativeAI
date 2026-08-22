import os
import csv
import json
from io import StringIO
from typing import Any, List, Dict
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from dotenv import load_dotenv
import uuid
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# --- GLOBAL STATIC DATA STORES ---
# Used for shared state and content storage before final disk write
READER_DATA_STORE: List[Dict[str, Any]] = []
WRITER_OUTPUT_STORE: Dict[str, str] = {}
PROCESSING_LOG: List[Dict[str, Any]] = []

# --- LLM Setup ---
llm = LLM(
    model="gemini/gemini-2.5-flash", 
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.7
)

# --- Mock Content for Dual Ingestion ---
# This content simulates reading the physical CSV files.
# APP_STORE_CONTENT = """review_id,platform,rating,review_text,user_name,date,app_version
# r001,Google Play,1,"App crashes when I try to open settings. Reinstalled twice, still crashes.",Sanjay K,2025-10-22,3.0.1
# r002,App Store,2,"Can't login since update. Keeps saying invalid token.",Priya M,2025-09-30,3.0.1
# r003,Google Play,5,"Amazing app! Very intuitive and fast.",Alex R,2025-08-14,2.9.0
# r005,Google Play,3,"Please add dark mode — would be great for night use.",Vikram P,2025-06-20,2.8.7
# r006,App Store,1,"Data sync not working across devices. Lost my notes.",Maya T,2025-05-11,2.7.4
# r011,Google Play,1,"Can't login after reinstall. Keeps saying 'network error'.",Anita P,2025-01-25,3.0.1
# r022,App Store,1,"Spam: Buy followers now!!! Visit www.fakepromo.example",spam_user_123,2024-02-28,1.0.0
# r027,Google Play,1,"Random characters: asdklj123!!! ???",weird_bot,2023-09-18,1.8.4
# r020,App Store,3,"Missing functionality: no offline mode which I need frequently.",Zara A,2024-04-12,2.2.8
# r014,App Store,2,"Poor customer service — didn't get a response for 2 weeks.",Carlos M,2024-10-21,2.5.4
# r019,Google Play,2,"Too expensive for what it offers. Subscription is steep.",Manish S,2024-05-05,2.3.2
# r009,Google Play,5,"Love the new feature — exporting reports is so easy now!",Sara L,2025-03-10,2.6.5
# """

# SUPPORT_EMAIL_CONTENT = """email_id,subject,body,sender_email,timestamp,priority
# e001,"App Crash Report - opening settings","Hi Team, Every time I open Settings the app crashes to home screen. Device: Samsung Galaxy S21, OS: Android 13. Steps: 1) Open app 2) Tap Profile 3) Tap Settings -> crash.",sanjay.k@example.com,2025-10-23 09:12:45,High
# e002,"Login Issue - invalid token","Can't login since the latest update. It shows 'invalid token'. I tried clearing cache. Device: iPhone 12, iOS 17.0.2. Please advise.",priya.m@example.com,2025-09-30T14:05:00Z,High
# e003,"Feature Request: Dark Mode","Would love a dark mode option for night use. Many users have requested this on forums. Thanks.",vikram.p@example.com,20-06-2025 18:30,Medium
# e004,"Data Loss Problem - notes missing after sync","After syncing between phone and tablet some notes disappeared. Device: Pixel 6, Android 12. Repro steps: sync on phone -> open tablet -> missing entries.",maya.t@example.com,11/05/2025 07:45 AM,Critical
# e009,"Spam: Buy followers now!!!","Buy followers cheap at www.fakepromo.example — increase your stats now!!!",spam_user_123@example.com,2024-02-28,
# e010,"Login: 'network error' after reinstall","I can't login after reinstalling. Keeps saying 'network error' though my internet is fine. Device: iPhone SE (2020), iOS 15.",anita.p@example.com,2025-01-25 19:02,High
# e012,"Complaint: Poor customer service","I emailed support 2 weeks ago and haven't received a response about billing.",carlos.m@example.com,2024-10-21 10:40:00,Medium
# e014,"Too expensive - subscription feedback","The subscription price is too steep for casual users. Consider a basic tier.",manish.s@example.com,2024-05-05 06:30:00,Medium
# e015,"Random garbage text","asdklj123!!! ???",weird_bot@example.com,2023-09-18 00:00,
# e008,"Praise: Excellent performance", "After the update the app feels snappier — great work!",noah@example.com,2024-03-04 08:00:00,Low
# """

# --- Tool Implementations ---

class CSVReaderTool(BaseTool):
    name: str = "CSV Reader Tool"
    description: str = "Reads and loads support_emails.csv and app_store_reviews.csv, merges them, and stores the unified data in READER_DATA_STORE. Expects a single string of file paths (e.g., 'file1.csv, file2.csv')."

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
            # In a real app, you would log this error properly.
            print(f"Error reading {source_type} data: {e}") 
        return data

    def _run(self, file_paths: str) -> str:
        """Reads dual CSV content, populates the global READER_DATA_STORE, and returns a summary."""
        global READER_DATA_STORE
        

        with open("app_store_reviews.csv", mode='r', newline='', encoding='utf-8') as file:
            APP_STORE_CONTENT = file.read()
        app_reviews = self._read_csv(APP_STORE_CONTENT, 'review')

        with open("support_emails.csv", mode='r', newline='', encoding='utf-8') as file:
            SUPPORT_EMAIL_CONTENT = file.read()
        support_emails = self._read_csv(SUPPORT_EMAIL_CONTENT, 'email')
        # Read mock content (simulating file access)
       
        
        
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
            
            # Store content globally for later physical disk write
            WRITER_OUTPUT_STORE[output_file] = content_to_write
                
            return f"SUCCESS: Logged {len(data)} records to {output_file}. Content is ready for final disk write."
        
        except Exception as e:
            return f"ERROR: Failed to write data to {output_file}. Details: {e}"

# Initialize tools
csv_reader = CSVReaderTool()
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

# --- Kickoff the Crew ---
print("🚀 Starting the Automated Support Ticket System Crew (6-Agent Architecture)...")
# *** UNCOMMENTED: This line now executes the crew. ***
final_result = customer_support_crew.kickoff()

print("\n\n################################")
print("## Crew Execution Finished ##")
print("################################\n")
print(final_result)


# --- FINAL DISK WRITE BLOCK ---
print("\n--- Writing Final Files to Disk ---")
files_written = 0
for filename, content in WRITER_OUTPUT_STORE.items():
    try:
        # *** ACTUAL FILE WRITE LOGIC ***
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Successfully wrote {filename} to disk ({len(content.splitlines()) - 1} data rows).")
        files_written += 1
    except Exception as e:
        print(f"❌ ERROR writing {filename}: {e}")

if files_written == 3:
    print("\nALL REQUIRED FILES HAVE BEEN CREATED in the current directory.")
else:
    print("\nWARNING: Not all required files were written successfully.")