# 🚀 Automated Support Ticket System: CrewAI & Streamlit

This project demonstrates a powerful 6-Agent **CrewAI** architecture integrated with a **Streamlit** user interface. It automates the end-to-end process of ingesting raw customer feedback (from app store reviews and support emails), classifying it, analyzing key data points, and generating structured tickets, logs, and metrics.

## ✨ Features

* **🌐 Streamlit UI:** Easy-to-use web interface for uploading source CSV files and initiating the CrewAI process.
* **🧠 6-Agent Architecture:** A robust multi-step pipeline for data ingestion, classification, analysis, and quality assurance.
* **📄 Real-time Logging:** Displays all agent thoughts, tool use, and execution steps directly in the Streamlit UI.
* **📥 Structured Output:** Generates three distinct CSV files for download:
    1.  `generated_tickets.csv` (Final structured tickets)
    2.  `processing_log.csv` (Detailed log of every original record)
    3.  `metrics.csv` (Key performance indicators of the run)
* **🛠️ Custom Tools:** Uses `CSVReaderTool` for data ingestion and `CSVWriterTool` for final output logging.

## ⚙️ Architecture Overview

The system runs a **Sequential Process** involving six specialized agents:

| Agent Role | Goal | Output |
| :--- | :--- | :--- |
| **CSV Reader Agent** | Ingest and unify data from `reviews.csv` and `emails.csv`. | Unified list of dictionaries. |
| **Feedback Classifier Agent** | Categorize records into: Bug, Feature Request, Praise, Complaint, or Spam. | Updated list with a `category` key. |
| **Bug Analysis Agent** | Extract technical details (`device`, `os`, `priority`) for all Bugs/Complaints. | Updated list with technical fields. |
| **Feature Extractor Agent** | Assign `feature_impact_score` and extract keywords for Feature Requests. | Updated list with product fields. |
| **Ticket Creator Agent** | Filter out Spam/Praise and map remaining data to the final ticket schema. | Final list of structured ticket dictionaries. |
| **Quality Critic Agent** | QA the final tickets and log all three required output files using the `CSVWriterTool`. | Final confirmation message. |

## 🚀 Getting Started

Follow these steps to set up and run the application.

### Prerequisites

* Python 3.9+
* A **Gemini API Key** (required for the LLM).

### 1. Setup Environment

Clone the repository and install the necessary dependencies:

```bash
# Assuming you have the project files locally

pip install crewai streamlit python-dotenv pandas

to run app -  streamlit run streamlit_app.py