# Langfuse AI Observability Demo

This project demonstrates AI observability using Langfuse in a multi-agent research application built with LangGraph, Groq API, and DuckDuckGo search. It runs in Google Colab.

NOTE:  This demo has been worked on and executed in Google Colab.

## Setup

1. **Open in Google Colab**:
   - Upload `Langfuse_AI_Observability_Demo` file to Google Colab.

2. **Set Up API Keys in Colab Secrets**:
   - Go to the “Secrets” tab in Colab (lock icon).
   - Add the following secrets:
     - `GROQ_API_KEY`: Your Groq API key (from https://console.groq.com/keys).
     - `LANGFUSE_API_KEY`: Your Langfuse secret key (from https://cloud.langfuse.com).
     - `LANGFUSE_PUBLIC_KEY`: Your Langfuse public key.
     - `LANGFUSE_HOST`: Your Langfuse host (e.g., `https://cloud.langfuse.com`).

3. **Configure Langfuse Settings**:
   - Replace `"first-index"` with your Langfuse organization name.
   - Replace `"Langfuse_demo"` with your Langfuse project name.
   - Update `"demo-user-001"` with your user ID.

## Running the Demo

1. **Execute the Notebook**:
   - Run the cell in Colab.
   - Enter a research topic when prompted (e.g., "quantum computing").

2. **View Results**:
   - Check the output for search results, analysis, and summary.
   - Follow the instructions to view the trace in the Langfuse dashboard.

## Notes
- Ensure your API keys and Langfuse settings are correct to avoid errors.
- This demo was last tested on May 29, 2025, at 01:53 PM IST.