# Stock Analysis Agent with CrewAI

## Overview
This project implements a multi-agent system using [CrewAI](https://crewai.com/) to perform stock analysis for Apple Inc. (AAPL). The system consists of two agents:
- **Stock Analyst**: Fetches recent news via DuckDuckGo and analyzes stock price trends using Yahoo Finance.
- **Report Writer**: Generates a concise, investor-friendly report summarizing the findings.\

**NOTE**: This demonstrationh as been performed on Google Colab Notebook.

The project integrates external APIs (Yahoo Finance, DuckDuckGo) and the Google Gemini LLM to deliver actionable insights. It serves as a hands-on demonstration for the "Multi Agent Systems with LangGraph and CrewAI" module.

## Features
- Fetches real-time stock news and 1-month price history.
- Analyzes price trends and technical highlights.
- Produces a professional investor report.
- Modular design for scalability and easy extension to other stocks.

## Prerequisites
- Python 3.8+
- Google Gemini API key (obtain from Google Cloud or Gemini platform)
- Internet connection for API calls

## Project Structure
```
stock_analysis_project/
├── .env                    # Environment variables (API keys)
├── stock_analysis.ipynb    # Jupyter Notebook with the implementation
├── README.md               # This file
└── requirements.txt        # Project dependencies
```

## Setup Instructions

**Install Dependencies**:
   Create a virtual environment and install the required packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   **requirements.txt**:
   ```
   crewai
   google-generativeai
   duckduckgo_search
   yfinance
   pandas
   beautifulsoup4
   python-dotenv
   crewai-tools
   ```

**Set Up API Key**:
   Create a `.env` file in the project root with your Gemini API key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   Alternatively, the Jupyter Notebook prompts for the key during runtime using `getpass`.

**Run the Jupyter Notebook**:
   Launch Jupyter Notebook and open `stock_analysis.ipynb`:
   ```bash
   jupyter notebook
   ```

## Usage
1. Open `stock_analysis.ipynb` in Jupyter Notebook.
2. Execute the cells sequentially to:
   - Install dependencies.
   - Set up the Gemini LLM.
   - Define custom tools and agents.
   - Run the CrewAI workflow.
3. The final output is an investor report summarizing AAPL's recent performance and outlook.

**Sample Output**:
```
**Apple Inc. (AAPL) – Investor Report**
**Market Summary:**
Apple Inc. (AAPL) has demonstrated an overall upward trend over the past month, with some inherent market volatility...
**Key Trends and Technical Highlights:**
* Overall Trend: Generally upward, although with some volatility...
**Investment Outlook:**
Apple's stock shows potential for continued growth...
```

## Extending the Project
- **Add More Stocks**: Modify the `search_task` and `analysis_task` to accept different ticker symbols.
- **Enhance Tools**: Include additional data sources (e.g., SEC filings, Twitter sentiment).
- **Improve Report**: Add visualizations (e.g., price charts) using Matplotlib or Plotly.

## Troubleshooting
- **API Key Errors**: Ensure the Gemini API key is valid and correctly set in `.env` or entered during runtime.
- **Network Issues**: Verify internet connectivity for Yahoo Finance and DuckDuckGo APIs.
- **Dependency Conflicts**: Use a clean virtual environment to avoid package conflicts.

## Acknowledgments
- [CrewAI](https://crewai.com/) for the multi-agent framework.
- [Yahoo Finance](https://finance.yahoo.com/) and [DuckDuckGo](https://duckduckgo.com/) for data sources.
- Google Gemini for LLM support.