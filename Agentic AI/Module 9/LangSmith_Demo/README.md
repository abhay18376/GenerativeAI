AI Observability with Langsmith

Project Structure
- app.py: Main application code for the AI Resume Screening with LangSmith Observability .
- requirements.txt: List of dependencies required to run the application.
- .env: Environment file to store the Google API key (GOOGLE_API_KEY).
- venv/: Virtual environment directory (e.g., env for storing installed packages).

Setup Instructions:-
Create a Virtual Environment:
- Navigate to the project directory.
- Run: python -m venv venv (or env if preferred).
- Activate the virtual environment:
- Windows: venv\Scripts\activate
- macOS/Linux: source venv/bin/activate

Install Dependencies:
- Ensure requirements.txt is in the project directory.
- Run: pip install -r requirements.txt

Set Up the API Key:
- Create a .env file in the project directory.
- Add your Google API key: GOOGLE_API_KEY=your_api_key_here
- Add your LangSmith API key: LANGCHAIN_API_KEY=your_api_key_here
- Add your LangSmith Project Name: LANGCHAIN_PROJECT=resume-screening-demo
- Ensure the .env file is loaded using python-dotenv (already included in app.py).

Run the Application Locally:
- Run: streamlit run app.py
- Open the provided URL (e.g., http://localhost:8501) in your browser to access the app.

---

## 🛠️ Setup Instructions
````

1. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your API key**
   Create a `.env` file in the project root and add:

   ```
   GOOGLE_API_KEY=your_google_api_key
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

---

## 📄 File Structure

```plaintext
├── app.py                  # Main Streamlit app
├── chroma_store/           # Folder to store vector DB files
├── .env                    # Contains API key (not committed)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

