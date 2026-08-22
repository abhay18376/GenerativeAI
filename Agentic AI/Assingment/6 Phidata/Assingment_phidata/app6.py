import os
import json
import re
from datetime import datetime
from typing import List, Dict, Union
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from phi.tools import tool
from phi.agent import Agent
from phi.workflow import Workflow
from phi.model.openai import OpenAIChat

# =========================
# 1) Setup
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

st.set_page_config(page_title="Support Triage Agent", page_icon="🤖", layout="wide")
st.title("🤖 Support Triage Agent (Agent + Workflow)")

# LLM for the Agent (used for tool routing + formatting)
model = OpenAIChat(id="gpt-4o-mini")

# Pinecone (v7)
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "support-complaints"
DIM = 384  # all-MiniLM-L6-v2 embedding dimension

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)

# Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# 2) CSV schema mappings
# =========================
CSV_TEXT_COLS = ["Ticket Subject", "Ticket Description"]  # concatenated
CSV_DATE_COL = "Date of Purchase"
CSV_TYPE_COL = "Ticket Type"
CSV_PRIORITY_COL = "Ticket Priority"
CSV_CHANNEL_COL = "Ticket Channel"
CSV_RATING_COL = "Customer Satisfaction Rating"
CSV_ID_COL = "Ticket ID"

MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12"
}

def to_sentiment_label(rating):
    """1–2 => negative, 3 => neutral, 4–5 => positive"""
    try:
        r = float(rating)
    except Exception:
        return None
    if r <= 2:
        return "negative"
    if r == 3:
        return "neutral"
    if r >= 4:
        return "positive"
    return None

def safe_iso_date(val):
    if pd.isna(val):
        return None
    try:
        return pd.to_datetime(val).strftime("%Y-%m-%d")
    except Exception:
        try:
            return pd.to_datetime(str(val), errors="coerce").strftime("%Y-%m-%d")
        except Exception:
            return None

def matches_date_filter(complaint_date: str, date_filter: str) -> bool:
    if not complaint_date or not date_filter:
        return True
    try:
        cd = str(complaint_date).lower()
        df = str(date_filter).lower().strip()
        if df in cd:
            return True
        m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", df)
        y = re.search(r"20\d{2}", df)
        if m and y:
            month_num = MONTH_MAP[m.group(1)]
            year = y.group(0)
            return f"{year}-{month_num}" in cd
        return False
    except Exception:
        return True

# =========================
# 3) Tools
# =========================
@tool
def process_and_store_complaints(filepath: str) -> str:
    """
    Read CSV/Excel, build text = Ticket Subject + Ticket Description,
    derive sentiment from rating, and upsert rich metadata to Pinecone.
    """
    df = pd.read_csv(filepath) if filepath.endswith(".csv") else pd.read_excel(filepath)

    # normalization helper to find proper-cased headers
    norm = {c.lower().strip(): c for c in df.columns}
    def col(name):  # returns exact column name if present
        return norm.get(name.lower().strip())

    text_cols = [col(c) for c in CSV_TEXT_COLS if col(c) in df.columns]
    if not text_cols:
        text_cols = [c for c in df.columns if any(k in c.lower() for k in ["subject","description","message","issue","problem"])]

    vectors = []
    for i, row in df.iterrows():
        parts = []
        for c in text_cols:
            val = row.get(c)
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
        complaint_text = " | ".join(parts).strip()
        if not complaint_text:
            continue

        md = {
            "text": complaint_text,
            "source_file": os.path.basename(filepath),
        }

        if col(CSV_ID_COL) in df.columns:
            md["ticket_id"] = str(row.get(col(CSV_ID_COL)))

        if col(CSV_DATE_COL) in df.columns:
            iso = safe_iso_date(row.get(col(CSV_DATE_COL)))
            if iso:
                md["date"] = iso

        if col(CSV_TYPE_COL) in df.columns:
            v = row.get(col(CSV_TYPE_COL))
            if pd.notna(v): md["category"] = str(v)

        if col(CSV_PRIORITY_COL) in df.columns:
            v = row.get(col(CSV_PRIORITY_COL))
            if pd.notna(v): md["priority"] = str(v)

        if col(CSV_CHANNEL_COL) in df.columns:
            v = row.get(col(CSV_CHANNEL_COL))
            if pd.notna(v): md["channel"] = str(v)

        if col(CSV_RATING_COL) in df.columns:
            r = row.get(col(CSV_RATING_COL))
            lbl = to_sentiment_label(r)
            if lbl: md["sentiment"] = lbl
            if pd.notna(r):
                try: md["rating"] = float(r)
                except: pass

        emb = embedding_model.encode(complaint_text).tolist()
        vectors.append({
            "id": f"{os.path.basename(filepath)}::row_{i}",
            "values": emb,
            "metadata": md
        })

    if vectors:
        index.upsert(vectors=vectors)

    return f"Stored {len(vectors)} complaints from {os.path.basename(filepath)}."

@tool
def search_complaints(
    query: str,
    sentiment: str = None,
    date_filter: str = None,
    category: str = None,
    top_k: int = 10,
) -> str:
    q_emb = embedding_model.encode(query).tolist()

    flt = {}
    if sentiment:
        flt["sentiment"] = {"$eq": sentiment.lower()}
    if category:
        flt["category"] = {"$eq": category}

    res = index.query(
        vector=q_emb,
        top_k=top_k,
        include_metadata=True,
        filter=flt if flt else None,
    )

    rows = []
    for m in res.get("matches", []):
        md = m.get("metadata", {}) or {}
        if date_filter and not matches_date_filter(md.get("date"), date_filter):
            continue
        txt = (md.get("text") or "").strip()
        short = (txt[:200] + "...") if len(txt) > 200 else txt

        # return ONLY these two fields, with safe defaults
        rows.append({
            "ticket_id": str(md.get("ticket_id") or "unknown"),
            "text": short,
        })

    return json.dumps(rows, ensure_ascii=False)


@tool
def analyze_complaint_patterns(filepath: str) -> str:
    """Basic summary + distributions (sentiment from rating)."""
    df = pd.read_csv(filepath) if filepath.endswith(".csv") else pd.read_excel(filepath)
    out = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "missing_counts": df.isna().sum().to_dict(),
    }
    if CSV_RATING_COL in df.columns:
        labels = df[CSV_RATING_COL].apply(to_sentiment_label)
        out["sentiment_distribution"] = labels.value_counts(dropna=True).to_dict()
    if CSV_TYPE_COL in df.columns:
        out["category_distribution"] = df[CSV_TYPE_COL].value_counts().to_dict()
    if CSV_PRIORITY_COL in df.columns:
        out["priority_distribution"] = df[CSV_PRIORITY_COL].value_counts().to_dict()
    return json.dumps(out, indent=2)


    # Helper function for cleaning special1 string values
    def clean_string_value(value: str) -> Optional[str]:
        """Converts 'Unknown' and 'Not available' to None."""
        cleaned = value.strip()
        if cleaned.lower() in ("unknown", "not available", ""):
            return None
        return cleaned

    for match in matches:
        raw_id = match.group(1).strip()
        raw_text = match.group(2).strip()
        raw_rating = match.group(4).strip()
        
        # Clean up multiline text: replace excessive newlines and whitespace with a single space
        clean_text = re.sub(r'\s+', ' ', raw_text).strip()

        # Convert Rating to float or None
        rating: Optional[float]
        try:
            rating = float(raw_rating)
        except ValueError:
            rating = None

        complaint = {
            "id": clean_string_value(raw_id),
            "text": clean_text,
            "sentiment": clean_string_value(match.group(3)),
            "rating": rating,
            "category": clean_string_value(match.group(5)),
            "priority": clean_string_value(match.group(6)),
            "channel": clean_string_value(match.group(7)),
            "date": clean_string_value(match.group(8)),
        }
        results.append(complaint)

    return results
# =========================
# 4) Agent + Workflow
# =========================
agent = Agent(
    tools=[process_and_store_complaints, search_complaints, analyze_complaint_patterns],
    model=model,
    name="Support Triage Agent",
    description="Processes support complaints, indexes them in Pinecone, enables semantic search with filters, and analyzes patterns.",
)


workflow = Workflow(
    agents=[agent],
    name="support_triage_workflow"
)

import re
import json # <-- Import the json module
from typing import List, Dict, Union

def extract_complaints_v3(text_content: str) -> str: # <-- Change the return type to 'str'
    """
    Extracts complaint information (Ticket ID and Text) from two different,
    complex-formatted string variations and returns the result as a JSON string.
    """
    
    # 1. Clean the input text by removing surrounding metadata like 'content=' and tool call info
    text_content = re.sub(r'content\=\'|\' content_type\=.*', '', text_content, flags=re.DOTALL).strip()
    
    # --- Check for Single-Ticket Format (String 1) ---
    single_ticket_pattern = re.compile(
        r'- \*\*Ticket ID\*\*:\s*([^\n]+)\n' 
        r'- \*\*Complaint Text\*\*:\s*"([^"]+)"', 
        re.DOTALL
    )

    single_match = single_ticket_pattern.search(text_content)
    
    if single_match:
        # Success: Create the single extracted ticket list
        ticket_id = single_match.group(1).strip()
        complaint_text = single_match.group(2).strip()
        extracted_data = [{
            "ticket_id": ticket_id,
            "complaint_text": complaint_text
        }]
        
    else:
        # --- Extract from Multi-Ticket List Format (String 2) ---
        pattern_multi = re.compile(
            r'(?:^|\n\n)\s*\d+\.\s*' 
            r'\*\*Ticket ID:\s*([^\*]+?)\*\*' 
            r'\s*-\s*\*\*Text:\*\s*' 
            r'(.*?)(?=\n\n\d+\.|\s*If you need|$)', 
            re.DOTALL | re.IGNORECASE
        )
        
        extracted_data = [] # Changed 'all_tickets' to 'extracted_data' for consistency
        
        # Perform a global search across the content
        matches = pattern_multi.finditer(text_content)
        
        for match in matches:
            ticket_id = match.group(1).strip().strip(':').strip()
            complaint_text = match.group(2).strip()
                  
            extracted_data.append({
                "ticket_id": ticket_id,
                "complaint_text": complaint_text
            })

    # --- FINAL STEP: Convert the list of dictionaries to a JSON string ---
    # We use indent=4 for a "pretty-printed" string output, which is generally better for display.
    return json.dumps(extracted_data, indent=4)


# =========================
# 5) Streamlit UI
# =========================
st.caption("CSV-aware pipeline: joins Ticket Subject + Ticket Description, "
           "derives sentiment from Customer Satisfaction Rating, and stores metadata (Type, Priority, Channel, Date).")

st.header("📁 Upload CSV")
uploaded = st.file_uploader(
    "Upload support tickets CSV",
    type=["csv"],
    help="Expected columns: Ticket Subject, Ticket Description, Ticket Type, Ticket Priority, Customer Satisfaction Rating, Date of Purchase, Ticket ID"
)

temp_path = None
if uploaded is not None:
    temp_path = "uploaded_data.csv"
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    try:
        df = pd.read_csv(temp_path)
        st.success("CSV loaded!")
        st.dataframe(df.describe(), use_container_width=True)

        # ---- Use Agent via Workflow to process & index ----
        with st.spinner("Indexing via Agent..."):
            # Natural-language tool invocation (function calling handled by phi)
            msg = agent.run(f"process_and_store_complaints(filepath='{temp_path}')")
        st.info(msg)

        #with st.expander("Dataset summary (via Agent)"):
            #summary = agent.run(f"analyze_complaint_patterns(filepath='{temp_path}')")
            #try:
                #st.json(json.loads(summary))
            #except Exception:
                #st.text(summary)

    except Exception as e:
        st.error(f"Failed to read/process CSV: {e}")

st.header("🔍 Semantic Search (via Agent)")
c1, c2 = st.columns([2, 1])
with c1:
    q = st.text_input("Query", placeholder="e.g., delivery delays, refund issues, payment problems")
with c2:
    top_k = st.slider("Max results", 1, 25, 5)

f1, f2, f3 = st.columns(3)
with f1:
    sent = st.selectbox("Sentiment", ["Any", "negative", "neutral", "positive"])
    sent = None if sent == "Any" else sent
with f2:
    cat = st.text_input("Category (Ticket Type)", placeholder="e.g., billing, shipping")
    cat = cat.strip() or None
with f3:
    dfilter = st.text_input("Date/Period", placeholder="e.g., 2024-05 or May 2024")
    dfilter = dfilter.strip() or None
cmd=""
if st.button("🔎 Search"):
    with st.spinner("Searching via Agent..."):
        # Build search command
        cmd = f"search_complaints(query={json.dumps(q)}, sentiment={json.dumps(sent)}, date_filter={json.dumps(dfilter)}, category={json.dumps(cat)}, top_k={top_k})"
        st.write(f"**Command:** {cmd}")

        results = agent.run(cmd)
        # 1. Safely extract the raw string content
        raw_text_content = ""

        # Try the most likely attribute names for the final output string
        if hasattr(results, 'content'):
            raw_text_content = results.content
        elif hasattr(results, 'text'):
            raw_text_content = results.text
        # If the agent framework uses a list of messages, the content might be in the last one
        elif hasattr(results, 'messages') and results.messages:
            # Check the content of the last message object
            raw_text_content = results.messages[-1].content
        else:
            # Fallback if no expected attribute is found
            st.error(f"Error: Agent result object type '{type(results)}' does not contain a recognizable 'content' or 'text' attribute.")


        # 2. Call your function ONLY if a string was successfully extracted
        if isinstance(raw_text_content, str):
            # Pass the actual string content to your function
            extracted_info_json_string = extract_complaints_v3(raw_text_content)
            
            st.write("**Extracted Ticket Information:**")
            # Use st.code for a clean, syntax-highlighted JSON display
            st.code(extracted_info_json_string, language='json')
        else:
            # Handle the case where raw_text_content couldn't be found
            st.error("Extraction failed: Could not isolate the complaint text string from the agent's response.")

