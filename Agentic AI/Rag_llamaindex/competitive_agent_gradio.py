

import os
import csv
import sys
import pandas as pd
from dotenv import load_dotenv
from typing import List, Any, Optional

# Load environment variables
load_dotenv()

# ---- Optional Gradio import ----
GRADIO_AVAILABLE = True
try:
    import gradio as gr
except Exception:
    GRADIO_AVAILABLE = False

# ---- Robust import of llama_index and cohere dependencies ----
LLAMA_AVAILABLE = True
COHERE_AVAILABLE = True
try:
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
    from llama_index.core import Settings
    from llama_index.embeddings.cohere import CohereEmbedding
    from llama_index.llms.cohere import Cohere
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.tools import QueryEngineTool, ToolMetadata
    from llama_index.core.agent import ReActAgent
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core import Document
except Exception:
    # LlamaIndex (or some parts of it) not available. Provide fallbacks.
    LLAMA_AVAILABLE = False

    class Document:
        def __init__(self, text: str):
            self.text = text

    class VectorStoreIndex:
        """Minimal fallback index: stores documents and performs keyword-overlap retrieval."""
        def __init__(self, documents: List[Document]):
            self.docs = documents

        @classmethod
        def from_documents(cls, documents: List[Document], embed_model=None):
            return cls(documents)

        def as_query_engine(self, similarity_top_k: int = 3, llm=None):
            index = self

            class QueryEngine:
                def __init__(self, index, top_k):
                    self.index = index
                    self.top_k = top_k

                def query(self, query_text: str):
                    q_words = set([w.strip().lower() for w in query_text.split() if w.strip()])
                    scored = []
                    for doc in self.index.docs:
                        text = doc.text if hasattr(doc, 'text') else str(doc)
                        doc_words = set([w.strip().lower() for w in text.split() if w.strip()])
                        score = len(q_words & doc_words)
                        scored.append((score, text))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    top = [text for score, text in scored[: self.top_k] if score > 0]
                    return "\n\n---\n\n".join(top) if top else "(no relevant documents found)"

            return QueryEngine(index=index, top_k=similarity_top_k)

    class QueryEngineTool:
        @classmethod
        def from_defaults(cls, query_engine=None, name=None, description=None):
            def tool_fn(q: str):
                return query_engine.query(q)

            tool_fn.name = name or "query_tool"
            tool_fn.description = description or "A simple query tool"
            return tool_fn

    class ChatMemoryBuffer:
        @classmethod
        def from_defaults(cls, token_limit=2000):
            return {"token_limit": token_limit, "history": []}

    class ReActAgent:
        def __init__(self, tools: List[Any], llm: Any = None, memory: Any = None, verbose: bool = False):
            self.tools = tools
            self.llm = llm
            self.memory = memory
            self.verbose = verbose

        def chat(self, question: str):
            results = []
            for t in self.tools:
                try:
                    res = t(question)
                except Exception:
                    try:
                        res = t.query(question)
                    except Exception as e:
                        res = f"(tool failed: {e})"
                results.append(str(res))

            combined = "\n\n--- TOOL OUTPUT ---\n\n".join(results)
            analysis = f"Agent analysis (heuristic): found {len(results)} tool outputs.\n\n{combined}"

            if isinstance(self.memory, dict):
                self.memory.get("history", []).append({"q": question, "a": analysis})
            return analysis

    # Cohere fallbacks (very light)
    COHERE_AVAILABLE = False

    class CohereEmbedding:
        def __init__(self, model_name: str = None, cohere_api_key: str = None):
            self.model_name = model_name
            self.key = cohere_api_key

    class Cohere:
        def __init__(self, model: str = None, api_key: str = None):
            self.model = model
            self.key = api_key

    class SentenceSplitter:
        def __init__(self, chunk_size: int = 512, chunk_overlap: int = 20):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

# ---- End robust import section ----


class CompetitiveAnalysisAgent:
    def __init__(self, log_append):
        self.log_append = log_append
        self.log_append("Initializing CompetitiveAnalysisAgent object...")

        try:
            self.embed_model = CohereEmbedding(
                model_name="embed-english-v3.0",
                cohere_api_key=os.getenv("COHERE_API_KEY")
            )

            self.llm = Cohere(
                model="command-a-03-2025",
                api_key=os.getenv("COHERE_API_KEY")
            )

            try:
                # Settings may not exist if LLAMA isn't available; ignore failures
                Settings.embed_model = self.embed_model  # type: ignore
                Settings.llm = self.llm  # type: ignore
                Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)  # type: ignore
            except Exception:
                pass

            self.index = None
            self.query_engine = None
            self.agent = None
            self.workflow = None

            self.log_append("Models and settings initialized (real or fallback).")
        except Exception as e:
            self.log_append(f"Error initializing models: {e}")

    def load_and_preprocess_data(self, csv_file_path: str) -> List[str]:
        try:
            if not os.path.exists(csv_file_path):
                self._create_sample_csv(csv_file_path)
                self.log_append(f"Sample CSV created at {csv_file_path} for testing.")

            df = pd.read_csv(csv_file_path)

            documents: List[str] = []
            for _, row in df.iterrows():
                document_content = f"""
Competitor: {row.get('Competitor Name', '')}
Product: {row.get('Product Description', '')}
Marketing Strategy: {row.get('Marketing Strategy', '')}
Financial Summary: {row.get('Financial Summary', '')}
"""
                documents.append(document_content.strip())

            self.log_append(f"Loaded {len(documents)} competitor records from {csv_file_path}")
            return documents

        except Exception as e:
            self.log_append(f"Error loading data: {e}")
            return []

    def _create_sample_csv(self, path: str):
        rows = [
            {
                "Competitor Name": "RetailX",
                "Product Description": "Retail analytics platform for supermarkets",
                "Marketing Strategy": "Content marketing, trade shows",
                "Financial Summary": "Revenue growth 30% YoY"
            },
            {
                "Competitor Name": "ShopSense",
                "Product Description": "AI-powered checkout optimization",
                "Marketing Strategy": "Partnerships with POS vendors",
                "Financial Summary": "Seed funded, expanding"
            },
            {
                "Competitor Name": "DataMart",
                "Product Description": "B2B data marketplace for retail insights",
                "Marketing Strategy": "Developer community + API-first",
                "Financial Summary": "Profitable last year"
            }
        ]
        keys = rows[0].keys()
        with open(path, "w", newline="", encoding="utf-8") as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(rows)

    def create_index(self, documents: List[str]):
        try:
            llama_documents = [Document(text=d) if LLAMA_AVAILABLE else Document(d) for d in documents]
            self.index = VectorStoreIndex.from_documents(llama_documents, embed_model=self.embed_model)
            self.log_append("Vector store index created successfully (real or fallback)")
            return self.index

        except Exception as e:
            self.log_append(f"Error creating index: {e}")
            return None

    def setup_query_engine(self):
        if self.index is None:
            self.log_append("Index not created. Please create index first.")
            return None

        try:
            self.query_engine = self.index.as_query_engine(similarity_top_k=3, llm=self.llm)
            self.log_append("Query engine setup completed")
            return self.query_engine

        except Exception as e:
            self.log_append(f"Error setting up query engine: {e}")
            return None

    def create_competitive_analysis_agent(self):
        if self.query_engine is None:
            self.log_append("Query engine not setup. Please setup query engine first.")
            return None

        try:
            if LLAMA_AVAILABLE:
                sales_tool = QueryEngineTool.from_defaults(
                    query_engine=self.query_engine,
                    name="competitor_data",
                    description="Provides access to competitor information including products, marketing strategies, and financial data"
                )
            else:
                sales_tool = QueryEngineTool.from_defaults(query_engine=self.query_engine, name="competitor_data")

            memory = ChatMemoryBuffer.from_defaults(token_limit=2000) if LLAMA_AVAILABLE else ChatMemoryBuffer.from_defaults(token_limit=2000)

            self.agent = ReActAgent(tools=[sales_tool], llm=self.llm, memory=memory, verbose=True)

            self.log_append("Competitive analysis agent created successfully (real or fallback)")
            return self.agent

        except Exception as e:
            self.log_append(f"Error creating agent: {e}")
            return None

    def query_competitor_data(self, query: str):
        if self.query_engine is None:
            msg = "Query engine not initialized. Please setup the system first."
            self.log_append(msg)
            return msg

        try:
            response = self.query_engine.query(query)
            self.log_append("Direct query executed.")
            return response

        except Exception as e:
            self.log_append(f"Error querying data: {e}")
            return f"Error querying data: {e}"

    def ask_agent(self, question: str):
        if self.agent is None:
            msg = "Agent not initialized. Please create the agent first."
            self.log_append(msg)
            return msg

        try:
            response = self.agent.chat(question)
            self.log_append("Agent produced a response.")
            return response
        except Exception as e:
            self.log_append(f"Error asking agent: {e}")
            return f"Error asking agent: {e}"


# UI glue: replace prints by appending to a UI log and updating a Textbox
LOG_LINES: List[str] = []


def log_append(msg: str):
    """Append a message to the global log and keep it small."""
    LOG_LINES.append(str(msg))
    # keep last 500 lines to avoid runaway memory
    if len(LOG_LINES) > 500:
        del LOG_LINES[: len(LOG_LINES) - 500]


def get_log_text():
    return "\n".join(LOG_LINES)


# Global agent holder
CA_AGENT: CompetitiveAnalysisAgent | None = None


def initialize_system(csv_path: str = "competitors.csv"):
    global CA_AGENT
    log_append("Starting system initialization...")
    CA_AGENT = CompetitiveAnalysisAgent(log_append=log_append)

    documents = CA_AGENT.load_and_preprocess_data(csv_path)
    if not documents:
        log_append("No documents loaded. Initialization stopped.")
        return get_log_text()

    CA_AGENT.create_index(documents)
    CA_AGENT.setup_query_engine()
    CA_AGENT.create_competitive_analysis_agent()
    log_append("System initialization complete. Ready to accept queries.")
    return get_log_text()


# Handler for submit button
def on_submit(user_input: str):
    global CA_AGENT
    log_append(f"User input: {user_input}")

    # Lazy-initialize if not already initialized
    if CA_AGENT is None:
        log_append("Agent not found. Initializing now (will attempt to read 'competitors.csv').")
        initialize_system()

    # First perform a direct query
    direct = CA_AGENT.query_competitor_data(user_input)
    log_append("--- Direct Query Result Start ---")
    log_append(str(direct))
    log_append("--- Direct Query Result End ---")

    # Then ask the agent for a reasoned answer
    agent_resp = CA_AGENT.ask_agent(user_input)
    log_append("--- Agent Analysis Start ---")
    log_append(str(agent_resp))
    log_append("--- Agent Analysis End ---")

    return get_log_text()


# Build Gradio UI
with gr.Blocks(title="Competitive Analysis Agent") as demo:
    gr.Markdown("# Competitive Analysis Agent")
    with gr.Row():
        inp = gr.Textbox(lines=2, placeholder="Enter your question about competitors...", label="Query")
        submit = gr.Button("Submit")

    log_box = gr.Textbox(lines=25, interactive=False, label="System Log / Output")

    # Initialize button (optional) - uncomment if you want manual init
    init_btn = gr.Button("Initialize System")

    submit.click(fn=on_submit, inputs=[inp], outputs=[log_box])
    init_btn.click(fn=lambda: initialize_system(), inputs=None, outputs=[log_box])


if __name__ == "__main__":
    # Start Gradio app
    # Note: Ensure competitors.csv exists in the working directory and COHERE_API_KEY is set in your .env
    demo.launch()
