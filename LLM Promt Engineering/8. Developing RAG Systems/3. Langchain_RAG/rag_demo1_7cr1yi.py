import os
import argparse
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
###
from langchain.chains import RetrievalQA
import nltk
nltk.download('punkt', quiet=True)
from dotenv import load_dotenv
load_dotenv()
### to run in terminal:
## $ python rag_demo1.py --demo rag_pipeline
# .env should have GOOGLE_API_KEY set
# --------------------------------------------------
# RAG Pipeline with FAISS Persistence
# --------------------------------------------------
class RAGPipeline:
    def __init__(self, docs=None, persist_path="faiss_index_demo1"):
        self.docs = docs
        self.persist_path = persist_path
        # ✅ Using HuggingFace SentenceTransformer embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore = None
        self.retriever = None
        self.llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)

    def build(self, force_rebuild=False):
        # Load FAISS if exists
        if os.path.exists(self.persist_path) and not force_rebuild:
            print(f"✅ Loading existing FAISS index from {self.persist_path}")
            self.vectorstore = FAISS.load_local(
                self.persist_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("⚙️ Building new FAISS index...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", " ", ""],
                length_function=len,
                keep_separator=False
            )
            split_docs = text_splitter.split_documents(self.docs)
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            self.vectorstore.save_local(self.persist_path)
            print(f"💾 Saved FAISS index at {self.persist_path}")

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def ask(self, query):
        qa = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever)
        result = qa.invoke({"query": query})
        return result["result"] if isinstance(result, dict) else result

# --------------------------------------------------
# Helpers for multi-format document loading
# --------------------------------------------------
def load_documents(file_paths):
    docs = []
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            loader = TextLoader(path)
        elif ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(path)
        else:
            print(f"⚠️ Skipping unsupported file: {path}")
            continue
        docs.extend(loader.load())
    return docs

# --------------------------------------------------
# Demo 1: Basic RAG Pipeline
# --------------------------------------------------
def demo_rag_pipeline():
    file_paths = [
        "sample_docs/DL.txt",
        "sample_docs/IDirect_RelianceInd_CoUpdate_Dec22.pdf",
        "sample_docs/ML.docx"
    ]
    # loads the data, returns list of Document objects
    docs = load_documents(file_paths) 
    print(f"Loaded {len(docs)} documents.")
    # splits text in to chunks, and creates embeddings, stores in FAISS with persistence
    rag = RAGPipeline(docs, persist_path="faiss_index_demo1")
    rag.build()
    print("--- DEMO 1: Basic RAG (TXT, PDF, DOCX) ---")
    print(rag.ask("What is this collection of documents about?"))
    print(rag.ask("growth rate of Reliance Retail?"))
    print(rag.ask("What is the ARPU for reliance jio"))
    print(rag.ask("who won the match between India and Australia in 2023?"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline Demo")
    parser.add_argument(
        "--demo",
        type=str,
        default="rag_pipeline",
        choices=["rag_pipeline"],
        help="Choose the demo to run"
    )
    args = parser.parse_args()

    if args.demo == "rag_pipeline":
        demo_rag_pipeline()

