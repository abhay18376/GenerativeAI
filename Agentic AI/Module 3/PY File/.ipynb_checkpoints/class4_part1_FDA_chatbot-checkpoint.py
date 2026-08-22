"""
🏥 FDA MALARIA GUIDANCE Q&A CHATBOT - FIXED VERSION
===================================================
Compatible LangChain implementation with correct imports and error handling.

Installation (run this first):
pip install --upgrade langchain langchain-openai langchain-community python-dotenv faiss-cpu tiktoken colorama pydantic

Create .env file with:
OPENAI_API_KEY=sk-your-actual-key-here
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# Third-party imports with error handling
try:
    from dotenv import load_dotenv
    from colorama import init, Fore, Style
    init()
    load_dotenv()
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Run: pip install python-dotenv colorama")
    exit(1)

# LangChain imports with proper error handling and fallbacks
try:
    from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.chains import LLMChain, RetrievalQA
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
    print("✅ LangChain imports successful")
except ImportError as e:
    print(f"❌ Missing LangChain component: {e}")
    print("\nTo fix this, run:")
    print("pip install --upgrade langchain langchain-openai langchain-community python-dotenv faiss-cpu tiktoken colorama pydantic")
    exit(1)


# ===== CONFIGURATION =====
@dataclass
class ChatbotConfig:
    """Configuration for the malaria guidance chatbot"""
    openai_api_key: str
    model_name: str = "gpt-3.5-turbo"
    instruct_model: str = "gpt-3.5-turbo-instruct"
    temperature: float = 0.3
    max_tokens: int = 500
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5
    
    @classmethod
    def from_env(cls) -> "ChatbotConfig":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return cls(openai_api_key=api_key)


# ===== DOCUMENT PROCESSOR =====
class DocumentProcessor:
    """Process and structure the FDA malaria guidance document"""
    
    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.documents = []
        self.metadata = {}
    
    def extract_text_from_pdf_content(self, pdf_content: str) -> List[Document]:
        """Extract and structure text from PDF content"""
        
        # Clean and structure the text
        sections = self._parse_document_sections(pdf_content)
        
        # Create Document objects with metadata
        documents = []
        for section_title, content in sections.items():
            if len(content.strip()) == 0:
                continue
                
            # Further split long sections
            if len(content) > self.config.chunk_size:
                subsections = self._split_long_section(content, section_title)
                documents.extend(subsections)
            else:
                doc = Document(
                    page_content=content,
                    metadata={
                        "section": section_title,
                        "document_type": "FDA_guidance",
                        "topic": "malaria_drug_development",
                        "length": len(content)
                    }
                )
                documents.append(doc)
        
        self.documents = documents
        return documents
    
    def _parse_document_sections(self, text: str) -> Dict[str, str]:
        """Parse document into logical sections"""
        
        sections = {}
        
        # Split by major Roman numeral sections
        section_splits = re.split(r'\n\s*([IVX]+\.\s+[A-Z][A-Z\s,]+)\n', text)
        
        if len(section_splits) > 1:
            current_section = "Introduction"
            for i, part in enumerate(section_splits):
                if re.match(r'^[IVX]+\.\s+', part):
                    current_section = part.strip()
                elif part.strip():
                    if current_section in sections:
                        sections[current_section] += "\n" + part
                    else:
                        sections[current_section] = part
        else:
            # Fallback: split by clear section headers
            lines = text.split('\n')
            current_section = "Document Content"
            current_content = []
            
            for line in lines:
                line = line.strip()
                # Check for section headers
                if (line.startswith(('I.', 'II.', 'III.', 'IV.', 'V.')) or
                    line.startswith(('A.', 'B.', 'C.', 'D.', 'E.')) or
                    line.isupper() and len(line) > 3):
                    
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    
                    # Start new section
                    current_section = line
                    current_content = []
                else:
                    if line:
                        current_content.append(line)
            
            # Save last section
            if current_content:
                sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _split_long_section(self, content: str, section_title: str) -> List[Document]:
        """Split long sections into smaller chunks"""
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        chunks = text_splitter.split_text(content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "section": section_title,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "document_type": "FDA_guidance",
                    "topic": "malaria_drug_development"
                }
            )
            documents.append(doc)
        
        return documents


# ===== VECTOR STORE MANAGER =====
class VectorStoreManager:
    """Manage FAISS vector store with embeddings"""
    
    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.retriever = None
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """Create FAISS vector store from documents"""
        
        print(f"Creating FAISS vector store with {len(documents)} documents...")
        
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        # Create the vector store
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.retrieval_k}
        )
        
        print("Vector store created successfully!")
        return self.vectorstore
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Document]:
        """Direct similarity search"""
        if not self.vectorstore:
            return []
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_retriever(self):
        """Get the retriever"""
        return self.retriever


# ===== QUERY ANALYZER =====
class QueryAnalyzer:
    """Analyze user queries to determine intent"""
    
    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.llm = ChatOpenAI(model_name=config.model_name, temperature=0.2)
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze user query"""
        
        # Simple rule-based analysis for now
        query_lower = query.lower()
        
        intent_type = "factual"
        if any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            intent_type = "comparative"
        elif any(word in query_lower for word in ["how to", "procedure", "steps", "process"]):
            intent_type = "procedural"
        elif any(word in query_lower for word in ["what is", "define", "definition", "meaning"]):
            intent_type = "definition"
        elif any(word in query_lower for word in ["summary", "summarize", "overview"]):
            intent_type = "summary"
        
        # Extract key entities
        entities = []
        medical_terms = ["plasmodium", "malaria", "parasitemia", "clinical trial", "endpoint", "efficacy"]
        for term in medical_terms:
            if term in query_lower:
                entities.append(term)
        
        return {
            "intent_type": intent_type,
            "key_entities": entities,
            "confidence": 0.8,
            "query_length": len(query.split())
        }


# ===== RESPONSE GENERATOR =====
class ResponseGenerator:
    """Generate responses using LangChain"""
    
    def __init__(self, config: ChatbotConfig, vectorstore_manager: VectorStoreManager):
        self.config = config
        self.vectorstore_manager = vectorstore_manager
        self.llm = ChatOpenAI(model_name=config.model_name, temperature=config.temperature)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self._setup_chains()
    
    def _setup_chains(self):
        """Setup response generation chains"""
        
        # Standard QA chain
        if self.vectorstore_manager.get_retriever():
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore_manager.get_retriever(),
                return_source_documents=True
            )
        else:
            self.qa_chain = None
        
        # Custom prompt for detailed responses
        self.custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert assistant for FDA malaria drug development guidance.
Use the provided context to answer questions accurately and comprehensively.

Context: {context}

Question: {question}

Please provide a detailed, accurate answer based on the FDA guidance document.
If the information is not in the context, say so clearly.

Answer:"""
        )
        
        self.custom_chain = LLMChain(llm=self.llm, prompt=self.custom_prompt)
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate comprehensive response to user query"""
        
        try:
            if self.qa_chain:
                # Use QA chain
                result = self.qa_chain.invoke({"query": query})
                
                return {
                    "answer": result.get("result", "No answer generated"),
                    "source_documents": [
                        {
                            "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in result.get("source_documents", [])
                    ],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Fallback: direct search and custom chain
                docs = self.vectorstore_manager.search_similar_documents(query, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                result = self.custom_chain.invoke({
                    "context": context,
                    "question": query
                })
                
                return {
                    "answer": result.get("text", "No answer generated"),
                    "source_documents": [
                        {
                            "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in docs
                    ],
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "answer": f"Error generating response: {e}",
                "source_documents": [],
                "error": str(e)
            }


# ===== MAIN CHATBOT CLASS =====
class MalariaGuidanceChatbot:
    """Main chatbot class"""
    
    def __init__(self, pdf_content: str):
        """Initialize the chatbot with FDA guidance document"""
        
        print("🏥 Initializing FDA Malaria Guidance Chatbot...")
        
        try:
            # Setup configuration
            self.config = ChatbotConfig.from_env()
            
            # Initialize components
            self.doc_processor = DocumentProcessor(self.config)
            self.vectorstore_manager = VectorStoreManager(self.config)
            self.query_analyzer = QueryAnalyzer(self.config)
            
            # Process document
            print("📄 Processing FDA guidance document...")
            documents = self.doc_processor.extract_text_from_pdf_content(pdf_content)
            
            if not documents:
                raise ValueError("No documents were processed from the PDF content")
            
            # Create vector store
            self.vectorstore_manager.create_vectorstore(documents)
            
            # Initialize response generator
            self.response_generator = ResponseGenerator(self.config, self.vectorstore_manager)
            
            print(f"✅ Chatbot initialized successfully!")
            print(f"📊 Processed {len(documents)} document chunks")
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Main chat interface"""
        
        print(f"\nUser Query: {query}")
        
        # Analyze query
        query_analysis = self.query_analyzer.analyze_query(query)
        print(f"🔍 Query Intent: {query_analysis.get('intent_type', 'unknown')}")
        
        # Generate response
        response = self.response_generator.generate_response(query)
        response['query_analysis'] = query_analysis
        
        return response
    
    def get_document_info(self) -> Dict[str, Any]:
        """Get information about the processed document"""
        return {
            "total_chunks": len(self.doc_processor.documents),
            "sections": list(set(doc.metadata.get("section", "Unknown") for doc in self.doc_processor.documents))
        }


# ===== DEMO FUNCTION =====
def run_demo():
    """Run a simple demo of the chatbot"""
    
    # Sample FDA document content
    fda_content = """
    I. INTRODUCTION
    
    This guidance is intended to assist sponsors in the overall development program for drug and biological products for the treatment of malaria, caused by clinically relevant Plasmodium species (e.g., P. falciparum, P. vivax, P. malariae, P. ovale, and P. knowlesi).
    
    II. BACKGROUND
    
    Malaria is a parasitic disease primarily transmitted by Anopheles species mosquitoes. Clinical manifestations, including the severity of malaria, are dependent on the infecting species and host factors.
    
    The terminology used for assessing clinical and parasitological responses includes:
    • Clinical cure is directed at eradication of the parasites in addition to adequate clinical response
    • Radical cure eliminates both erythrocytic and exoerythrocytic stages of infection, including the hypnozoites
    • Recrudescence is recurrence of the existing parasitemia due to the survival of the erythrocytic stage parasites
    • Relapse is recurrence of the original parasitemia attributable to the Plasmodium parasites that have a hypnozoite stage
    
    III. DEVELOPMENT CONSIDERATIONS
    
    B. Uncomplicated Malaria
    
    1. Trial Design
    Use of an active control regimen containing FDA-approved antimalarial drugs is strongly recommended.
    
    2. Trial Population
    Some relevant inclusion criteria include:
    — Identification of the Plasmodium species on peripheral blood smears; parasitemia should be limited to values between 1000/µL and <250,000/µL
    — Fever should be documented at study entry
    — At least two symptoms or signs of malaria should be present
    
    4. Efficacy Endpoints
    Primary endpoint: Clinical cure is typically defined as the absence of parasitemia on day 28 in participants who did not previously meet criteria for treatment failure.
    
    Secondary endpoints:
    — Parasite clearance time
    — Fever clearance time
    — Corrected/adjusted cure rates
    
    C. Severe or Complicated Malaria
    
    2. Trial Population
    Inclusion criteria:
    — Participants with hyperparasitemia: Parasitemia of greater than or equal to 10 percent or parasitemia of greater than or equal to 5 percent accompanied by major complications
    
    3. Efficacy Endpoints
    Primary endpoint: All-cause mortality
    
    Secondary endpoints:
    — Incidence of neurological sequelae
    — Combined death or neurological sequelae
    """
    
    print("🏥 FDA MALARIA GUIDANCE CHATBOT DEMO")
    print("=" * 50)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found!")
        print("Please create a .env file with: OPENAI_API_KEY=your-key-here")
        return
    
    try:
        # Initialize chatbot
        chatbot = MalariaGuidanceChatbot(fda_content)
        
        # Test questions
        test_questions = [
            "What are the primary endpoints for uncomplicated malaria trials?",
            "What is the difference between recrudescence and relapse?",
            "What are the inclusion criteria for severe malaria studies?"
        ]
        
        print("\n🧪 TESTING CHATBOT:")
        print("=" * 30)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Question: {question}")
            print("-" * 40)
            
            response = chatbot.chat(question)
            answer = response.get('answer', 'No answer')
            
            print(f"Answer: {answer}")
            
            # Show sources
            sources = response.get('source_documents', [])
            if sources:
                print(f"Sources: {len(sources)} relevant sections")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have set OPENAI_API_KEY in .env file")
        print("2. Check that all packages are installed")
        print("3. Verify your API key is valid and has credits")


if __name__ == "__main__":
    run_demo()