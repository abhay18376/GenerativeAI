import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import Settings
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, ToolMetadata
#from llama_index.agent import ReActAgent
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
import types

# Load environment variables
load_dotenv()

class CompetitiveAnalysisAgent:
    def __init__(self):
        # Initialize Cohere models
        self.embed_model = CohereEmbedding(
            model_name="embed-english-v3.0",
            cohere_api_key=os.getenv("COHERE_API_KEY")
        )
        
        self.llm = Cohere(
            model="command-a-03-2025",
            api_key=os.getenv("COHERE_API_KEY")
        )
        
        # Configure global settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        
        self.index = None
        self.query_engine = None
        self.agent = None
        self.workflow=None
        
    def load_and_preprocess_data(self, csv_file_path):
        """Load and preprocess competitor data from CSV"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            
            # Convert DataFrame to documents
            documents = []
            for _, row in df.iterrows():
                document_content = f"""
                Competitor: {row['Competitor Name']}
                Product: {row['Product Description']}
                Marketing Strategy: {row['Marketing Strategy']}
                Financial Summary: {row['Financial Summary']}
                """
                documents.append(document_content)
            
            print(f"Loaded {len(documents)} competitor records")
            return documents
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
    
    def create_index(self, documents):
        """Create vector store index from documents"""
        try:
            from llama_index.core import Document
            from llama_index.core.schema import TextNode
            
            # Convert documents to LlamaIndex Document objects
            llama_documents = []
            for doc_content in documents:
                llama_documents.append(Document(text=doc_content))
            
            # Create index
            self.index = VectorStoreIndex.from_documents(
                llama_documents,
                embed_model=self.embed_model
            )
            
            print("Vector store index created successfully")
            return self.index
            
        except Exception as e:
            print(f"Error creating index: {e}")
            return None
    
    def setup_query_engine(self):
        """Set up the query engine with retrieval"""
        if self.index is None:
            print("Index not created. Please create index first.")
            return None
        
        try:
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=3,
                llm=self.llm
            )
            print("Query engine setup completed")
            return self.query_engine
            
        except Exception as e:
            print(f"Error setting up query engine: {e}")
            return None
    
    def create_competitive_analysis_agent(self):
        """Create the competitive analysis agent"""
        if self.query_engine is None:
            print("Query engine not setup. Please setup query engine first.")
            return None
        
        try:
            # Define tools for the agent
            sales_tool = QueryEngineTool.from_defaults(
                query_engine=self.query_engine,
                name="competitor_data",
                description="Provides access to competitor information including products, marketing strategies, and financial data"
            )
            
            memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
            self.agent = ReActAgent(tools=[sales_tool], llm=self.llm,memory=memory, verbose=True)
           
           
            print("Competitive analysis agent created successfully")
            return self.agent
            
        except Exception as e:
            print(f"Error creating agent: {e}")
            return None
    
    def query_competitor_data(self, query):
        """Query the competitor data directly"""
        if self.query_engine is None:
            return "Query engine not initialized. Please setup the system first."
        
        try:
            response = self.query_engine.query(query)
            return response
            
        except Exception as e:
            return f"Error querying data: {e}"
    

    def ask_agent(self, question):
        """Ask a question to the competitive analysis agent"""
        if self.agent is None:
            return "Agent not initialized. Please create the agent first."
        
        try:           
            response = self.agent.chat(question)
            return response
        except Exception as e:
            return f"Error asking agent: {e}"


# Example usage and demonstration
def main():
    
    # Initialize the competitive analysis system
    print("Initializing Competitive Analysis Agent...")
    ca_agent = CompetitiveAnalysisAgent()
    
    # Load and preprocess data
    documents = ca_agent.load_and_preprocess_data("competitors.csv")
    
    if not documents:
        print("No documents loaded. Exiting.")
        return
    
    # Create vector index
    ca_agent.create_index(documents)
    
    # Setup query engine
    ca_agent.setup_query_engine()
    
    # Create the agent
    ca_agent.create_competitive_analysis_agent()
    
    print("\n" + "="*50)
    print("COMPETITIVE ANALYSIS AGENT READY")
    print("="*50)
    
    # Example queries
    example_queries = [
        "Which competitors focus on retail analytics and what are their marketing strategies?",
        "Compare the financial performance and growth rates of all competitors.",
        "What are the different funding strategies among the competitors?",
        "Which companies offer AI-powered solutions and how do they market them?",
        "Provide a strategic analysis of the competitive landscape with recommendations."
    ]
    

    # Test the system
    for i, query in enumerate(example_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        # Option 1: Direct query (simple retrieval)
        print("\nDirect Query Result:")
        direct_response = ca_agent.query_competitor_data(query)
        print(str(direct_response))
        
        # Option 2: Agent response (reasoning + retrieval)
        print("\nAgent Analysis:")
        agent_response = ca_agent.ask_agent(query)
        print(str(agent_response))
        
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()