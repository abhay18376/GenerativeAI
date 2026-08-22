# Resume Screening Application for Google Colab with FAISS and OpenAI
# Run each cell in sequence in Google Colab

# ============================================
# CELL 1: Install Required Libraries
# ============================================
"""
!pip install langchain==0.1.0
!pip install langchain-core==0.1.15
!pip install langchain-openai==0.0.5
!pip install langchain-community==0.0.13
!pip install openai==1.12.0
!pip install python-dotenv==1.0.0
!pip install docx2txt==0.8
!pip install pypdf==3.17.1
!pip install faiss-cpu==1.7.4
!pip install gradio==4.19.2
!pip install tiktoken==0.5.2
"""

# ============================================
# CELL 2: Import Libraries and Setup
# ============================================
import os
import gradio as gr
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnableMap
import re
import tempfile
import pickle
from typing import Optional
import shutil
from getpass import getpass

# ============================================
# CELL 3: Set up API Key
# ============================================
# Option 1: Direct input in Colab
OPENAI_API_KEY = getpass("Enter your OpenAI API Key: ")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Option 2: If you prefer using environment variables
# Create a .env file in your Google Drive and mount it
"""
from google.colab import drive
drive.mount('/content/drive')

# Then load from .env file
from dotenv import load_dotenv
load_dotenv('/content/drive/MyDrive/your_folder/.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
"""

# ============================================
# CELL 4: Initialize Components
# ============================================

# Setup embedding model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

# Initialize or load FAISS vector store
VECTOR_STORE_DIR = "/content/faiss_index"
VECTOR_STORE_INDEX = os.path.join(VECTOR_STORE_DIR, "index.faiss")
VECTOR_STORE_PKL = os.path.join(VECTOR_STORE_DIR, "index.pkl")

def initialize_vector_store():
    """Initialize or load FAISS vector store using FAISS native methods"""
    global vectorstore
    
    # Create directory if it doesn't exist
    if not os.path.exists(VECTOR_STORE_DIR):
        os.makedirs(VECTOR_STORE_DIR)
    
    # Check if existing store exists
    if os.path.exists(VECTOR_STORE_INDEX) and os.path.exists(VECTOR_STORE_PKL):
        try:
            # Load existing FAISS store using native method
            vectorstore = FAISS.load_local(
                VECTOR_STORE_DIR, 
                embedding_model,
                index_name="index",
                allow_dangerous_deserialization=True  # Required for loading
            )
            print("Loaded existing FAISS vector store")
        except Exception as e:
            print(f"Error loading existing store: {e}")
            # Create new if loading fails
            from langchain.docstore.document import Document
            dummy_doc = Document(page_content="Initial document", metadata={"source": "init"})
            vectorstore = FAISS.from_documents([dummy_doc], embedding_model)
            print("Created new FAISS vector store after load error")
    else:
        # Create new FAISS store
        from langchain.docstore.document import Document
        dummy_doc = Document(page_content="Initial document", metadata={"source": "init"})
        vectorstore = FAISS.from_documents([dummy_doc], embedding_model)
        # Save it immediately
        vectorstore.save_local(VECTOR_STORE_DIR, index_name="index")
        print("Created and saved new FAISS vector store")
    
    return vectorstore

vectorstore = initialize_vector_store()

# ============================================
# CELL 5: Document Processing Functions
# ============================================

def extract_text_from_resume(file_path):
    """Extract text from uploaded resume files"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])
        return text
    except Exception as e:
        raise Exception(f"Error extracting text: {str(e)}")

def split_text(text):
    """Split text into chunks for processing"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.create_documents([text])

def store_resume_analysis(resume_text, analysis, candidate_name):
    """Store resume analysis in FAISS vector store using native save method"""
    global vectorstore
    
    try:
        # Create metadata for the document
        metadata = {
            "candidate_name": candidate_name,
            "type": "resume_analysis",
            "full_analysis": analysis[:1000]  # Store first 1000 chars in metadata
        }
        
        # Split the analysis into chunks
        documents = split_text(f"Resume: {resume_text}\n\nAnalysis: {analysis}")
        
        # Add metadata to each document
        for doc in documents:
            doc.metadata = metadata
        
        # Add to FAISS store
        vectorstore.add_documents(documents)
        
        # Save using FAISS native method instead of pickle
        vectorstore.save_local(VECTOR_STORE_DIR, index_name="index")
        
        return True
        
    except Exception as e:
        print(f"Error storing analysis: {str(e)}")
        return False

def extract_suitability_score(text):
    """Extract percentage score from analysis text"""
    match = re.search(r"Suitability Score[:\s]+(\d{1,3})%", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

# ============================================
# CELL 6: Main Analysis Function
# ============================================

def analyze_resume(job_requirements, resume_file):
    """Main function to analyze resume against job requirements"""
    
    if not job_requirements or not resume_file:
        return "Please provide both job requirements and a resume file.", None, None
    
    try:
        # Extract text from resume
        resume_text = extract_text_from_resume(resume_file.name)
        
        # Initialize OpenAI LLM
        llm = ChatOpenAI(
            model="gpt-4",  # You can also use "gpt-3.5-turbo" for lower cost
            openai_api_key=OPENAI_API_KEY,
            temperature=0.2,
            max_tokens=1500
        )
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["job_requirements", "resume_text"],
            template="""
            You are an expert HR and recruitment specialist. Analyze the resume below against the job requirements.

            Job Requirements:
            {job_requirements}

            Resume:
            {resume_text}

            Provide a comprehensive and structured analysis including:
            
            1. **Skills Assessment**: How well do the candidate's skills match the requirements?
            2. **Experience Relevance**: Does their experience align with what's needed?
            3. **Education Evaluation**: Is their educational background suitable?
            4. **Strengths**: What are the candidate's main strengths for this position?
            5. **Weaknesses/Gaps**: What key requirements are missing or weak?
            6. **Overall Recommendation**: Should this candidate be considered for interview?
            
            At the end, clearly state a "Suitability Score" as a percentage (0-100%) based on how well 
            the resume aligns with the job requirements.
            
            Format the final score as: Suitability Score: XX%
            """
        )
        
        # Create and run the chain using LCEL
        chain = (
            RunnableMap({
                "job_requirements": lambda x: x["job_requirements"],
                "resume_text": lambda x: x["resume_text"]
            })
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        # Run the analysis
        analysis = chain.invoke({
            "job_requirements": job_requirements,
            "resume_text": resume_text
        })
        
        # Extract suitability score
        suitability_score = extract_suitability_score(analysis)
        
        # Extract candidate name from resume (simple heuristic)
        lines = resume_text.split('\n')
        candidate_name = lines[0] if lines else "Unknown Candidate"
        
        # Store in FAISS vector store
        store_resume_analysis(resume_text, analysis, candidate_name)
        
        # Format the score display
        score_display = f"📊 **Suitability Score: {suitability_score}%**" if suitability_score else "Score not detected"
        
        # Create downloadable content
        download_content = f"""
RESUME SCREENING ANALYSIS REPORT
================================

Candidate: {candidate_name}
Date: {os.popen('date').read().strip()}

JOB REQUIREMENTS:
----------------
{job_requirements}

RESUME CONTENT:
--------------
{resume_text}

AI ANALYSIS:
-----------
{analysis}

================================
        """
        
        return analysis, score_display, download_content
        
    except Exception as e:
        return f"Error during analysis: {str(e)}", None, None

# ============================================
# CELL 7: Search Similar Resumes Function
# ============================================

def search_similar_resumes(query, k=3):
    """Search for similar resumes in the vector store"""
    try:
        results = vectorstore.similarity_search(query, k=k)
        
        if not results:
            return "No similar resumes found in the database."
        
        output = "**Similar Resumes Found:**\n\n"
        for i, doc in enumerate(results, 1):
            candidate_name = doc.metadata.get("candidate_name", "Unknown")
            analysis_snippet = doc.metadata.get("full_analysis", "No analysis available")[:200]
            output += f"**{i}. {candidate_name}**\n"
            output += f"   Analysis excerpt: {analysis_snippet}...\n\n"
        
        return output
    except Exception as e:
        return f"Error searching resumes: {str(e)}"

# ============================================
# CELL 7B: Batch Processing Function
# ============================================

def analyze_multiple_resumes(job_requirements, resume_files):
    """Analyze multiple resumes against job requirements"""
    
    if not job_requirements or not resume_files:
        return "Please provide both job requirements and resume files.", None
    
    all_results = []
    summary_data = []
    
    for i, resume_file in enumerate(resume_files):
        try:
            # Extract text from resume
            resume_text = extract_text_from_resume(resume_file.name)
            
            # Get candidate name
            lines = resume_text.split('\n')
            candidate_name = lines[0] if lines else f"Candidate {i+1}"
            
            # Initialize OpenAI LLM
            llm = ChatOpenAI(
                model="gpt-4",  # You can also use "gpt-3.5-turbo" for lower cost
                openai_api_key=OPENAI_API_KEY,
                temperature=0.2,
                max_tokens=1500
            )
            
            # Create prompt template
            prompt_template = PromptTemplate(
                input_variables=["job_requirements", "resume_text"],
                template="""
                You are an expert HR and recruitment specialist. Analyze the resume below against the job requirements.

                Job Requirements:
                {job_requirements}

                Resume:
                {resume_text}

                Provide a comprehensive and structured analysis including:
                
                1. **Skills Assessment**: How well do the candidate's skills match the requirements?
                2. **Experience Relevance**: Does their experience align with what's needed?
                3. **Education Evaluation**: Is their educational background suitable?
                4. **Strengths**: What are the candidate's main strengths for this position?
                5. **Weaknesses/Gaps**: What key requirements are missing or weak?
                6. **Overall Recommendation**: Should this candidate be considered for interview?
                
                At the end, clearly state a "Suitability Score" as a percentage (0-100%) based on how well 
                the resume aligns with the job requirements.
                
                Format the final score as: Suitability Score: XX%
                """
            )
            
            # Create and run the chain
            chain = (
                RunnableMap({
                    "job_requirements": lambda x: x["job_requirements"],
                    "resume_text": lambda x: x["resume_text"]
                })
                | prompt_template
                | llm
                | StrOutputParser()
            )
            
            # Run the analysis
            analysis = chain.invoke({
                "job_requirements": job_requirements,
                "resume_text": resume_text
            })
            
            # Extract suitability score
            suitability_score = extract_suitability_score(analysis)
            
            # Store in FAISS vector store
            store_resume_analysis(resume_text, analysis, candidate_name)
            
            # Add to results
            all_results.append({
                "candidate": candidate_name,
                "score": suitability_score if suitability_score else 0,
                "analysis": analysis,
                "file": resume_file.name
            })
            
            summary_data.append({
                "Candidate": candidate_name,
                "File": resume_file.name,
                "Score": f"{suitability_score}%" if suitability_score else "N/A"
            })
            
        except Exception as e:
            all_results.append({
                "candidate": f"Error processing {resume_file.name}",
                "score": 0,
                "analysis": f"Error: {str(e)}",
                "file": resume_file.name
            })
    
    # Sort results by score
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Create summary report
    summary_md = "## 📊 Batch Analysis Summary\n\n"
    summary_md += "### Ranking by Suitability Score:\n\n"
    
    for i, result in enumerate(all_results, 1):
        summary_md += f"**{i}. {result['candidate']}** - Score: {result['score']}%\n"
        summary_md += f"   File: {result['file']}\n\n"
    
    # Create detailed report
    detailed_report = "# BATCH RESUME SCREENING REPORT\n"
    detailed_report += "=" * 50 + "\n\n"
    detailed_report += f"Total Resumes Analyzed: {len(resume_files)}\n"
    detailed_report += f"Date: {os.popen('date').read().strip()}\n\n"
    detailed_report += "JOB REQUIREMENTS:\n"
    detailed_report += "-" * 20 + "\n"
    detailed_report += job_requirements + "\n\n"
    detailed_report += "=" * 50 + "\n\n"
    
    for i, result in enumerate(all_results, 1):
        detailed_report += f"\nCANDIDATE {i}: {result['candidate']}\n"
        detailed_report += f"File: {result['file']}\n"
        detailed_report += f"Score: {result['score']}%\n"
        detailed_report += "-" * 30 + "\n"
        detailed_report += result['analysis'] + "\n\n"
        detailed_report += "=" * 50 + "\n"
    
    return summary_md, detailed_report

# ============================================
# CELL 8: Gradio Interface
# ============================================

def create_gradio_interface():
    """Create Gradio interface for the application"""
    
    with gr.Blocks(title="Resume Screening with FAISS & OpenAI", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 📋 Resume Screening Application
        ### Powered by FAISS Vector Store and OpenAI GPT-4
        
        Upload a resume and provide job requirements to get an AI-powered analysis with suitability scoring.
        """)
        
        with gr.Tab("Single Resume Analysis"):
            with gr.Row():
                with gr.Column():
                    job_req_input = gr.Textbox(
                        label="Job Requirements",
                        placeholder="Enter the job requirements, skills needed, experience required, etc.",
                        lines=10
                    )
                
                with gr.Column():
                    resume_upload = gr.File(
                        label="Upload Resume",
                        file_types=[".pdf", ".docx", ".txt"],
                        type="filepath"
                    )
            
            analyze_btn = gr.Button("🔍 Analyze Resume", variant="primary")
            
            with gr.Row():
                analysis_output = gr.Markdown(label="Analysis Results")
            
            with gr.Row():
                score_output = gr.Markdown(label="Suitability Score")
            
            with gr.Row():
                download_text = gr.Textbox(
                    label="Analysis Report (Copy or Download)",
                    lines=10,
                    visible=False
                )
            
            analyze_btn.click(
                fn=analyze_resume,
                inputs=[job_req_input, resume_upload],
                outputs=[analysis_output, score_output, download_text]
            ).then(
                lambda x: gr.update(visible=True) if x else gr.update(visible=False),
                inputs=[download_text],
                outputs=[download_text]
            )
        
        with gr.Tab("Batch Resume Analysis"):
            gr.Markdown("""
            ### 📂 Upload Multiple Resumes
            Upload multiple resumes at once to compare candidates for the same position.
            """)
            
            with gr.Row():
                with gr.Column():
                    batch_job_req = gr.Textbox(
                        label="Job Requirements",
                        placeholder="Enter the job requirements for all resumes",
                        lines=10
                    )
                
                with gr.Column():
                    batch_resume_upload = gr.File(
                        label="Upload Multiple Resumes",
                        file_types=[".pdf", ".docx", ".txt"],
                        file_count="multiple",
                        type="filepath"
                    )
            
            batch_analyze_btn = gr.Button("🔍 Analyze All Resumes", variant="primary")
            
            with gr.Row():
                batch_summary = gr.Markdown(label="Summary Results")
            
            with gr.Row():
                batch_detailed_report = gr.Textbox(
                    label="Detailed Batch Report (Copy or Download)",
                    lines=15,
                    max_lines=30
                )
            
            batch_analyze_btn.click(
                fn=analyze_multiple_resumes,
                inputs=[batch_job_req, batch_resume_upload],
                outputs=[batch_summary, batch_detailed_report]
            )
        
        with gr.Tab("Search Similar Resumes"):
            gr.Markdown("### 🔎 Search for similar resumes in the database")
            
            search_input = gr.Textbox(
                label="Search Query",
                placeholder="Enter skills, job title, or requirements to find similar resumes",
                lines=3
            )
            
            search_btn = gr.Button("Search", variant="primary")
            search_results = gr.Markdown(label="Search Results")
            
            search_btn.click(
                fn=search_similar_resumes,
                inputs=[search_input],
                outputs=[search_results]
            )
        
        gr.Markdown("""
        ---
        ### 📝 Instructions:
        1. Enter detailed job requirements in the left panel
        2. Upload a resume (PDF, DOCX, or TXT format)
        3. Click "Analyze Resume" to get AI-powered insights
        4. Use the Search tab to find similar resumes in the database
        
        ### 💾 Data Storage:
        - All analyses are stored in a FAISS vector database for future reference
        - You can search through previously analyzed resumes using semantic search
        """)
    
    return demo

# ============================================
# CELL 9: Launch the Application
# ============================================

# Create and launch the Gradio interface
demo = create_gradio_interface()

# Launch with public link for Colab
demo.launch(share=True, debug=True)

# ============================================
# Optional: Save/Load Vector Store to Google Drive
# ============================================
"""
# To save your vector store to Google Drive:
from google.colab import drive
drive.mount('/content/drive')

# Create directory in Drive if it doesn't exist
import os
drive_faiss_dir = '/content/drive/MyDrive/resume_screening/faiss_index'
os.makedirs(drive_faiss_dir, exist_ok=True)

# Save vector store to Drive using FAISS native method
vectorstore.save_local(drive_faiss_dir, index_name="index")

# Load vector store from Drive
vectorstore = FAISS.load_local(
    drive_faiss_dir, 
    embedding_model,
    index_name="index",
    allow_dangerous_deserialization=True
)
"""