"""
============================================================
Requirements (install with single command):
pip install langchain langchain-openai python-dotenv faiss-cpu tiktoken colorama pydantic

UPDATE YOUR API KEY BELOW!
"""

# ===== CONFIGURATION SECTION =====
# 🔑 REPLACE WITH YOUR ACTUAL OPENAI API KEY
OPENAI_API_KEY = ""  # ⚠️ UPDATE THIS LINE!

# ===== IMPORTS SECTION =====
import os
import json
from typing import List, Dict, Any
from datetime import datetime

# Third-party imports
try:
    from colorama import init, Fore, Style

    # Initialize colorama for colored output
    init()

    # Set API key in environment
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

except ImportError as e:
    print(f"Missing required package: {e}")
    print("\nPlease install all requirements:")
    print("pip install langchain langchain-openai python-dotenv faiss-cpu tiktoken colorama pydantic")
    exit(1)

# LangChain imports
try:
    from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
    from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
    from langchain.chains import LLMChain, SimpleSequentialChain, ConversationChain
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
    from langchain.output_parsers import PydanticOutputParser, StructuredOutputParser, ResponseSchema
    from langchain.schema import HumanMessage, SystemMessage, AIMessage, Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.tools import Tool
    from langchain.agents import initialize_agent, AgentType
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from pydantic import BaseModel, Field
except ImportError as e:
    print(f"Missing LangChain component: {e}")
    print("\nPlease install all requirements:")
    print("pip install langchain langchain-openai python-dotenv faiss-cpu tiktoken colorama pydantic")
    exit(1)


# ===== MAIN ASSISTANT CLASS =====
class PersonalAIAssistant:
    """
    A comprehensive AI Assistant demonstrating all LangChain concepts:
    1. LLMs - Basic language model usage
    2. Prompt Templates - Structured prompts
    3. Chains - Combining components
    4. Sequential Chains - Multi-step processing
    5. Memory - Conversation history
    6. Output Parsers - Structured outputs
    7. Chat Models - Conversational AI
    8. RAG - Document retrieval
    9. Agents & Tools - Task automation
    10. Streaming - Real-time responses
    """

    def __init__(self):
        """Initialize all components of the AI Assistant"""
        print(f"{Fore.CYAN}🚀 Initializing Personal AI Assistant...{Style.RESET_ALL}\n")

        # Check API key
        if OPENAI_API_KEY == "sk-your-key-here" or not OPENAI_API_KEY:
            print(f"{Fore.RED}❌ ERROR: Please update your OPENAI_API_KEY!{Style.RESET_ALL}")
            print("\nTo fix this:")
            print("1. Open this Python file in a text editor")
            print("2. Find the line: OPENAI_API_KEY = \"sk-your-key-here\"")
            print("3. Replace 'sk-your-key-here' with your actual OpenAI API key")
            print("4. Save the file and run this script again")
            exit(1)

        try:
            # CONCEPT 1: Initialize LLMs
            self.llm = OpenAI(
                temperature=0.7,
                model_name="gpt-3.5-turbo-instruct",
                max_tokens=150
            )

            # CONCEPT 7: Chat Model for conversations
            self.chat_model = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=150
            )

            # CONCEPT 5: Memory for conversation
            self.memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )

            # CONCEPT 8: Setup knowledge base (RAG)
            self._setup_knowledge_base()

            # CONCEPT 9: Setup tools for agent
            self._setup_tools()

            print(f"{Fore.GREEN}✅ Assistant initialized successfully!{Style.RESET_ALL}\n")

        except Exception as e:
            print(f"{Fore.RED}❌ Initialization Error: {e}{Style.RESET_ALL}")
            print("\nPossible causes:")
            print("1. Invalid API key")
            print("2. No internet connection")
            print("3. OpenAI API is down")
            exit(1)

    def _setup_knowledge_base(self):
        """CONCEPT 8: RAG - Create a knowledge base with embeddings"""
        print(f"{Fore.YELLOW}📚 Setting up knowledge base...{Style.RESET_ALL}")

        # Sample knowledge documents
        knowledge_docs = [
            "LangChain is a framework for developing applications powered by language models.",
            "Prompt templates help structure inputs to language models for consistent results.",
            "Chains combine multiple LangChain components into a single workflow.",
            "Memory allows chatbots to remember previous conversations.",
            "Agents use tools to perform specific tasks like calculations or web searches.",
            "RAG (Retrieval Augmented Generation) combines document search with LLM generation.",
            "Vector databases store document embeddings for semantic search.",
            "Output parsers convert LLM text outputs into structured data.",
            "Sequential chains process data through multiple steps.",
            "Streaming provides real-time token-by-token responses."
        ]

        # Create Document objects
        docs = [Document(page_content=text, metadata={"topic": "langchain"}) for text in knowledge_docs]

        # Split documents (for longer texts)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        split_docs = text_splitter.split_documents(docs)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_documents(split_docs, embeddings)

        # Create QA chain for knowledge queries
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever()
        )

    def _setup_tools(self):
        """CONCEPT 9: Tools - Create tools for the agent"""

        def calculator(expression: str) -> str:
            """Perform mathematical calculations"""
            try:
                # Safe evaluation of mathematical expressions
                allowed_names = {
                    k: v for k, v in math.__dict__.items() if not k.startswith("__")
                }
                result = eval(expression, {"__builtins__": {}}, allowed_names)
                return f"The result is: {result}"
            except:
                try:
                    # Fallback to simple evaluation
                    result = eval(expression, {"__builtins__": {}})
                    return f"The result is: {result}"
                except:
                    return "Error: Invalid mathematical expression"

        def get_datetime() -> str:
            """Get current date and time"""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def word_counter(text: str) -> str:
            """Count words in text"""
            word_count = len(text.split())
            char_count = len(text)
            return f"Words: {word_count}, Characters: {char_count}"

        self.tools = [
            Tool(name="Calculator", func=calculator,
                 description="Perform math calculations. Input: mathematical expression like '2+2' or '10*5'"),
            Tool(name="DateTime", func=get_datetime,
                 description="Get current date and time"),
            Tool(name="WordCounter", func=word_counter,
                 description="Count words and characters in text"),
            Tool(name="KnowledgeBase", func=lambda q: self.qa_chain.invoke({"query": q})["result"],
                 description="Search the knowledge base about LangChain concepts")
        ]

        # Create agent with tools
        self.agent = initialize_agent(
            self.tools,
            self.chat_model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    def demonstrate_prompt_templates(self, topic: str = "recursion"):
        """CONCEPT 2: Prompt Templates - Show different template types"""
        print(f"\n{Fore.CYAN}📝 CONCEPT 2: Prompt Templates{Style.RESET_ALL}")
        print("-" * 50)

        try:
            # Basic template
            basic_template = PromptTemplate(
                input_variables=["topic"],
                template="Explain {topic} in simple terms:"
            )

            # Format and use template
            prompt = basic_template.format(topic=topic)
            print(f"Formatted Prompt: {prompt}")

            response = self.llm.invoke(prompt)
            print(f"Basic Template Result: {response.strip()}\n")

            # Few-shot template
            examples = [
                {"concept": "function", "explanation": "A reusable block of code"},
                {"concept": "variable", "explanation": "A container for storing data"},
            ]

            example_template = PromptTemplate(
                input_variables=["concept", "explanation"],
                template="Concept: {concept}\nSimple explanation: {explanation}"
            )

            few_shot = FewShotPromptTemplate(
                examples=examples,
                example_prompt=example_template,
                prefix="Explain programming concepts simply:\n",
                suffix="\nConcept: {input}\nSimple explanation:",
                input_variables=["input"]
            )

            few_shot_prompt = few_shot.format(input=topic)
            few_shot_response = self.llm.invoke(few_shot_prompt)

            print(f"Few-Shot Template Result: {few_shot_response.strip()}")

            return response
        except Exception as e:
            print(f"{Fore.RED}Error in prompt templates: {e}{Style.RESET_ALL}")
            return None

    def demonstrate_chains(self, product: str = "smart water bottle"):
        """CONCEPT 3 & 4: Chains and Sequential Chains"""
        print(f"\n{Fore.CYAN}⛓️ CONCEPT 3 & 4: Chains & Sequential Chains{Style.RESET_ALL}")
        print("-" * 50)

        try:
            # CONCEPT 3: Simple Chain
            product_prompt = PromptTemplate(
                input_variables=["product"],
                template="Generate a creative name for this product: {product}"
            )
            name_chain = LLMChain(llm=self.llm, prompt=product_prompt)

            # CONCEPT 4: Sequential Chain
            tagline_prompt = PromptTemplate(
                input_variables=["product_name"],
                template="Create a catchy tagline for a product called: {product_name}"
            )
            tagline_chain = LLMChain(llm=self.llm, prompt=tagline_prompt)

            # Combine chains
            sequential_chain = SimpleSequentialChain(
                chains=[name_chain, tagline_chain],
                verbose=True
            )

            result = sequential_chain.invoke(product)
            print(f"\nFinal Tagline: {result['output']}")

            return result
        except Exception as e:
            print(f"{Fore.RED}Error in chains: {e}{Style.RESET_ALL}")
            return None

    def demonstrate_output_parsing(self, request: str = "organize a birthday party"):
        """CONCEPT 6: Output Parsers - Structure the output"""
        print(f"\n{Fore.CYAN}🔧 CONCEPT 6: Output Parsing{Style.RESET_ALL}")
        print("-" * 50)

        try:
            # Define structured output
            class TaskList(BaseModel):
                title: str = Field(description="Title of the task list")
                tasks: List[str] = Field(description="List of tasks to complete")
                priority: str = Field(description="Priority level: low, medium, or high")
                estimated_time: int = Field(description="Estimated time in minutes")

            # Create parser
            parser = PydanticOutputParser(pydantic_object=TaskList)

            # Create prompt with format instructions
            prompt = PromptTemplate(
                template="Create a task list for: {request}\n{format_instructions}",
                input_variables=["request"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            # Generate and parse
            formatted_prompt = prompt.format(request=request)
            output = self.llm.invoke(formatted_prompt)

            try:
                parsed_output = parser.parse(output)
                print(f"✅ Parsed Task List:")
                print(f"  Title: {parsed_output.title}")
                print(f"  Tasks: {parsed_output.tasks}")
                print(f"  Priority: {parsed_output.priority}")
                print(f"  Time: {parsed_output.estimated_time} minutes")
                return parsed_output
            except Exception as parse_error:
                print(f"❌ Parsing failed: {parse_error}")
                print(f"Raw output: {output[:200]}...")
                return None
        except Exception as e:
            print(f"{Fore.RED}Error in output parsing: {e}{Style.RESET_ALL}")
            return None

    def chat_with_memory(self, user_input: str):
        """CONCEPT 5: Memory - Conversation with history"""
        print(f"\n{Fore.CYAN}🧠 CONCEPT 5: Conversation with Memory{Style.RESET_ALL}")
        print("-" * 50)

        try:
            # Create conversation chain with memory
            conversation = ConversationChain(
                llm=self.chat_model,
                memory=self.memory,
                verbose=True
            )

            response = conversation.invoke({"input": user_input})
            return response['response']
        except Exception as e:
            print(f"{Fore.RED}Error in memory conversation: {e}{Style.RESET_ALL}")
            return None

    def search_knowledge(self, query: str = "What is RAG?"):
        """CONCEPT 8: RAG - Search knowledge base"""
        print(f"\n{Fore.CYAN}🔍 CONCEPT 8: RAG Knowledge Search{Style.RESET_ALL}")
        print("-" * 50)

        try:
            result = self.qa_chain.invoke({"query": query})
            print(f"Knowledge Base Answer: {result['result']}")
            return result['result']
        except Exception as e:
            print(f"{Fore.RED}Error in knowledge search: {e}{Style.RESET_ALL}")
            return None

    def use_agent(self, task: str = "What's 25 * 4?"):
        """CONCEPT 9: Agents - Use tools to complete tasks"""
        print(f"\n{Fore.CYAN}🤖 CONCEPT 9: Agent with Tools{Style.RESET_ALL}")
        print("-" * 50)

        try:
            result = self.agent.invoke({"input": task})
            return result['output']
        except Exception as e:
            print(f"{Fore.RED}Error in agent: {e}{Style.RESET_ALL}")
            return None

    def stream_response(self, prompt: str = "Write a haiku about coding"):
        """CONCEPT 10: Streaming - Real-time token generation"""
        print(f"\n{Fore.CYAN}📡 CONCEPT 10: Streaming Response{Style.RESET_ALL}")
        print("-" * 50)

        try:
            # Create streaming LLM
            streaming_llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                temperature=0.7
            )

            print("Streaming: ", end="")
            response = streaming_llm.invoke([HumanMessage(content=prompt)])
            print("\n")
            return response
        except Exception as e:
            print(f"{Fore.RED}Error in streaming: {e}{Style.RESET_ALL}")
            return None

    def run_all_demos(self):
        """Run all demonstrations sequentially"""
        print(f"\n{Fore.MAGENTA}{'='*60}")
        print("🎯 RUNNING ALL LANGCHAIN CONCEPTS")
        print(f"{'='*60}{Style.RESET_ALL}\n")

        # Run each demonstration
        demos = [
            ("Prompt Templates", self.demonstrate_prompt_templates),
            ("Chains & Sequential", self.demonstrate_chains),
            ("Output Parsing", self.demonstrate_output_parsing),
            ("Chat with Memory", lambda: self.chat_with_memory("Hi, I'm learning LangChain")),
            ("Memory Follow-up", lambda: self.chat_with_memory("What did I just tell you?")),
            ("Knowledge Search", self.search_knowledge),
            ("Agent with Tools", lambda: self.use_agent("What's 25 * 4 and what time is it?")),
            ("Streaming Response", self.stream_response),
        ]

        for i, (name, func) in enumerate(demos, 1):
            print(f"\n{Fore.YELLOW}Demo {i}/8: {name}{Style.RESET_ALL}")
            try:
                func()
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

            if i < len(demos):
                input(f"\n{Fore.GREEN}Press Enter for next demo...{Style.RESET_ALL}")

        print(f"\n{Fore.GREEN}✨ All demonstrations complete!{Style.RESET_ALL}")


# ===== MAIN FUNCTION =====
def main():
    """Main entry point"""
    # Import math for calculator tool
    global math
    import math

    print(f"{Fore.MAGENTA}╔{'═'*58}╗")
    print(f"║{'🤖 LANGCHAIN ALL-IN-ONE DEMO 🤖'.center(58)}║")
    print(f"║{'All Concepts in a Single Program'.center(58)}║")
    print(f"╚{'═'*58}╝{Style.RESET_ALL}\n")

    # Check API key
    if OPENAI_API_KEY == "sk-your-key-here" or not OPENAI_API_KEY:
        print(f"{Fore.RED}❌ ERROR: Please update your OPENAI_API_KEY!{Style.RESET_ALL}")
        print("\nSetup Instructions:")
        print("1. Open this Python file in a text editor")
        print("2. Find the line: OPENAI_API_KEY = \"sk-your-key-here\"")
        print("3. Replace 'sk-your-key-here' with your actual OpenAI API key")
        print("4. Install packages: pip install langchain langchain-openai python-dotenv faiss-cpu tiktoken colorama pydantic")
        print("5. Run the program again")
        return

    try:
        # Initialize the assistant
        print(f"{Fore.YELLOW}Initializing AI Assistant...{Style.RESET_ALL}")
        assistant = PersonalAIAssistant()

        # Run all demonstrations
        assistant.run_all_demos()

    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Program interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
        print("\nTroubleshooting:")
        print("1. Check your OpenAI API key is valid")
        print("2. Ensure you have credits in your account")
        print("3. Check all packages are installed:")
        print("   pip install langchain langchain-openai python-dotenv faiss-cpu tiktoken colorama pydantic")


# ===== SCRIPT ENTRY POINT =====
if __name__ == "__main__":
    main()