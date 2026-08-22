"""
LangChain Concepts Tutorial
===========================
This program demonstrates core LangChain concepts with practical examples.
Prerequisites: pip install langchain langchain-community python-dotenv
"""

import os
from typing import List
from dotenv import load_dotenv

# Load environment variables (store your API keys in .env file)
load_dotenv()

# ============================================
# CONCEPT 1: LLMs (Large Language Models)
# ============================================
def concept_1_llms():
    """Basic LLM usage - the foundation of LangChain"""
    from langchain_community.llms import OpenAI
    
    print("\n=== CONCEPT 1: Basic LLM Usage ===")
    
    # Initialize an LLM (you can also use other providers like Anthropic, Cohere, etc.)
    # Note: Set OPENAI_API_KEY in your environment variables
    llm = OpenAI(temperature=0.7)
    
    # Simple text generation
    prompt = "Explain quantum computing in one sentence:"
    response = llm.invoke(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    return llm


# ============================================
# CONCEPT 2: PROMPT TEMPLATES
# ============================================
def concept_2_prompt_templates():
    """Structured prompts with variables"""
    from langchain.prompts import PromptTemplate
    from langchain_community.llms import OpenAI
    
    print("\n=== CONCEPT 2: Prompt Templates ===")
    
    # Create a prompt template with variables
    template = """You are a {profession} expert.
    
    Question: {question}
    
    Please provide a detailed answer in the style of a {profession}:
    """
    
    prompt = PromptTemplate(
        input_variables=["profession", "question"],
        template=template
    )
    
    # Format the prompt with actual values
    formatted_prompt = prompt.format(
        profession="chef",
        question="What's the secret to perfect pasta?"
    )
    
    print(f"Formatted Prompt:\n{formatted_prompt}")
    
    # Use with LLM
    llm = OpenAI(temperature=0.7)
    response = llm.invoke(formatted_prompt)
    print(f"\nResponse: {response}")
    
    return prompt


# ============================================
# CONCEPT 3: CHAINS
# ============================================
def concept_3_chains():
    """Combining LLMs with prompts in a chain"""
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_community.llms import OpenAI
    
    print("\n=== CONCEPT 3: Chains ===")
    
    # Create components
    llm = OpenAI(temperature=0.7)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="Write a creative product description for: {product}"
    )
    
    # Create a chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run the chain
    result = chain.run(product="wireless headphones with AI noise cancellation")
    print(f"Chain Result: {result}")
    
    return chain


# ============================================
# CONCEPT 4: SEQUENTIAL CHAINS
# ============================================
def concept_4_sequential_chains():
    """Chaining multiple operations together"""
    from langchain.chains import LLMChain, SimpleSequentialChain
    from langchain.prompts import PromptTemplate
    from langchain_community.llms import OpenAI
    
    print("\n=== CONCEPT 4: Sequential Chains ===")
    
    llm = OpenAI(temperature=0.7)
    
    # First chain: Generate a story premise
    premise_template = PromptTemplate(
        input_variables=["genre"],
        template="Create a one-sentence premise for a {genre} story:"
    )
    premise_chain = LLMChain(llm=llm, prompt=premise_template)
    
    # Second chain: Expand the premise into a synopsis
    synopsis_template = PromptTemplate(
        input_variables=["premise"],
        template="Expand this premise into a 3-sentence synopsis: {premise}"
    )
    synopsis_chain = LLMChain(llm=llm, prompt=synopsis_template)
    
    # Combine chains
    overall_chain = SimpleSequentialChain(
        chains=[premise_chain, synopsis_chain],
        verbose=True
    )
    
    # Run the sequential chain
    result = overall_chain.run("science fiction")
    print(f"\nFinal Result: {result}")
    
    return overall_chain


# ============================================
# CONCEPT 5: MEMORY
# ============================================
def concept_5_memory():
    """Adding memory to conversations"""
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationChain
    from langchain_community.llms import OpenAI
    
    print("\n=== CONCEPT 5: Memory in Conversations ===")
    
    # Create memory object
    memory = ConversationBufferMemory()
    
    # Create conversation chain with memory
    llm = OpenAI(temperature=0.7)
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Have a conversation
    response1 = conversation.predict(input="Hi, my name is Alice and I love programming")
    print(f"Response 1: {response1}")
    
    response2 = conversation.predict(input="What's my name and what do I love?")
    print(f"Response 2: {response2}")
    
    # View conversation history
    print(f"\nMemory Contents: {memory.buffer}")
    
    return conversation


# ============================================
# CONCEPT 6: OUTPUT PARSERS
# ============================================
def concept_6_output_parsers():
    """Structuring LLM outputs"""
    from langchain.output_parsers import PydanticOutputParser
    from langchain.prompts import PromptTemplate
    from langchain_community.llms import OpenAI
    from pydantic import BaseModel, Field
    
    print("\n=== CONCEPT 6: Output Parsers ===")
    
    # Define the desired output structure
    class Recipe(BaseModel):
        name: str = Field(description="name of the recipe")
        ingredients: List[str] = Field(description="list of ingredients")
        cooking_time: int = Field(description="cooking time in minutes")
        difficulty: str = Field(description="difficulty level: easy, medium, or hard")
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=Recipe)
    
    # Create prompt with format instructions
    prompt = PromptTemplate(
        template="Generate a recipe for {dish}.\n{format_instructions}",
        input_variables=["dish"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Generate and parse
    llm = OpenAI(temperature=0.7)
    formatted_prompt = prompt.format(dish="vegetarian pasta")
    output = llm.invoke(formatted_prompt)
    
    try:
        parsed_output = parser.parse(output)
        print(f"Parsed Recipe Object:")
        print(f"  Name: {parsed_output.name}")
        print(f"  Ingredients: {parsed_output.ingredients}")
        print(f"  Cooking Time: {parsed_output.cooking_time} minutes")
        print(f"  Difficulty: {parsed_output.difficulty}")
    except Exception as e:
        print(f"Parsing error: {e}")
        print(f"Raw output: {output}")
    
    return parser


# ============================================
# CONCEPT 7: AGENTS AND TOOLS
# ============================================
def concept_7_agents():
    """Agents that can use tools to accomplish tasks"""
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain.tools import Tool
    from langchain_community.llms import OpenAI
    from langchain.prompts import PromptTemplate
    
    print("\n=== CONCEPT 7: Agents and Tools ===")
    
    # Create custom tools
    def calculate(expression: str) -> str:
        """Simple calculator tool"""
        try:
            result = eval(expression)
            return str(result)
        except:
            return "Error in calculation"
    
    def get_word_count(text: str) -> str:
        """Count words in text"""
        return str(len(text.split()))
    
    tools = [
        Tool(
            name="Calculator",
            func=calculate,
            description="Useful for mathematical calculations. Input should be a mathematical expression."
        ),
        Tool(
            name="WordCounter",
            func=get_word_count,
            description="Counts the number of words in a text. Input should be a text string."
        )
    ]
    
    # Create agent prompt
    agent_prompt = PromptTemplate.from_template(
        """Answer the following question using the available tools.
        
        You have access to the following tools:
        {tools}
        
        Use the following format:
        Question: the input question
        Thought: think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (repeat Thought/Action/Action Input/Observation as needed)
        Thought: I now know the final answer
        Final Answer: the final answer
        
        Question: {input}
        Thought: {agent_scratchpad}
        """
    )
    
    # Create agent
    llm = OpenAI(temperature=0)
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=agent_prompt
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=3
    )
    
    # Test the agent
    result = agent_executor.invoke({
        "input": "What is 25 * 4, and how many words are in the sentence 'The quick brown fox jumps'?"
    })
    
    print(f"\nAgent Result: {result['output']}")
    
    return agent_executor


# ============================================
# CONCEPT 8: DOCUMENT LOADERS & VECTORSTORES
# ============================================
def concept_8_rag_basics():
    """Basic RAG (Retrieval Augmented Generation) setup"""
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain_community.llms import OpenAI
    
    print("\n=== CONCEPT 8: RAG Basics ===")
    
    # Sample documents
    texts = [
        "LangChain is a framework for developing applications powered by language models.",
        "It provides tools for prompt management, chains, agents, and memory.",
        "Vector databases store embeddings for similarity search.",
        "RAG combines retrieval with generation for better answers.",
        "Agents can use tools to interact with external systems."
    ]
    
    # Split texts into chunks
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    documents = text_splitter.create_documents(texts)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Create QA chain
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    # Ask questions
    question = "What is LangChain?"
    answer = qa_chain.run(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    return qa_chain


# ============================================
# MAIN EXECUTION
# ============================================
def main():
    """Run all concept demonstrations"""
    print("=" * 60)
    print("LANGCHAIN CONCEPTS TUTORIAL")
    print("=" * 60)
    
    # Note: Comment out concepts that require API keys if not available
    
    try:
        # Basic concepts
        concept_1_llms()
        concept_2_prompt_templates()
        concept_3_chains()
        concept_4_sequential_chains()
        
        # Advanced concepts
        concept_5_memory()
        concept_6_output_parsers()
        concept_7_agents()
        concept_8_rag_basics()
        
    except Exception as e:
        print(f"\nNote: Some examples require API keys to be set.")
        print(f"Error: {e}")
        print("\nTo run these examples:")
        print("1. Create a .env file in your project directory")
        print("2. Add your API key: OPENAI_API_KEY=your-key-here")
        print("3. Install required packages: pip install langchain langchain-community python-dotenv openai faiss-cpu")


# ============================================
# SIMPLIFIED EXAMPLES (No API Required)
# ============================================
def concept_demos_without_api():
    """Demonstrate concepts without requiring API keys"""
    print("\n" + "=" * 60)
    print("CONCEPTUAL EXAMPLES (No API Required)")
    print("=" * 60)
    
    # Example 1: Understanding Prompt Templates
    from langchain.prompts import PromptTemplate, FewShotPromptTemplate
    
    print("\n=== Prompt Template Structure ===")
    template = PromptTemplate(
        input_variables=["topic", "style"],
        template="Write about {topic} in the style of {style}"
    )
    print(f"Template: {template.template}")
    print(f"Variables: {template.input_variables}")
    print(f"Formatted: {template.format(topic='AI', style='Shakespeare')}")
    
    # Example 2: Few-Shot Prompting
    print("\n=== Few-Shot Prompt Template ===")
    examples = [
        {"word": "happy", "antonym": "sad"},
        {"word": "tall", "antonym": "short"},
    ]
    
    example_template = PromptTemplate(
        input_variables=["word", "antonym"],
        template="Word: {word}\nAntonym: {antonym}"
    )
    
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        prefix="Give the antonym of each word:\n",
        suffix="\nWord: {input}\nAntonym:",
        input_variables=["input"]
    )
    
    print(few_shot_prompt.format(input="hot"))
    
    # Example 3: Understanding Memory Types
    print("\n=== Memory Types in LangChain ===")
    print("1. ConversationBufferMemory: Stores entire conversation")
    print("2. ConversationSummaryMemory: Stores summarized conversation")
    print("3. ConversationBufferWindowMemory: Stores last K interactions")
    print("4. ConversationEntityMemory: Stores information about entities")


if __name__ == "__main__":
    # Run conceptual demos (no API needed)
    concept_demos_without_api()
    
    # Uncomment to run full examples (requires API keys)
    # main()