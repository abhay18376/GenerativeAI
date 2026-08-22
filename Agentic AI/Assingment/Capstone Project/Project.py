import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool

# Load GEMINI API key (replace with your key or set in .env)
from getpass import getpass
os.environ["GEMINI_API_KEY"] = "AIzaSyAHobTo6nnjoy_32gI7wipMxjDL5d3IufI"

# Setup Gemini LLM via CrewAI
llm = LLM(
    model="gemini/gemini-2.0-flash",  # Ensure this model is valid and accessible
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.7
)

# Define Tools
class CSVReaderTool(BaseTool):
    name: str = "CSVReader"
    description: str = "Reads and parses feedback data from CSV files"

    def _run(self, file: str) -> str:
        results = ("Read the CSV File")  # Limit to 2 results to reduce output
        return results

class FeedbackClassifierTool(BaseTool):
    name: str = "FeedbackClassifier"
    description: str = "Categorizes feedback using NLP (bug, feature request, praise, complaint, spam)"

    def _run(self, file: str) -> str:
        results = ("Classified the CSV File")  # Limit to 2 results to reduce output
        return results


class BugAnalysisTool(BaseTool):
    name: str = "BugAnalysis"
    description: str = "Extracts technical details: steps to reproduce, platform info, severity assessment"

    def _run(self, file: str) -> str:
        results = ("Bug Analysis")  # Limit to 2 results to reduce output
        return results

class FeatureExtractorTool(BaseTool):
    name: str = "FeatureExtractor"
    description: str = "Identifies feature requests and estimates user impact/demand"

    def _run(self, file: str) -> str:
        results = ("Feature Extractor")  # Limit to 2 results to reduce output
        return results

class TicketCreatorTool(BaseTool):
    name: str = "TicketCreator"
    description: str = "Generates structured tickets and logs them to output CSV files"

    def _run(self, file: str) -> str:
        results = ("Ticket Creator")  # Limit to 2 results to reduce output
        return results
             
# Define Agents

CSVReader = Agent(
    role='CSV Reader',
    goal='function as CSV Reader to read csv file',
    backstory='Reads and parses feedback data from CSV files',
    verbose=True,  # Keep agent verbose for debugging, but we'll adjust Crew verbose
    allow_delegation=False,
    llm=llm
)

FeedbackClassifier = Agent(
    role='Feedback Classifier',
    goal='Categorizes feedback using NLP',
    backstory='Categorizes feedback using NLP (bug, feature request, praise, complaint, spam)',
    verbose=True,  # Keep agent verbose for debugging, but we'll adjust Crew verbose
    allow_delegation=False,
    llm=llm
)

BugAnalyszer = Agent(
    role='Bug Analysis',
    goal='steps to reproduce, platform info, severity assessment',
    backstory='Extracts technical details: steps to reproduce, platform info, severity assessment',
    verbose=True,  # Keep agent verbose for debugging, but we'll adjust Crew verbose
    allow_delegation=False,
    llm=llm
)

FeatureExtractor = Agent(
    role='Feature ExtractorTool',
    goal='Identifies feature requests and estimates user impact/demand',
    backstory='Identifies feature requests and estimates user impact/demand',
    verbose=True,  # Keep agent verbose for debugging, but we'll adjust Crew verbose
    allow_delegation=False,
    llm=llm
)

TicketCreator = Agent(
    role='TicketCreator Tool',
    goal='Generates structured tickets and logs them to output CSV files',
    backstory='Generates structured tickets and logs them to output CSV files',
    verbose=True,  # Keep agent verbose for debugging, but we'll adjust Crew verbose
    allow_delegation=False,
    llm=llm
)

#Create Tasks

csv_tool=CSVReaderTool()
fbclassifier_tool=FeedbackClassifierTool()
buganalysis_tool=BugAnalysisTool()
featureextractor_tool=FeatureExtractorTool()
ticketcreator_tool=TicketCreatorTool()

csv_task = Task(
    description="Reads and parses feedback data from CSV files.",
    expected_output="df created after reading csv file",
    agent=CSVReader,
    tools=[csv_tool])

fbclassifier_task = Task(
    description="Categorizes feedback using NLP",
    expected_output="feedback is classified as bug,feature..",
    agent=FeedbackClassifier,
    tools=[fbclassifier_tool])

buganalysis_task = Task(
    description="steps to reproduce, platform info, severity assessment",
    expected_output="Extracts technical details: steps to reproduce, platform info, severity assessment",
    agent=BugAnalyszer,
    tools=[buganalysis_tool])

featureextractor_task = Task(
    description="Identifies feature requests",
    expected_output="Identifies feature requests and estimates user impact/demand",
    agent=FeatureExtractor,
    tools=[featureextractor_tool])


ticketcreator_task=Task(
    description="Generates structured tickets and logs them to output CSV files",
    expected_output="Generates structured tickets and logs them to output CSV files",
    agent=BugAnalyszer,
    tools=[buganalysis_tool])

#ssemble the Crew

crew = Crew(
    agents=[CSVReader,FeedbackClassifier,BugAnalyszer,FeatureExtractor,TicketCreator],
    tasks=[csv_task,fbclassifier_task,buganalysis_task,featureextractor_task,ticketcreator_task],
    verbose=False  # Set to False to reduce rich console output and avoid RecursionError
)


result = crew.kickoff()

# STEP 10: Output the result
print("\n📊 Final Stock Analysis Report:\n")
print(result)