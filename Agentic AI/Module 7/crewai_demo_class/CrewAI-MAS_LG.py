# -*- coding: utf-8 -*-
"""Building_Multi_AI_Agents_Chatbots_With_LangGraph_OpenAI.ipynb

Converted to use OpenAI API instead of Anthropic
"""

# Install required packages
# pip install langchain langgraph langchain-openai openai

import os
from google.colab import userdata

# # Access the secret
# OPENAI_API_KEY = userdata.get("OPENAI_API_KEY")

# Set it as environment variable
os.environ["OPENAI_API_KEY"] = "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"
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import BaseTool
from typing import Any

# Initialize OpenAI model instead of Anthropic
model = ChatOpenAI(model="gpt-4o", temperature=0.7)

class MultiAgentState(MessagesState):
    last_active_agent: str

def get_travel_recommendations(query: str) -> str:
    """
    This function takes a user query about travel recommendations and returns a string containing travel recommendations.

    Args:
        query: The user's query about travel recommendations.

    Returns:
        A string containing travel recommendations.
    """
    # Basic keyword-based recommendations
    if "beach" in query.lower():
        recommendations = "For a beach vacation, I recommend Hawaii, Bali, or the Maldives."
    elif "mountain" in query.lower():
        recommendations = "For a mountain getaway, consider the Swiss Alps, the Rockies, or the Himalayas."
    elif "city" in query.lower():
        recommendations = "For a city break, I suggest exploring New York, Paris, or Tokyo."
    else:
        recommendations = "I recommend visiting popular destinations like Hawaii, Bali, or Paris."  # Default recommendations

    return recommendations


def make_handoff_tool(agent_name: str) -> BaseTool:
    class HandoffTool(BaseTool):
        name: str = f"handoff_to_{agent_name}"  # Type annotation added
        description: str = f"Hand off the conversation to {agent_name}."  # Type annotation added

        def _run(self, tool_input: str,  # type: ignore
                **kwargs: Any, # Type annotation modified
            ) -> str:
            # Added an indented block here
            return f"Switching to {agent_name}"

        async def _arun(self, tool_input: str, # type: ignore
                **kwargs: Any, # Type annotation modified
            ) -> str:
            # Added an indented block here
            return f"Switching to {agent_name}"

    return HandoffTool()


travel_advisor_tools = [
    get_travel_recommendations,
    make_handoff_tool(agent_name="hotel_advisor"),
]

travel_advisor = create_react_agent(
    model,
    travel_advisor_tools,
    prompt=(
        "You are a general travel expert that can recommend travel destinations "
        "(e.g. countries, cities, etc). If you need hotel recommendations, ask 'hotel_advisor' for help. "
        "You MUST include a human-readable response before transferring to another agent."
    ),
)


def call_travel_advisor(state: MultiAgentState) -> Command:
    response = travel_advisor.invoke(state)
    update = {**response, "last_active_agent": "travel_advisor"}
    return Command(update=update, goto="human")

def get_hotel_recommendations(query: str) -> str:
    """
    This function takes a user query about hotel recommendations and returns a string containing hotel recommendations.

    Args:
        query: The user's query about hotel recommendations.

    Returns:
        A string containing hotel recommendations.
    """
    # Basic keyword-based recommendations
    if "beach" in query.lower():
        recommendations = "For beach hotels, I recommend The Ritz-Carlton, Bali or Four Seasons Maui."
    elif "city" in query.lower():
        recommendations = "For city hotels, consider The Peninsula, Hong Kong or The Savoy, London."
    else:
        recommendations = "I recommend checking out hotels like The Ritz-Carlton or Four Seasons."

    return recommendations

hotel_advisor_tools = [
    get_hotel_recommendations,
    make_handoff_tool(agent_name="travel_advisor"),
]

hotel_advisor = create_react_agent(
    model,
    hotel_advisor_tools,
    prompt=(
        "You are a hotel expert that can provide hotel recommendations for a given destination. "
        "If you need help picking travel destinations, ask 'travel_advisor' for help. "
        "You MUST include a human-readable response before transferring to another agent."
    ),
)


def call_hotel_advisor(state: MultiAgentState) -> Command:
    response = hotel_advisor.invoke(state)
    update = {**response, "last_active_agent": "hotel_advisor"}
    return Command(update=update, goto="human")

def human_node(state: MultiAgentState, config) -> Command:
    user_input = interrupt(value="Ready for user input.")
    active_agent = state["last_active_agent"]

    return Command(
        update={
            "messages": [{"role": "human", "content": user_input}]
        },
        goto=active_agent,
    )

builder = StateGraph(MultiAgentState)

builder.add_node("travel_advisor", call_travel_advisor)
builder.add_node("hotel_advisor", call_hotel_advisor)
builder.add_node("human", human_node)

builder.add_edge(START, "travel_advisor")  # Initial entry point
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

import uuid

thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

inputs = [
    {"messages": [{"role": "user", "content": "i wanna go somewhere warm in the caribbean"}]},
    Command(resume="could you recommend a nice hotel in one of the areas and tell me which area it is."),
    Command(resume="i like the first one. could you recommend something to do near the hotel?"),
]

for idx, user_input in enumerate(inputs):
    print(f"\n--- Conversation Turn {idx + 1} ---\n")
    print(f"User: {user_input}\n")
    for update in graph.stream(user_input, config=thread_config, stream_mode="updates"):
        for node_id, value in update.items():
            if isinstance(value, dict) and value.get("messages", []):
                last_message = value["messages"][-1]
                if isinstance(last_message, dict) or last_message.type != "ai":
                    continue
                print(f"{node_id}: {last_message.content}")
