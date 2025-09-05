import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ChatMessage,
    AnyMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from tools.web_search import search_web
from agents import states
from prompts import prompt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===================== Story Teller Agent ==============================
class StoryTeller:
    """
    Raises awareness with entertaining, rhymey and short stories.
    have the right tone to narrate to kids and adults.
    searches the web for topics to make a story about.
    """

    def __init__(self, llm):
        self.llm = llm
        self.web_search_function = search_web

        build_story = StateGraph(states.StoryTellerState)

        build_story.add_node("search", self.search_node)
        build_story.add_node("adults", self.adults_node)
        build_story.add_node("kids", self.kids_node)

        build_story.set_entry_point("search")

        build_story.add_conditional_edges(
            "search", self.decision, {"adults": "adults", "kids": "kids"}
        )

        build_story.add_edge("adults", END)
        build_story.add_edge("kids", END)

        self.storyteller = build_story.compile()

    def search_node(self, state: states.StoryTellerState):
        messages = [
            SystemMessage(content=prompt.SEARCH_PROMPT),
            HumanMessage(content=state.get("task", "")),
        ]
        search_queries = self.llm.with_structured_output(states.Query).invoke(messages)

        search_results = []
        for q in search_queries.query:
            response = self.web_search_function.invoke(
                q, max_results=search_queries.max_results
            )
            for item in response:
                search_results.append(
                    {
                        "url": item["url"],
                        "content": item["content"],
                    }
                )

        return {
            "node_name": "search",
            "retrieved_content": search_results,
            "task": state.get("task", ""),
        }

    def adults_node(self, state: states.StoryTellerState):
        messages = [
            SystemMessage(content=prompt.STORYTELLER_PROMPT_ADULTS),
            HumanMessage(
                content=f"{state.get("task", "")} \n\n Here are the search results: \n\n {state.get('retrieved_content')}"
            ),
        ]

        response = self.llm.invoke(messages)

        return {
            "node_name": "adults",
            "task": state.get("task", ""),
            "story": response.content,
        }

    def kids_node(self, state: states.StoryTellerState):
        messages = [
            SystemMessage(content=prompt.STORYTELLER_PROMPT_KIDS),
            HumanMessage(
                content=f"{state.get("task", "")} \n\n Here are the search results: \n\n {state.get('retrieved_content')}"
            ),
        ]

        response = self.llm.invoke(messages)

        return {
            "node_name": "kids",
            "task": state.get("task", ""),
            "story": response.content,
        }

    def decision(self, state: states.StoryTellerState):
        if state.get("age", 12) < 16:
            return "kids"
        else:
            return "adults"
