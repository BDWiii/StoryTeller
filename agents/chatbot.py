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

from tools import web_search
from agents import states, storyteller_agent
from prompts import prompt
import yaml


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ===================== Chatbot ==============================


class Chatbot:
    def __init__(self):
        self.llm = ChatOllama(model=config["model"])

        self.storyteller_agent = storyteller_agent.StoryTeller(self.llm).storyteller

        build_chatbot = StateGraph(states.ChatState)

        build_chatbot.add_node("router", self.router_node)
        build_chatbot.add_node("chat", self.chat_node)
        build_chatbot.add_node("storyteller", self.storyteller_agent)

        build_chatbot.add_edge("storyteller", END)
        build_chatbot.add_edge("chat", END)

        build_chatbot.set_entry_point("router")
        build_chatbot.add_conditional_edges(
            "router",
            lambda state: state.get("next_node", ""),
            {
                "chat": "chat",
                "storyteller": "storyteller",
            },
        )

        conn = sqlite3.connect(
            "checkpoints/checkpoints.sqlite", check_same_thread=False
        )
        memory = SqliteSaver(conn)
        compile_kwargs = {"checkpointer": memory}

        self.chatbot = build_chatbot.compile(**compile_kwargs)

    def router_node(self, state: states.ChatState):
        messages = [
            SystemMessage(content=prompt.ROUTER_PROMPT),
            HumanMessage(
                content=f"{state.get("task", "")} \n\n Retrieved content: \n\n {state.get('retrieved_content', '')}"
            ),
        ]

        response = self.llm.with_structured_output(states.Router).invoke(messages)

        return {
            "task": state.get("task", ""),
            "next_node": response.next_node,
        }

    def chat_node(self, state: states.ChatState):
        # Pull last 5 messages
        history = state.get("history", [])[-5:]

        messages = [
            SystemMessage(content=prompt.CHAT_PROMPT),
            *[
                (
                    HumanMessage(content=m["content"])
                    if m["role"] == "user"
                    else AIMessage(content=m["content"])
                )
                for m in history
            ],
            HumanMessage(content=f'{state.get("task", "")}'),
        ]

        response = self.llm.invoke(messages)

        # Update history
        new_history = state.get("history", [])
        new_history.append({"role": "user", "content": state["task"]})
        new_history.append({"role": "assistant", "content": response.content})

        return {
            "node_name": "chat",
            "task": state.get("task", ""),
            "story": response.content,
            "history": new_history,
        }

    def storyteller_agent(self, state: states.ChatState):
        story_state = state["story_state"]
        story_state["task"] = state["task"]

        output = self.storyteller_agent.invoke(story_state)

        return {
            "node_name": "storyteller_agent",
            "story_state": output,
            # "story": output.get("story", ""),
            "retrieved_content": output.get("retrieved_content", []),
        }
