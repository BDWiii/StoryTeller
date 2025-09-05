from agents import chatbot
from agents.states import _initialize_state
import uuid


# ==================== Main Entry ==========================


class RunStoryTeller:
    def __init__(self):
        self.agent = chatbot.Chatbot().chatbot
        self.thread_id = None
        self.config = {}

    def new_thread(self, Input: str):
        self.thread_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}
        state = _initialize_state(Input)
        result = self.agent.invoke(state, self.config)

        return {"session_id": self.thread_id, "story": result.get("story", "")}

    def existing_thread(self, Input: str):
        if not self.thread_id:
            raise ValueError("No existing thread_id to resume")

        snapshot = self.agent.get_state(self.config)
        state = dict(snapshot.values)
        state["task"] = Input
        state["next_node"] = ""

        result = self.agent.invoke(state, config=self.config)

        return {"session_id": self.thread_id, "story": result.get("story", "")}

    def get_current_state(self, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        return self.agent.get_state(config)


if __name__ == "__main__":
    # example usage
    
    user_input = "Talk to me about swords (katana) in japanese culture"
    runner = RunStoryTeller()

    print("Starting new thread")
    response = runner.new_thread(user_input)
    print(response)

    print("Resuming existing thread")
    response = runner.existing_thread("Tell me about traditions samurai used with their katanas")
    print(response)
