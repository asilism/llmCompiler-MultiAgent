from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain as as_runnable
from langchain_core.runnables.base import Runnable
from langgraph.types import interrupt

def build() -> Runnable:

    @as_runnable
    def human_in_the_loop_node(state: dict) -> dict:
        """This node is called when the model needs human input."""

        answer = interrupt("Please provide missing information.")
        print(f"> Received an input from the interrupt: {answer}")
        return {
            "messages": state["messages"] + [HumanMessage(content=answer)]
        }

    return human_in_the_loop_node