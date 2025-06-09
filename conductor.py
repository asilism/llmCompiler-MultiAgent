################################################################################
# Conductor: LangGraph
################################################################################

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from players.planner import build as build_planner
from players.scheduler import build as build_scheduler
from players.joiner import build as build_joiner
from players.human_in_the_loop import build as build_hitl

from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list, add_messages]

def build(
        model: BaseChatModel,
        tools: list[BaseTool],
        prompts: dict[str, ChatPromptTemplate | str],
):
    planner: Runnable = build_planner(model, tools, prompts["plan"], prompts["replan"])
    plan_and_schedule: Runnable = build_scheduler(planner)
    join: Runnable = build_joiner(model, prompts["join"].partial(examples=''))
    human_in_the_loop_node: Runnable = build_hitl()

    # A checkpointer must be enabled for interrupts to work!
    checkpointer = MemorySaver()

    graph = StateGraph(State)

    # Define vertices
    # We defined plan_and_schedule above already.
    # Assign each node to a state variable to update.
    graph.add_node("plan_and_schedule", plan_and_schedule)
    graph.add_node("join", join)
    graph.add_node("human_in_the_loop", human_in_the_loop_node)

    # Define edges
    graph.add_edge("plan_and_schedule", "join")

    # This condition determines looping logic
    def should_continue(state):
        if isinstance(state["messages"][-1], AIMessage):
            return END
        elif isinstance(state["messages"][-1], SystemMessage) and "[HumanInTheLoop]" in state["messages"][-1].content:
            return "human_in_the_loop"
        else:
            return "plan_and_schedule"

    graph.add_conditional_edges(
        "join",
        # Next, we pass in the function that will determine which node is called next.
        should_continue)

    graph.add_edge(START, "plan_and_schedule")
    graph.add_edge("human_in_the_loop", "join")

    return graph.compile(checkpointer=checkpointer)
