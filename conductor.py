################################################################################
# Conductor: LangGraph
################################################################################

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from players.preplanner import build as build_preplanner
from players.planner import build as build_planner
from players.scheduler import build as build_scheduler
from players.joiner import build as build_joiner
from players.human_in_the_loop import build as build_hitl

from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    user_request: str
    messages: Annotated[list, add_messages]


def build(
        model: BaseChatModel,
        tools: dict[str, BaseTool],
        prompts: dict[str, ChatPromptTemplate],
):
    preplan: Runnable = build_preplanner(model)  # TODO: prompt
    planner: Runnable = build_planner(model, tools, prompts["plan"], prompts["replan"])
    plan_and_execute: Runnable = build_scheduler(planner, model)
    join: Runnable = build_joiner(model, prompts["join"].partial(examples=''))
    human_in_the_loop: Runnable = build_hitl()

    # A checkpointer must be enabled for interrupts to work!
    checkpointer = MemorySaver()

    graph = StateGraph(State)

    # Define vertices
    preplanner_node = "preplan"
    planner_executor_node = "plan_and_execute"
    joiner_node = "join"
    human_in_the_loop_node = "human_in_the_loop"

    graph.add_node(preplanner_node, preplan)
    graph.add_node(planner_executor_node, plan_and_execute)
    graph.add_node(joiner_node, join)
    graph.add_node(human_in_the_loop_node, human_in_the_loop)

    # Define edges
    graph.add_edge(preplanner_node, planner_executor_node)
    graph.add_edge(planner_executor_node, joiner_node)

    # This condition determines looping logic
    def should_continue(state):
        print("[should_continue] #########################################################")
        last_msg = state["messages"][-1]
        print("type:", type(last_msg))
        print("repr:", repr(last_msg))
        print(last_msg)
        print(last_msg.content)
        print("[should_continue] #########################################################")
        if isinstance(last_msg, AIMessage):
            return END
        elif "[HumanInTheLoop]" in last_msg.content:
            return human_in_the_loop_node
        else:
            return planner_executor_node

    # Next, we pass in the function that will determine which node is called next.
    graph.add_conditional_edges(joiner_node, should_continue)
    graph.add_edge(START, preplanner_node)
    graph.add_edge(human_in_the_loop_node, joiner_node)

    return graph.compile(checkpointer=checkpointer)
