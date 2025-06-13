################################################################################
# Joiner
################################################################################

from typing import Union
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


class FinalResponse(BaseModel):
    """The final response/answer."""

    response: str


class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )

class HumanInTheLoop(BaseModel):
    question: str = Field(
        description="Additional input from the user to clarify or complete the information required for successful task execution."
    )

class JoinOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response."""

    thought: str = Field(description="The chain of thought reasoning for the selected action")
    action: Union[FinalResponse, Replan, HumanInTheLoop]


def _parse_joiner_output(decision: JoinOutputs):  # -> List[BaseMessage]:
    print("#################JOINER OUTPUT#########################")
    response = [AIMessage(content=f"Thought: {decision.thought}")]
    if isinstance(decision.action, Replan):
        print("################# REPLAN ###############")
        messages = response + [SystemMessage(content=f"[Replan] Context from last attempt: {decision.action.feedback}")]
    elif isinstance(decision.action, HumanInTheLoop):
        print("################# HUMAN IN THE LOOP ###############")
        messages = response + [SystemMessage(content=f"[HumanInTheLoop] Context from last attempt: {decision.action.question}")]
    else:
        print("################# FINAL RESPONSE ###############")
        messages = response + [AIMessage(content=decision.action.response)]
    return {"messages": messages}

def _select_recent_messages(state) -> dict:
    # messages = state["messages"]
    # selected = []
    # for msg in messages[::-1]:
    #     selected.append(msg)
    #     if isinstance(msg, HumanMessage):
    #         break
    # return {"messages": selected[::-1]}
    return {"messages": state["messages"]}


def build(
        model: BaseChatModel,
        prompt_template: ChatPromptTemplate,
) -> Runnable:
    _runnable = prompt_template | model.with_structured_output(JoinOutputs, method="function_calling")
    return _select_recent_messages | _runnable | _parse_joiner_output
