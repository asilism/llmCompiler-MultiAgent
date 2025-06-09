################################################################################
# Prompts
################################################################################

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.load.load import loads
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_from_json(path) -> ChatPromptTemplate:
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
    return loads(data)

_join: ChatPromptTemplate = load_from_json(get_project_root() / "players" / "join.json")

_plan: ChatPromptTemplate = hub.pull("wfh/llm-compiler")
#_join: ChatPromptTemplate = hub.pull("wfh/llm-compiler-joiner")

_replan: str = \
    ' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results' \
    ' (given as Observation) of each plan and a general thought (given as Thought) about the executed results.' \
    ' You MUST use these information to create the next plan under "Current Plan".\n' \
    ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n' \
    ' - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n' \
    ' - You must continue the task index from the end of the previous one. Do not repeat task indices.'

# TODO: simple memory DB
_default_key = "default"
_DATA: dict[str, dict[str, ChatPromptTemplate | str]] = {
    "default": {"plan": _plan, "replan": _replan, "join": _join},
}


class PromptManager:
    @staticmethod
    def get(key: str) -> ChatPromptTemplate | str | None:
        global _DATA
        return _DATA.get(key, _DATA[_default_key])
