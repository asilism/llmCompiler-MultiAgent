################################################################################
# Prompts
################################################################################

import os
from langchain_core.load.load import loads
from langchain_core.prompts import ChatPromptTemplate


def load_from_json(data: str | dict | list) -> ChatPromptTemplate:
    import json
    if isinstance(data, str) and os.path.isfile(data):
        with open(data, 'r', encoding='utf-8') as f:
            json_obj = json.load(f)
    elif isinstance(data, str):
        json_obj = json.loads(data)
    else:
        json_obj = data
    json_str = json.dumps(json_obj, ensure_ascii=True)
    return loads(json_str)


plan_prompt_template_json = os.path.abspath(os.path.join(os.path.dirname(__file__), 'plan.json'))
join_prompt_template_json = os.path.abspath(os.path.join(os.path.dirname(__file__), 'join-hitl.json'))
_plan: ChatPromptTemplate = load_from_json(plan_prompt_template_json)
_join: ChatPromptTemplate = load_from_json(join_prompt_template_json)

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
