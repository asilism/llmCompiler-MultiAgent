import asyncio
import time
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
import sys
import uuid

from langchain_core.messages import HumanMessage, AIMessage
from managers.llm_manager import LLM
from managers.tool_manager import ToolManager
from managers.prompt_manager import PromptManager
from conductor import build
from _demo import prepare
import agents


prepare()
# TODO : make thread_id logic by user_id
config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
    }
}

ToolManager.set(agents.get_agent_client("mcp", {
    "name": "mail_agent",
    "description": f"""
{{agent_name}}(input="input message for {{agent_name}}", context="context for {{agent_name}}") -> str:
* This is a unified interface to a multi-tool agent. It takes a natural language input, interprets the request, and uses internal MCP tools to execute the appropriate actions.
* The agent is equipped with multiple tools (e.g., list unread mails, read mail, send mail, etc.) and can autonomously choose the most suitable tool for the user's intent.
* `query` can be either a simple keyword (e.g. "latest email") or a natural language question (e.g. "when did I receive the last email from John?").
* You cannot handle multiple request in one call. For instance, `{{agent_name}}('get email from John, get email from Jane')` does not work. If you need to process multiple request, you need to call them separately like `{{agent_name}}('get email from John')` and then `{{agent_name}}('get email from Jane')`.
* Minimize the number of `mail` actions as much as possible. For instance, instead of calling 2. {{agent_name}}("what is the subject of $1") and then 3. {{agent_name}}("what is the sender of $1"), you MUST call 2. {{agent_name}}("what is the subject and sender of $1") instead, which will reduce the number of mail actions.
* You can optionally provide a list of strings as `context` to help the agent understand the query. If there are multiple contexts you need to answer the query, you can provide them as a list of strings.
* `{{agent_name}}` action will not see the output of the previous actions unless you provide it as `context`. You MUST provide the output of the previous actions as `context` if you need to refer to them.
* You MUST NEVER provide `search` type action's outputs as a variable in the `query` argument. This is because `search` returns a text blob, not a structured email object. Therefore, when you need to provide an output of `search` action, you MUST provide it as a `context` argument to `mail` action. For example, 1. search("John’s email") and then 2. {{agent_name}}("get sender of $1") is NEVER allowed. Use 2. {{agent_name}}("get sender of John’s email", context=["$1"]) instead.
* When you ask a question about `context`, specify the email fields explicitly. For instance, "what is the subject of this email?" or "who is the sender?" instead of vague questions like "what is this?"
    """.strip().format(agent_name='mail_agent'),
    "mcp": {
        "transport": "streamable_http",
        "url": "http://localhost:8002/mcp",
    },
}, LLM.get()))

# ToolManager.set(agents.get_agent_client("mcp", {
#     "name": "A25T4_calendar_agent",
#     "description": f"""
# {{agent_name}}(input="input message for {{agent_name}}", context="context for {{agent_name}}") -> str:
# * This is a unified interface to a multi-tool agent. It takes a natural language input, interprets the request, and uses internal MCP tools to execute the appropriate actions.
# * The agent is equipped with multiple tools (e.g., math, weather queries, etc.) and can autonomously choose the most suitable tool for the user's intent.
# * The `input` should be a plain English request describing what the user wants to know or compute.
# * The `context` field is optional and can include supplemental information from previous steps or system memory to improve accuracy.
# * The output is a final answer generated after the agent completes reasoning and tool execution.
# * You should not assume the agent knows everything; it only knows what its tools allow it to observe or compute.
# * Do not include multiple unrelated questions in a single input. The agent processes one task per request.
#     """.strip().format(agent_name='A25T4_calendar_agent'),
#     "mcp": {
#         "transport": "streamable_http",
#         "url": "http://localhost:8003/mcp",
#     },
# }, LLM.get()))

ToolManager.set(agents.get_agent_client("mcp", {
    "name": "weather_agent",
    "description": f"""
{{agent_name}}(input="input message for {{agent_name}}", context="context for {{agent_name}}") -> str:
* This is a unified interface to a multi-tool agent. It takes a natural language input, interprets the request, and uses internal MCP tools to execute the appropriate actions.
* The agent is equipped with multiple tools (e.g., math, weather queries, etc.) and can autonomously choose the most suitable tool for the user's intent.
* The `input` should be a plain English request describing what the user wants to know or compute.
* The `context` field is optional and can include supplemental information from previous steps or system memory to improve accuracy.
* The output is a final answer generated after the agent completes reasoning and tool execution.
* You should not assume the agent knows everything; it only knows what its tools allow it to observe or compute.
* Do not include multiple unrelated questions in a single input. The agent processes one task per request.
    """.strip().format(agent_name='weather_agent'),
    "mcp": {
        "transport": "streamable_http",
        "url": "http://localhost:8001/mcp",
    },
}, LLM.get()))

# Conductor build
start_time = time.time()
conductor = build(LLM.get(), ToolManager.data(), PromptManager.get(LLM.name()))
print(f'# Built conductor ({time.time() - start_time:.3f} seconds)')

async def generate_response(user_message: str) -> AsyncGenerator[bytes, None]:
    states = list(conductor.get_state_history(config))
    messages = []
    last_message:BaseMessage = None

    if states and states[0].values:
        last_message = states[0].values["messages"][-1]

    if isinstance(last_message, AIMessage) or last_message is None:
        print("Last message is from the AI or no messages found. Starting a new conversation.")
        config["configurable"]["thread_id"] = uuid.uuid4()
        messages = [HumanMessage(content=user_message)]
    elif isinstance(last_message, SystemMessage) and "[HumanInTheLoop]" in last_message.content:
        print("Last message indicates human input is needed. Continuing the conversation.")
        messages = states[0].values["messages"] + [HumanMessage(content=user_message)]
    else:
        print("Last message is from the user or an unexpected type. Starting a new conversation.")
        config["configurable"]["thread_id"] = uuid.uuid4()
        messages = [HumanMessage(content=user_message)]

    start_time = time.time()
    print('\n########## START ##########\n')
    n_steps = 0
    yield '<< Processing >>'
    await asyncio.sleep(0.5)
    async for step in conductor.astream({
        "user_request": user_message,
        "messages": messages,
    }, config=config):
        n_steps += 1
        step_name = list(step)[0]
        messages = step[step_name]["messages"]
        print(f'\n#### [STEP-{n_steps}-{step_name}] ####')
        for i, msg in enumerate(messages):
            print(f'# [message-{i}] {msg}')
        yield str(messages).encode('utf-8')
        await asyncio.sleep(0.5)
    yield '<< Done >>'
    print(f'\n########## DONE ({time.time() - start_time:.3f} seconds) ##########\n')


app = FastAPI()


@app.post('/test')
async def test(request: Request):
    data = await request.json()
    return StreamingResponse(generate_response(data.get("message", '')), media_type='text/plain')


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
