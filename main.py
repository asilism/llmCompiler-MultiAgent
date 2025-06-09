import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from managers.llm_manager import LLM
from managers.tool_manager import ToolManager
from managers.prompt_manager import PromptManager
from conductor import build
from _demo import prepare
import uuid
from langgraph.checkpoint.memory import MemorySaver

prepare()
conductor = build(LLM.get(), ToolManager.list(), PromptManager.get(LLM.name()))

config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
    }
}

async def generate_response(user_message: str) -> AsyncGenerator[bytes, None]:
    states = list(conductor.get_state_history(config))
    messages = []
    last_message:BaseMessage = None

    # for state in states:
    #     print('#######################')
    #     print(state.values)

    if states and states[0].values:
        # print('#### lastest state . values ####')
        # print(states[0].values)
        # print('#### lastest state . messages ####')
        # print(states[0].values["messages"])
        # print('#### lastest state . messages[-1] ####')
        # print(states[0].values["messages"][-1])
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

    print('\n########## START ##########\n')
    yield '[Processing]'
    await asyncio.sleep(0.5)
    for step in conductor.stream({"messages": messages}, config=config):
        print('========== <STEP> ==========')
        print(step)
        print('============================')
        yield str(step).encode('utf-8')
        await asyncio.sleep(0.5)
    yield '[Done]'
    print('\n########## DONE ##########\n')


app = FastAPI()


@app.post('/test')
async def test(request: Request):
    data = await request.json()
    return StreamingResponse(generate_response(data.get("message", '')), media_type='text/plain')


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3000)
