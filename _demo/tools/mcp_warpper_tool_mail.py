from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import asyncio
import threading
import json

from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.tools import StructuredTool
from typing import List

from langchain_core.tools import BaseTool
from typing import List, get_type_hints, Optional

from managers.llm_manager import LLM
from langchain.schema.messages import ToolMessage
from langgraph.prebuilt.chat_agent_executor import AgentState

# This function runs an async coroutine
def async_to_sync_safe(coro):
    result = None
    error = None

    def runner():
        nonlocal result, error
        try:
            result = asyncio.run(coro)
        except Exception as e:
            error = e

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()

    if error:
        raise error
    return result

def create_subagent_tool(mcp_agent, tool_name: str = "subagent", desc: str = None) -> StructuredTool:
    # Define the input schema
    class SubAgentInput(BaseModel):
        input: str = Field(..., description="The input string to process through the sub-agent")
        context: Optional[list[str]] = Field(default=[], description="Optional context")

    class CustomState(AgentState):
        user_name: str

    # Define the tool function
    def call_agent(input: str, context: Optional[list[str]] = []) -> str:
        # You can optionally inject context if needed
        agent_input = {
            "messages": [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": ",".join(context) if context and isinstance(context, list) and any(context) else str(context)
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": input
                    }]
                }
            ]
        }

        #agent_input = {"input": input}
        # if context:
        #     config={"configurable": {"context": context}}
        # else:
        #     config = {}
        print("#############################################")
        print(f"Calling [{tool_name}] agent with input: {input} and context: {context}")
        print("#############################################")
        output = async_to_sync_safe(mcp_agent.ainvoke(agent_input))
        
        # ToolMessage의 content만 추출
        content = [
            message.content
            for message in output['messages']
            if isinstance(message, ToolMessage)
        ]
        
        try:
            result = json.loads(content[-1])
        except json.JSONDecodeError:
            result = content
        
        if isinstance(result, list) and len(result) == 1:
            result = result[0]

        print("#############################################")
        print(f"Output from [{tool_name}] agent: {result}")
        print("#############################################")
        
        return result

    AGENT_TOOL_DESCRIPTION = (
        "mail_agent(input: str, context: Optional[list[str]]) -> str\n"
        "- This is a unified interface to a multi-tool agent. It takes a natural language input, interprets the request, and uses internal MCP tools to execute the appropriate actions.\n"
        "- The agent is equipped with multiple tools about mail service and can autonomously choose the most suitable tool for the user's intent.\n"
        "- The `input` should be a plain English request describing what the user wants to know or compute.\n"
        "- The `context` includes supplemental information from previous steps or system memory to improve accuracy.\n"
        "- If any required information is missing or unclear, the agent must rely only on the provided `context`. It must not access or explore other emails to guess missing data.\n"
        "- The output is a final answer generated after the agent completes reasoning and tool execution.\n"
        "- You should not assume the agent knows everything; it only knows what its tools allow it to observe or compute.\n"
        "- To send an email, a valid and correctly formatted email address is required. If unknown, use the contact_agent to retrieve the correct email address.\n"
        "- Do not include multiple unrelated questions in a single input. The agent processes one task per request.\n"
        "- Do not try to fabricate or guess email content.\n"
    )

    # Return as structured tool
    return StructuredTool.from_function(
        name=tool_name,
        description=f"{AGENT_TOOL_DESCRIPTION}",
        func=call_agent,
        args_schema=SubAgentInput,
    )

def generate_tool_description(tool: StructuredTool) -> str:
    print(f"Generating description for tool: {tool.name}")
    
    sig = tool.args_schema if tool.args_schema else tool.func.__annotations__
    type_hints = get_type_hints(tool.func) if tool.func else get_type_hints(tool.coroutine)
    return_type = type_hints.get("return", "Unknown")
    doc = tool.description.strip() if tool.description else ""
    func_name = tool.name

    lines = []
    lines.append(f"{func_name}(...) -> {return_type.__name__ if hasattr(return_type, '__name__') else str(return_type)}:")
    lines.append(f" - {doc if doc else 'Performs the function defined by this tool.'}")

    # 인자 설명
    if hasattr(sig, "__annotations__"):
        for name, typ in sig.__annotations__.items():
            typename = typ.__name__ if hasattr(typ, '__name__') else str(typ)
            lines.append(f" - `{name}`: {typename} type input.")
            if name == "context" and "list" in typename.lower():
                lines.append(" - You can optionally provide a list of strings as `context` to help the tool operate correctly.")
    else:
        lines.append(" - (No input schema found)")

    return "\n".join(lines)

def generate_descriptions_for_tools(tools: List[BaseTool]) -> List[str]:
    header = (
        "You are an agent equipped with a set of MCP tools. Use these tools to accurately fulfill user requests.\n\n"
        "Each tool has a specific function signature, input requirements, and output format. Read them carefully before selecting and invoking a tool.\n\n"
        "- Always choose the most relevant tool based on the task.\n"
        "- Strictly follow the input type and parameter names as described.\n"
        "- If `context` is provided, use it to improve the accuracy of your answer.\n"
        "- Do not fabricate tool outputs. Only return what the tool provides.\n"
        "- You MUST call this tool only once per type of weather data. For example, you cannot call `get_weather('Seoul', 'temperature, precipitation')`. "
        "Instead, call `get_weather('Seoul', 'temperature')` and then `get_weather('Seoul', 'precipitation')` separately.\n"
        "- Minimize the number of `get_weather` calls by grouping what you need logically. For example, if all values are needed, call them individually but only once per type.\n"
        "- You can optionally provide a list of strings as `context` to clarify any ambiguity (e.g., time of day, elevation, past weather).\n"
        "- This tool does NOT retain the output of previous calls. If chaining values (e.g., using temperature in math), you MUST explicitly pass prior outputs via `context`.\n"
        "- You MUST NEVER treat `search`-type tool outputs as inputs for `get_weather`. If needed, extract values or use them in `context` only.\n"
        "- Always specify the units you expect when asking about weather. For example, ask 'what is the temperature in Celsius' instead of just 'what is the temperature'.\n"
    )
    tool_descriptions = [generate_tool_description(tool) for tool in tools]
    return header + "\n\n" + "\n\n".join(tool_descriptions)

def get_mail_agent_tool() -> StructuredTool:
    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "streamable_http",
                "url": "http://localhost:8002/mcp"
            },
        }
    )

    tools = asyncio.run(client.get_tools())
    desc = generate_descriptions_for_tools(tools)

    agent = create_react_agent(model=LLM.get(), tools=tools, prompt=desc)

    # StructuredTool로 wrapping
    mail_agent_tool = create_subagent_tool(agent, tool_name="mail_agent")

    return mail_agent_tool