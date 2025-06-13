from typing import List, get_type_hints, Optional, Union
from pydantic import BaseModel, Field
import asyncio
import threading
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, FunctionMessage
from langchain_core.tools import BaseTool
from langchain_core.tools import StructuredTool
from langgraph.graph.graph import CompiledGraph
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

def create_subagent_tool(
        mcp_agent: CompiledGraph,
        tool_name: str,
        tool_desc: str,
) -> BaseTool:
    # Define the input schema
    class SubAgentInput(BaseModel):
        input: str = Field(..., description="The input string to process through the sub-agent")
        context: Optional[Union[str, List[str]]] = Field(default=[], description="Optional context")

    # Define the tool function
    async def call_agent(input: str, context: Optional[Union[str, List[str]]] = None) -> str:
        content = input

        # You can optionally inject context if needed
        # TODO: context test
        if context is not None:
            if isinstance(context, (list, tuple)):
                context = ',\n'.join([v for c in context if (v := str(c).strip()) != ''])
            context = str(context).strip()
            if context != '':
                content = f'{content}\n\n<context>{context}</context>'

        # `mcp_agent` input_schema is
        # from langgraph.prebuilt.chat_agent_executor import AgentState
        agent_input = {"messages": [HumanMessage(content)]}
        agent_output = await mcp_agent.ainvoke(agent_input)
        output = None
        if isinstance(agent_output, dict) and "messages" in agent_output:
            output = []
            for msg in agent_output["messages"]:
                # TODO: AIMessage?
                if not isinstance(msg, (AIMessage, ToolMessage, FunctionMessage)):
                    continue
                content = msg.content.strip()
                if content != '' and content not in output:
                    output.append(content)
            if len(output) == 0:
                # TODO: no response? None vs error message
                # output = None
                output = f'[ERROR] No response from the agent: {tool_name}'
            else:
                output = '\n'.join(output)
        if output is None:
            output = str(agent_output)

        print(f'@@ [call_agent] {tool_name}\n'
              f' >> input={input}, context={context}\n'
              f' >> agent_input={agent_input}\n'
              f' >> agent_output={agent_output}\n'
              f' >> output={output}\n')
        return output

    # Return as structured tool
    return StructuredTool.from_function(
        name=tool_name,
        description=tool_desc,
        coroutine=call_agent,
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
    #else:
    #    lines.append(" - (No input schema found)")
    return "\n".join(lines)


@tool
def request_user_input_tool(question: str) -> str:
    """Request additional input from the user. This tool should be called when more information is needed from the user to complete the task."""
    return f"[HumanInTheLoop] {question}"

def generate_descriptions_for_tools(tools: List[BaseTool]) -> str:
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
        "- If any critical information is missing or if there is ambiguity that requires confirmation from the user (e.g. multiple possible recipients, unclear instructions), then do use `request_user_input_tool` to explicitly ask the user for the required input or clarification."
        
    )
    tool_descriptions = [generate_tool_description(tool) for tool in tools]
    return header + "\n\n" + "============== Available Tool ==============\n" +"\n\n".join(tool_descriptions)


def get_agent_client(config: dict, llm: BaseChatModel, *args, **kwargs) -> BaseTool:
    name = config["name"]
    mcp_config = config["mcp"]
    client = MultiServerMCPClient({
        name: {
            "url": mcp_config["url"],
            "transport": mcp_config.get("transport", "streamable-http")
        },
    })
    tools = asyncio.run(client.get_tools())
    tools.append(request_user_input_tool)

    desc = generate_descriptions_for_tools(tools)
    agent: CompiledGraph = create_react_agent(model=llm, tools=tools, prompt=desc)
    print(f'# agent: {name}')
    for n, t in enumerate(tools, 1):
        print(f'  tool-#{n}: {t.name}, {t.description}')
    print(f'# ==== agent desc ====\n{desc}\n========')
    return create_subagent_tool(agent, tool_name=name, tool_desc=config["description"])
