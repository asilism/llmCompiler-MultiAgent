################################################################################
# Task Fetching Unit: plan and schedule
################################################################################

import re
import time
import asyncio
import itertools

from typing import Any, Dict, Iterable, List, Union
from typing_extensions import TypedDict
from concurrent.futures import ThreadPoolExecutor, wait
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.runnables import chain as as_runnable
from langchain_core.runnables.base import Runnable
from .output_parser import Task

def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
    print(f'@@ {__file__} >> _get_observations: messages={messages}')
    # Get all previous tool responses
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            results[int(message.additional_kwargs["idx"])] = message.content
    print(f'@@ {__file__} >> _get_observations: results={results}')
    return results

class SchedulerInput(TypedDict):
    messages: List[BaseMessage]
    tasks: Iterable[Task]


async def _execute_task(task, observations, config):
    print(f'@@ {__file__} >> _execute_task: task={task}')
    print(f'@@ {__file__} >> _execute_task: observations={observations}')
    print(f'@@ {__file__} >> _execute_task: config={config}')
    tool_to_use = task["tool"]
    if isinstance(tool_to_use, str):
        return tool_to_use
    args = task["args"]
    try:
        if args is None:
            resolved_args = None
        elif isinstance(args, str):
            resolved_args = _resolve_arg(args, observations)
        elif isinstance(args, dict):
            resolved_args = {
                key: _resolve_arg(val, observations) for key, val in args.items()
            }
        else:
            # This will likely fail
            resolved_args = args
        print(f'@@ {__file__} >> _execute_task: resolved_args={resolved_args}')
    except Exception as e:
        return (
            f"ERROR"
            f" (Failed to call {tool_to_use.name} with args {args}."
            f" Args could not be resolved. Error: {repr(e)})"
        )
    try:
        # return tool_to_use.invoke(resolved_args, config)
        print('###############################')
        print(f'Tool to use: {tool_to_use.name}')
        print('###############################')
        output = await tool_to_use.ainvoke(resolved_args, config)
        print(f'@@ {__file__} >> _execute_task: output={output}')
        return output
    except Exception as e:
        return (
            f"ERROR"
            f" (Failed to call {tool_to_use.name} with args {args}."
            f" Args resolved to {resolved_args}. Error: {repr(e)})"
        )


def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):
    print(f'@@ {__file__} >> _resolve_arg: arg={arg}')
    print(f'@@ {__file__} >> _resolve_arg: observations={observations}')
    # $1 or ${1} -> 1
    _id_pattern = r"\$\{?(\d+)\}?"

    def replace_match(match):
        # If the string is ${123}, match.group(0) is ${123}, and match.group(1) is 123.

        # Return the match group, in this case the index, from the string. This is the index
        # number we get back.
        idx = int(match.group(1))

        value = observations.get(idx, match.group(0))

        # 조건에 따른 반환
        if value in (None, [], {}, ""):
            return []
        else:
            return str(value)

    # For dependencies on other tasks
    # if isinstance(arg, str):
    #     return re.sub(_id_pattern, replace_match, arg)
    # elif isinstance(arg, list):
    #     return [_resolve_arg(a, observations) for a in arg]
    # else:
    #     return str(arg)
    if arg is None:
        output = None
    elif isinstance(arg, str):
        output = re.sub(_id_pattern, replace_match, arg)
    elif isinstance(arg, list):
        output = [_resolve_arg(a, observations) for a in arg]
    else:
        output = str(arg)
    print(f'@@ {__file__} >> _resolve_arg: output={output}')
    return output


@as_runnable
async def _schedule_task(task_inputs, config):
    print(f'@@ {__file__} >> _schedule_task: task_inputs={task_inputs}')
    print(f'@@ {__file__} >> _schedule_task: config={config}')
    task: Task = task_inputs["task"]
    observations: Dict[int, Any] = task_inputs["observations"]
    try:
        observation = await _execute_task(task, observations, config)
    except Exception as e:
        import traceback

        observation = traceback.format_exception(e)  # repr(e) +
    observations[task["idx"]] = observation
    print(f'@@ {__file__} >> _schedule_task: observations={observations}')


async def _schedule_pending_task(
    task: Task, observations: Dict[int, Any], retry_after: float = 0.2
):
    print(f'@@ {__file__} >> _schedule_pending_task: task={task}')
    print(f'@@ {__file__} >> _schedule_pending_task: observations={observations}')
    print(f'@@ {__file__} >> _schedule_pending_task: retry_after={retry_after}')
    while True:
        deps = task["dependencies"]
        if deps and (any([dep not in observations for dep in deps])):
            # Dependencies not yet satisfied
            await asyncio.sleep(retry_after)
            continue
        await _schedule_task.ainvoke({"task": task, "observations": observations})
        break


@as_runnable
async def _schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
    print(f'@@ {__file__} >> _schedule_tasks: scheduler_input={scheduler_input}')
    """Group the tasks into a DAG schedule."""
    # For streaming, we are making a few simplifying assumption:
    # 1. The LLM does not create cyclic dependencies
    # 2. That the LLM will not generate tasks with future deps
    # If this ceases to be a good assumption, you can either
    # adjust to do a proper topological sort (not-stream)
    # or use a more complicated data structure
    tasks = scheduler_input["tasks"]
    args_for_tasks = {}
    messages = scheduler_input["messages"]
    # If we are re-planning, we may have calls that depend on previous
    # plans. Start with those.
    observations = _get_observations(messages)
    task_names = {}
    originals = set(observations)
    # ^^ We assume each task inserts a different key above to
    # avoid race conditions...
    retry_after = 0.2
    tasks = []
    for task in scheduler_input["tasks"]:
        deps = task["dependencies"]
        task_names[task["idx"]] = (
            task["tool"] if isinstance(task["tool"], str) else task["tool"].name
        )
        args_for_tasks[task["idx"]] = task["args"]

        if deps and any([dep not in observations for dep in deps]):
            tasks.append(asyncio.create_task(_schedule_pending_task(task, observations, retry_after)))
        else:
            tasks.append(asyncio.create_task(_schedule_task.ainvoke(dict(task=task, observations=observations))))
    await asyncio.gather(*tasks)

    # Convert observations to new tool messages to add to the state
    new_observations = {
        k: (task_names[k], args_for_tasks[k], observations[k])
        for k in sorted(observations.keys() - originals)
    }
    tool_messages = [
        FunctionMessage(
            name=name,
            content=str(obs),
            additional_kwargs={"idx": k, "args": task_args},
            tool_call_id=k,
        )
        for k, (name, task_args, obs) in new_observations.items()
    ]
    print(f'@@ {__file__} >> _schedule_tasks: tool_messages={tool_messages}')
    return tool_messages


def build(planner: Runnable) -> Runnable:
    @as_runnable
    async def plan_and_schedule(state):
        print(f'@@ {__file__} >> plan_and_schedule: state={state}')
        messages = state["messages"]
        tasks = planner.stream(messages)
        # Begin executing the planner immediately
        try:
            tasks = itertools.chain([next(tasks)], tasks)
        except StopIteration:
            # Handle the case where tasks is empty.
            tasks = iter([])
        scheduled_tasks = await _schedule_tasks.ainvoke({"messages": messages, "tasks": tasks})
        print(f'@@ {__file__} >> plan_and_schedule: scheduled_tasks={scheduled_tasks}')
        return {"messages": scheduled_tasks}

    return plan_and_schedule
