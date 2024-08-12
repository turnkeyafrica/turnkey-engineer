import json
from typing import TypedDict, Annotated, Optional, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AnyMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END
from langgraph.graph import add_messages, StateGraph
from langgraph.pregel import RetryPolicy
from rich.markdown import Markdown
from rich.panel import Panel

from config import console, DEFAULT_MODEL_CONFIG, MAIN_MODEL_CONFIG
from tools import invoke_tool, get_content, tools

main_llm = init_chat_model(**{**DEFAULT_MODEL_CONFIG, **MAIN_MODEL_CONFIG}).bind_tools(
    tools
)


class State(TypedDict):
    system_message: str
    messages: Annotated[list[AnyMessage], add_messages]
    token_usage: Optional[dict]


def get_graph(
    *,
    llm: LanguageModelLike,
    checkpoint: Optional[BaseCheckpointSaver] = None,
):
    def should_continue(state):
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        return "continue"

    def agent(state):
        token_usage = state["token_usage"]
        system_message = state["system_message"]
        response = llm.invoke(
            [SystemMessage(content=system_message)] + state["messages"]
        )
        token_usage["input"] += response.usage_metadata["input_tokens"]
        token_usage["output"] += response.usage_metadata["output_tokens"]
        console.print(
            Panel(
                Markdown(get_content(response)),
                title="Assistant's Response",
                title_align="left",
                border_style="blue",
                expand=False,
            )
        )
        return {"messages": response, "token_usage": token_usage}

    async def call_tool(state):
        tool_messages = []
        messages = state["messages"]
        last_message = cast(AIMessage, messages[-1])
        for tool_use in last_message.tool_calls:
            tool_name = tool_use["name"]
            tool_input = tool_use["args"]
            tool_use_id = tool_use["id"]
            console.print(Panel(f"Tool Used: {tool_name}", style="green"))
            console.print(
                Panel(f"Tool Input: {json.dumps(tool_input, indent=2)}", style="green")
            )
            tool_result = await invoke_tool(tool_name, tool_input)
            if tool_result["is_error"]:
                console.print(
                    Panel(
                        tool_result["content"],
                        title="Tool Execution Error",
                        style="bold red",
                    )
                )
            else:
                console.print(
                    Panel(
                        tool_result["content"],
                        title_align="left",
                        title="Tool Result",
                        style="green",
                    )
                )
            tool_messages.append(
                ToolMessage(
                    content=tool_result["content"],
                    tool_call_id=tool_use["id"],
                    name=tool_use["name"],
                )
            )
        return {"messages": tool_messages}

    workflow = StateGraph(State)
    # Define the two nodes we will cycle between
    workflow.add_node("agent", agent, retry=RetryPolicy(max_attempts=5))
    workflow.add_node("action", call_tool)
    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")
    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "action",
            # Otherwise we finish.
            "end": END,
        },
    )
    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("action", "agent")
    return workflow.compile(checkpointer=checkpoint)


eng_graph = get_graph(llm=main_llm)
