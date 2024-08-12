import asyncio
import json

from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
)
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from rich.panel import Panel

from config import *
from graph import eng_graph
from tools import encode_image_to_base64, get_content


async def get_user_input(prompt="You: "):
    style = Style.from_dict(
        {
            "prompt": "cyan bold",
        }
    )
    session = PromptSession(style=style)
    return await session.prompt_async(prompt, multiline=False)


import datetime
from typing import Optional


def update_system_prompt(
    current_iteration: Optional[int] = None, max_iterations: Optional[int] = None
) -> str:
    global file_contents
    chain_of_thought_prompt = """
    Answer the user's request using relevant tools (if they are available). Before calling a tool, do some analysis within <thinking></thinking> tags. First, think about which of the provided tools is the relevant tool to answer the user's request. Second, go through each of the required parameters of the relevant tool and determine if the user has directly provided or given enough information to infer a value. When deciding if the parameter can be inferred, carefully consider all the context to see if it supports a specific value. If all of the required parameters are present or can be reasonably inferred, close the thinking tag and proceed with the tool call. BUT, if one of the values for a required parameter is missing, DO NOT invoke the function (not even with fillers for the missing params) and instead, ask the user to provide the missing parameters. DO NOT ask for more information on optional parameters if it is not provided.

    Do not reflect on the quality of the returned search results in your response.
    """

    file_contents_prompt = "\n\nFile Contents:\n"
    for path, content in file_contents.items():
        file_contents_prompt += f"\n--- {path} ---\n{content}\n"

    if automode:
        iteration_info = ""
        if current_iteration is not None and max_iterations is not None:
            iteration_info = f"You are currently on iteration {current_iteration} out of {max_iterations} in automode."
        return (
            BASE_SYSTEM_PROMPT
            + file_contents_prompt
            + "\n\n"
            + AUTOMODE_SYSTEM_PROMPT.format(iteration_info=iteration_info)
            + "\n\n"
            + chain_of_thought_prompt
        )
    else:
        return (
            BASE_SYSTEM_PROMPT + file_contents_prompt + "\n\n" + chain_of_thought_prompt
        )


def save_chat():
    # Generate filename
    now = datetime.datetime.now()
    filename = f"Chat_{now.strftime('%H%M')}"
    with open(f"{filename}.json", "w") as f:
        f.write(dump_message(conversation_history))
    filename += ".md"

    # Format conversation history
    formatted_chat = "# TurnQuest AI Engineer Chat Log\n\n"
    for message in conversation_history:
        if message.type == "human":
            formatted_chat += f"## User\n\n{message.content}\n\n"
        elif message.type == "ai":
            content = get_content(message)
            formatted_chat += f"## AI Engineer\n\n{content}\n\n"
            for t in message.tool_calls:
                formatted_chat += f"### Tool Use: \nid: `{t['id']}`\n{t['name']}\n\n```json\n{json.dumps(t['args'], indent=2)}\n```\n\n"
        elif message.type == "tool":
            formatted_chat += f"### Tool Result\nid:`{message.tool_call_id}`)\n\n```\n{message.content}\n```\n\n"

    # Save to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(formatted_chat)

    return filename


async def chat_with_claude(
    user_input, image_path=None, current_iteration=None, max_iterations=None
):
    global conversation_history, automode, main_model_tokens

    # This function uses MAINMODEL, which maintains context across calls
    current_conversation = []
    exit_continuation = False

    if image_path:
        console.print(
            Panel(
                f"Processing image at path: {image_path}",
                title_align="left",
                title="Image Processing",
                expand=False,
                style="yellow",
            )
        )
        image_base64 = encode_image_to_base64(image_path)

        if image_base64.startswith("Error"):
            console.print(
                Panel(
                    f"Error encoding image: {image_base64}",
                    title="Error",
                    style="bold red",
                )
            )
            return (
                "I'm sorry, there was an error processing the image. Please try again.",
                False,
            )
        image_message = HumanMessage(
            content=[
                {"type": "text", "text": f"User input for image: {user_input}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ]
        )
        current_conversation.append(image_message)
        console.print(
            Panel(
                "Image message added to conversation history",
                title_align="left",
                title="Image Added",
                style="green",
            )
        )
    else:
        current_conversation.append(HumanMessage(user_input))

    # Combine filtered history with current conversation to maintain context
    messages = conversation_history + current_conversation
    current_prompt = update_system_prompt(current_iteration, max_iterations)
    eng = eng_graph
    try:
        # MAINMODEL call, which maintains context
        response = await eng.ainvoke(
            input={
                "messages": messages,
                "system_message": current_prompt,
                "token_usage": main_model_tokens,
            },
            config={"recursion_limit": 100},
        )
        # Update token usage for MAINMODEL
        main_model_tokens = response["token_usage"]
        messages = response["messages"]
        last_message = messages[-1]
    except Exception as e:
        with open("main_model_error_log.txt", "a") as f:
            f.write(f"# {datetime.datetime.now()}\n\n")
            f.write(dump_message(messages))
        console.print(
            Panel(f"API Error: {str(e)}", title="API Error", style="bold red")
        )
        return (
            "I'm sorry, there was an error communicating with the AI. Please try again.",
            False,
        )

    if CONTINUATION_EXIT_PHRASE in last_message.content:
        exit_continuation = True

    console.print(
        Panel(
            f"Workflow total messages: {len(messages)}",
            title="Response Completed",
            title_align="left",
            border_style="blue",
            expand=False,
        )
    )

    # Display files in context
    if file_contents:
        files_in_context = "\n".join(file_contents.keys())
    else:
        files_in_context = "No files in context. Read, create, or edit files to add."
    console.print(
        Panel(
            files_in_context,
            title="Files in Context",
            title_align="left",
            border_style="white",
            expand=False,
        )
    )

    conversation_history = messages

    # Display token usage at the end
    display_token_usage()

    return last_message.content, exit_continuation


def reset_code_editor_memory():
    global code_editor_memory
    code_editor_memory = []
    console.print(
        Panel("Code editor memory has been reset.", title="Reset", style="bold green")
    )


def reset_conversation():
    global \
        conversation_history, \
        main_model_tokens, \
        tool_checker_tokens, \
        code_editor_tokens, \
        code_execution_tokens, \
        file_contents, \
        code_editor_files
    conversation_history = []
    main_model_tokens = {"input": 0, "output": 0}
    tool_checker_tokens = {"input": 0, "output": 0}
    code_editor_tokens = {"input": 0, "output": 0}
    code_execution_tokens = {"input": 0, "output": 0}
    file_contents = {}
    code_editor_files = set()
    reset_code_editor_memory()
    console.print(
        Panel(
            "Conversation history, token counts, file contents, code editor memory, and code editor files have been reset.",
            title="Reset",
            style="bold green",
        )
    )
    display_token_usage()


def display_token_usage():
    from rich.table import Table
    from rich.box import ROUNDED

    table = Table(box=ROUNDED)
    table.add_column("Model", style="cyan")
    table.add_column("Input", style="magenta")
    table.add_column("Output", style="magenta")
    table.add_column("Total", style="green")
    table.add_column(f"% of Context ({MAX_CONTEXT_TOKENS:,})", style="yellow")
    table.add_column("Cost ($)", style="red")

    model_costs = {
        "Main Model": {"input": 3.00, "output": 15.00, "has_context": True},
        "Tool Checker": {"input": 3.00, "output": 15.00, "has_context": False},
        "Code Editor": {"input": 3.00, "output": 15.00, "has_context": True},
        "Code Execution": {"input": 3.00, "output": 15.00, "has_context": False},
    }

    total_input = 0
    total_output = 0
    total_cost = 0
    total_context_tokens = 0

    for model, tokens in [
        ("Main Model", main_model_tokens),
        ("Tool Checker", tool_checker_tokens),
        ("Code Editor", code_editor_tokens),
        ("Code Execution", code_execution_tokens),
    ]:
        input_tokens = tokens["input"]
        output_tokens = tokens["output"]
        total_tokens = input_tokens + output_tokens

        total_input += input_tokens
        total_output += output_tokens

        input_cost = (input_tokens / 1_000_000) * model_costs[model]["input"]
        output_cost = (output_tokens / 1_000_000) * model_costs[model]["output"]
        model_cost = input_cost + output_cost
        total_cost += model_cost

        if model_costs[model]["has_context"]:
            total_context_tokens += total_tokens
            percentage = (total_tokens / MAX_CONTEXT_TOKENS) * 100
        else:
            percentage = 0

        table.add_row(
            model,
            f"{input_tokens:,}",
            f"{output_tokens:,}",
            f"{total_tokens:,}",
            f"{percentage:.2f}%"
            if model_costs[model]["has_context"]
            else "Doesn't save context",
            f"${model_cost:.3f}",
        )

    grand_total = total_input + total_output
    total_percentage = (total_context_tokens / MAX_CONTEXT_TOKENS) * 100

    table.add_row(
        "Total",
        f"{total_input:,}",
        f"{total_output:,}",
        f"{grand_total:,}",
        "",  # Empty string for the "% of Context" column
        f"${total_cost:.3f}",
        style="bold",
    )

    console.print(table)


async def main():
    global automode, conversation_history
    console.print(
        Panel(
            "Welcome to the Turnkey AI Software Engineer with Multi-Agent and Image Support!",
            title="Welcome",
            style="bold green",
        )
    )
    console.print("Type 'exit' to end the conversation.")
    console.print("Type 'image' to include an image in your message.")
    console.print(
        "Type 'automode [number]' to enter Autonomous mode with a specific number of iterations."
    )
    console.print("Type 'reset' to clear the conversation history.")
    console.print("Type 'save chat' to save the conversation to a Markdown file.")
    console.print(
        "While in automode, press Ctrl+C at any time to exit the automode to return to regular chat."
    )

    while True:
        user_input = await get_user_input()

        if user_input.lower() == "exit":
            console.print(
                Panel(
                    "Thank you for chatting. Goodbye!",
                    title_align="left",
                    title="Goodbye",
                    style="bold green",
                )
            )
            break

        if user_input.lower() == "reset":
            reset_conversation()
            continue

        if user_input.lower() == "save chat":
            filename = save_chat()
            console.print(
                Panel(
                    f"Chat saved to {filename}", title="Chat Saved", style="bold green"
                )
            )
            continue

        if user_input.lower() == "image":
            image_path = (
                (
                    await get_user_input(
                        "Drag and drop your image here, then press enter: "
                    )
                )
                .strip()
                .replace("'", "")
            )

            if os.path.isfile(image_path):
                user_input = await get_user_input("You (prompt for image): ")
                response, _ = await chat_with_claude(user_input, image_path)
            else:
                console.print(
                    Panel(
                        "Invalid image path. Please try again.",
                        title="Error",
                        style="bold red",
                    )
                )
                continue
        elif user_input.lower().startswith("automode"):
            try:
                parts = user_input.split()
                if len(parts) > 1 and parts[1].isdigit():
                    max_iterations = int(parts[1])
                else:
                    max_iterations = MAX_CONTINUATION_ITERATIONS

                automode = True
                console.print(
                    Panel(
                        f"Entering automode with {max_iterations} iterations. Please provide the goal of the automode.",
                        title_align="left",
                        title="Automode",
                        style="bold yellow",
                    )
                )
                console.print(
                    Panel(
                        "Press Ctrl+C at any time to exit the automode loop.",
                        style="bold yellow",
                    )
                )
                user_input = await get_user_input()

                iteration_count = 0
                try:
                    while automode and iteration_count < max_iterations:
                        response, exit_continuation = await chat_with_claude(
                            user_input,
                            current_iteration=iteration_count + 1,
                            max_iterations=max_iterations,
                        )

                        if exit_continuation or CONTINUATION_EXIT_PHRASE in response:
                            console.print(
                                Panel(
                                    "Automode completed.",
                                    title_align="left",
                                    title="Automode",
                                    style="green",
                                )
                            )
                            automode = False
                        else:
                            console.print(
                                Panel(
                                    f"Continuation iteration {iteration_count + 1} completed. Press Ctrl+C to exit automode. ",
                                    title_align="left",
                                    title="Automode",
                                    style="yellow",
                                )
                            )
                            user_input = "Continue with the next step. Or STOP by saying 'AUTOMODE_COMPLETE' if you think you've achieved the results established in the original request."
                        iteration_count += 1

                        if iteration_count >= max_iterations:
                            console.print(
                                Panel(
                                    "Max iterations reached. Exiting automode.",
                                    title_align="left",
                                    title="Automode",
                                    style="bold red",
                                )
                            )
                            automode = False
                except KeyboardInterrupt:
                    console.print(
                        Panel(
                            "\nAutomode interrupted by user. Exiting automode.",
                            title_align="left",
                            title="Automode",
                            style="bold red",
                        )
                    )
                    automode = False
                    if (
                        conversation_history
                        and conversation_history[-1]["role"] == "user"
                    ):
                        conversation_history.append(
                            {
                                "role": "assistant",
                                "content": "Automode interrupted. How can I assist you further?",
                            }
                        )
            except KeyboardInterrupt:
                console.print(
                    Panel(
                        "\nAutomode interrupted by user. Exiting automode.",
                        title_align="left",
                        title="Automode",
                        style="bold red",
                    )
                )
                automode = False
                if conversation_history and conversation_history[-1]["role"] == "user":
                    conversation_history.append(
                        {
                            "role": "assistant",
                            "content": "Automode interrupted. How can I assist you further?",
                        }
                    )

            console.print(
                Panel("Exited automode. Returning to regular chat.", style="green")
            )
        else:
            response, _ = await chat_with_claude(user_input)


def dump_message(message: list):
    msgs = []
    for m in message:
        if isinstance(m, BaseMessage):
            msgs.append(m.dict())
        else:
            msgs.append(dict(m))
    return json.dumps(msgs)


if __name__ == "__main__":
    asyncio.run(main())
