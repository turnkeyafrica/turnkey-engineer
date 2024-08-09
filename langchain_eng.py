import asyncio
import base64
import difflib
import io
import json
import os
import re
import time

from PIL import Image
from anthropic import APIStatusError, APIError
from dotenv import load_dotenv
from google.api_core.exceptions import GoogleAPIError
from jira import JIRA, JIRAError
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.tools import tool
from langchain_google_genai._function_utils import tool_to_dict, convert_to_genai_function_declarations
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from tavily import TavilyClient

APIStatusError = (APIStatusError, GoogleAPIError)
APIError = (APIError, GoogleAPIError)
async def get_user_input(prompt="You: "):
    style = Style.from_dict(
        {
            "prompt": "cyan bold",
        }
    )
    session = PromptSession(style=style)
    return await session.prompt_async(prompt, multiline=False)


from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import datetime
import venv
import sys
import signal
import logging
from typing import Tuple, Optional, Dict, Any, List


def setup_virtual_environment() -> Tuple[str, str]:
    venv_name = "code_execution_env"
    venv_path = os.path.join(os.getcwd(), venv_name)
    try:
        if not os.path.exists(venv_path):
            venv.create(venv_path, with_pip=True)

        # Activate the virtual environment
        if sys.platform == "win32":
            activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
        else:
            activate_script = os.path.join(venv_path, "bin", "activate")

        return venv_path, activate_script
    except Exception as e:
        logging.error(f"Error setting up virtual environment: {str(e)}")
        raise


# Load environment variables from .env file
load_dotenv()

# Initialize the Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables")
tavily = TavilyClient(api_key=tavily_api_key)

console = Console()

# Token tracking variables
main_model_tokens = {"input": 0, "output": 0}
tool_checker_tokens = {"input": 0, "output": 0}
code_editor_tokens = {"input": 0, "output": 0}
code_execution_tokens = {"input": 0, "output": 0}

# Set up the conversation memory (maintains context for MAINMODEL)
conversation_history = []

# Store file contents (part of the context for MAINMODEL)
file_contents = {}

# Code editor memory (maintains some context for CODEEDITORMODEL between calls)
code_editor_memory = []

# Files already present in code editor's context
code_editor_files = set()

# automode flag
automode = False

# Store file contents
file_contents = {}

# Global dictionary to store running processes
running_processes = {}

# Constants
CONTINUATION_EXIT_PHRASE = "AUTOMODE_COMPLETE"
MAX_CONTINUATION_ITERATIONS = 25
MAX_CONTEXT_TOKENS = 200000  # Reduced to 200k tokens for context window

DEFAULT_MODEL_CONFIG = {
    "model": "claude-3-5-sonnet-20240620",
    "model_provider": "anthropic",
    "max_tokens": 8000,
    "extra_headers": {"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
}
# Models
# Models that maintain context memory across interactions
MAIN_MODEL_CONFIG = {
    "model": "gemini-1.5-flash",
    "model_provider": "google_genai",
}

# Models that don't maintain context (memory is reset after each call)
TOOL_CHECKER_CONFIG = {
    "model": "gemini-1.5-flash",
    "model_provider": "google_genai",
}
CODE_EDITOR_CONFIG = {
    "model": "gemini-1.5-pro",
    "model_provider": "google_genai",
}
CODE_EXECUTION_CONFIG = {
    "model": "gemini-1.5-flash",
    "model_provider": "google_genai",
}
llm = init_chat_model(
    model=DEFAULT_MODEL_CONFIG.get("model"), configurable_fields="any"
)
# System prompts
BASE_SYSTEM_PROMPT = """
You are Claude, an AI assistant powered by Anthropic's Claude-3.5-Sonnet model, specialized in software development with access to a variety of tools and the ability to instruct and direct a coding agent and a code execution one. Your capabilities include:

1. Creating and managing project structures
2. Writing, debugging, and improving code across multiple languages
3. Providing architectural insights and applying design patterns
4. Staying current with the latest technologies and best practices
5. Analyzing and manipulating files within the project directory
6. Performing web searches for up-to-date information
7. Executing code and analyzing its output within an isolated 'code_execution_env' virtual environment
8. Managing and stopping running processes started within the 'code_execution_env'

Available tools and their optimal use cases:

1. create_folder: Create new directories in the project structure.
2. create_file: Generate new files with specified content. Strive to make the file as complete and useful as possible.
3. edit_and_apply: Examine and modify existing files by instructing a separate AI coding agent. You are responsible for providing clear, detailed instructions to this agent. When using this tool:
   - Provide comprehensive context about the project, including recent changes, new variables or functions, and how files are interconnected.
   - Clearly state the specific changes or improvements needed, explaining the reasoning behind each modification.
   - Include ALL the snippets of code to change, along with the desired modifications.
   - Specify coding standards, naming conventions, or architectural patterns to be followed.
   - Anticipate potential issues or conflicts that might arise from the changes and provide guidance on how to handle them.
4. execute_code: Run Python code exclusively in the 'code_execution_env' virtual environment and analyze its output. Use this when you need to test code functionality or diagnose issues. Remember that all code execution happens in this isolated environment. This tool now returns a process ID for long-running processes.
5. stop_process: Stop a running process by its ID. Use this when you need to terminate a long-running process started by the execute_code tool.
6. read_file: Read the contents of an existing file.
7. read_multiple_files: Read the contents of multiple existing files at once. Use this when you need to examine or work with multiple files simultaneously.
8. list_files: List all files and directories in a specified folder.
9. tavily_search: Perform a web search using the Tavily API for up-to-date information.

Tool Usage Guidelines:
- Always use the most appropriate tool for the task at hand.
- Provide detailed and clear instructions when using tools, especially for edit_and_apply.
- After making changes, always review the output to ensure accuracy and alignment with intentions.
- Use execute_code to run and test code within the 'code_execution_env' virtual environment, then analyze the results.
- For long-running processes, use the process ID returned by execute_code to stop them later if needed.
- Proactively use tavily_search when you need up-to-date information or additional context.
- When working with multiple files, consider using read_multiple_files for efficiency.

Error Handling and Recovery:
- If a tool operation fails, carefully analyze the error message and attempt to resolve the issue.
- For file-related errors, double-check file paths and permissions before retrying.
- If a search fails, try rephrasing the query or breaking it into smaller, more specific searches.
- If code execution fails, analyze the error output and suggest potential fixes, considering the isolated nature of the environment.
- If a process fails to stop, consider potential reasons and suggest alternative approaches.

Project Creation and Management:
1. Start by creating a root folder for new projects.
2. Create necessary subdirectories and files within the root folder.
3. Organize the project structure logically, following best practices for the specific project type.

Always strive for accuracy, clarity, and efficiency in your responses and actions. Your instructions must be precise and comprehensive. If uncertain, use the tavily_search tool or admit your limitations. When executing code, always remember that it runs in the isolated 'code_execution_env' virtual environment. Be aware of any long-running processes you start and manage them appropriately, including stopping them when they are no longer needed.

When using tools:
1. Carefully consider if a tool is necessary before using it.
2. Ensure all required parameters are provided and valid.
3. Handle both successful results and errors gracefully.
4. Provide clear explanations of tool usage and results to the user.

Remember, you are an AI assistant, and your primary goal is to help the user accomplish their tasks effectively and efficiently while maintaining the integrity and security of their development environment.
"""

AUTOMODE_SYSTEM_PROMPT = """
You are currently in automode. Follow these guidelines:

1. Goal Setting:
   - Set clear, achievable goals based on the user's request.
   - Break down complex tasks into smaller, manageable goals.

2. Goal Execution:
   - Work through goals systematically, using appropriate tools for each task.
   - Utilize file operations, code writing, and web searches as needed.
   - Always read a file before editing and review changes after editing.

3. Progress Tracking:
   - Provide regular updates on goal completion and overall progress.
   - Use the iteration information to pace your work effectively.

4. Tool Usage:
   - Leverage all available tools to accomplish your goals efficiently.
   - Prefer edit_and_apply for file modifications, applying changes in chunks for large edits.
   - Use tavily_search proactively for up-to-date information.

5. Error Handling:
   - If a tool operation fails, analyze the error and attempt to resolve the issue.
   - For persistent errors, consider alternative approaches to achieve the goal.

6. Automode Completion:
   - When all goals are completed, respond with "AUTOMODE_COMPLETE" to exit automode.
   - Do not ask for additional tasks or modifications once goals are achieved.

7. Iteration Awareness:
   - You have access to this {iteration_info}.
   - Use this information to prioritize tasks and manage time effectively.

Remember: Focus on completing the established goals efficiently and effectively. Avoid unnecessary conversations or requests for additional tasks.
"""


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


@tool
def get_jira_issue_details(issue_key: str) -> str:
    """
    Fetch details of a Jira issue, including its description and comments.

    This function takes an issue key as input, retrieves the corresponding Jira issue,
    and returns a formatted string containing the issue description and comments.
    If the issue is not found, it returns a message indicating that the issue was not found.
    If any other error occurs, it returns a generic error message.

    Args:
        issue_key: The key of the Jira issue.

    Returns:
        str: A formatted string containing the issue description and comments.
         If the issue is not found, returns a message indicating that the issue was not found.
         If any other error occurs, returns a generic error message.
    """
    try:
        # Get credentials from environment variables
        jira_url = os.getenv("JIRA_URL")
        username = os.getenv("JIRA_USERNAME")
        password = os.getenv("JIRA_PASSWORD")

        # Connect to Jira
        jira = JIRA(jira_url, basic_auth=(username, password))

        # Get the issue
        issue = jira.issue(issue_key)

        # Get the issue description
        description = issue.fields.description

        # Get the comments
        comments = issue.fields.comment.comments
        comment_texts = [comment.body for comment in comments]

        # Create the formatted string
        result = f"The description of the issue {issue_key} is: {description}\n"
        result += "Comments:\n"
        for i, comment in enumerate(comment_texts, start=1):
            result += f"Comment {i}: {comment}\n"

        return result

    except JIRAError as e:
        if e.status_code == 404:
            return f"Issue {issue_key} not found."
        else:
            return f"An error occurred: {e.text}"


@tool
def create_folder(path: str = ".") -> str:
    """
    Create a new folder at the specified path. This tool should be used when you need to create a new directory in the project structure. It will create all necessary parent directories if they don't exist. The tool will return a success message if the folder is created or already exists, and an error message if there's a problem creating the folder.

    Args:
        path: The absolute or relative path where the folder should be created. Use forward slashes (/) for path separation, even on Windows systems.

    Returns:
        str: A message indicating the success or failure of the folder creation.
    """
    try:
        os.makedirs(path, exist_ok=True)
        return f"Folder created: {path}"
    except Exception as e:
        return f"Error creating folder: {str(e)}"


@tool
def create_file(path: str = ".", content: str = "") -> str:
    """
    Create a new file at the specified path with the given content. This tool should be used when you need to create a new file in the project structure. It will create all necessary parent directories if they don't exist. The tool will return a success message if the file is created, and an error message if there's a problem creating the file or if the file already exists. The content should be as complete and useful as possible, including necessary imports, function definitions, and comments.

    Args:
        path: The absolute or relative path where the file should be created. Use forward slashes (/) for path separation, even on Windows systems.
        content: The content of the file. This should include all necessary code, comments, and formatting.

    Returns:
        str: A message indicating the success or failure of the file creation.
    """
    global file_contents
    try:
        with open(path, "w") as f:
            f.write(content)
        file_contents[path] = content
        return f"File created and added to system prompt: {path}"
    except Exception as e:
        return f"Error creating file: {str(e)}"


def highlight_diff(diff_text):
    return Syntax(diff_text, "diff", theme="monokai", line_numbers=True)


async def generate_edit_instructions(
    file_path, file_content, instructions, project_context, full_file_contents
):
    global code_editor_tokens, code_editor_memory, code_editor_files
    try:
        # Prepare memory context (this is the only part that maintains some context between calls)
        memory_context = "\n".join(
            [f"Memory {i + 1}:\n{mem}" for i, mem in enumerate(code_editor_memory)]
        )

        # Prepare full file contents context, excluding the file being edited if it's already in code_editor_files
        full_file_contents_context = "\n\n".join(
            [
                f"--- {path} ---\n{content}"
                for path, content in full_file_contents.items()
                if path != file_path or path not in code_editor_files
            ]
        )

        system_prompt = f"""
        You are an AI coding agent that generates edit instructions for code files. Your task is to analyze the provided code and generate SEARCH/REPLACE blocks for necessary changes. Follow these steps:

        1. Review the entire file content to understand the context:
        {file_content}

        2. Carefully analyze the specific instructions:
        {instructions}

        3. Take into account the overall project context:
        {project_context}

        4. Consider the memory of previous edits:
        {memory_context}

        5. Consider the full context of all files in the project:
        {full_file_contents_context}

        6. Generate SEARCH/REPLACE blocks for each necessary change. Each block should:
           - Include enough context to uniquely identify the code to be changed
           - Provide the exact replacement code, maintaining correct indentation and formatting
           - Focus on specific, targeted changes rather than large, sweeping modifications

        7. Ensure that your SEARCH/REPLACE blocks:
           - Address all relevant aspects of the instructions
           - Maintain or enhance code readability and efficiency
           - Consider the overall structure and purpose of the code
           - Follow best practices and coding standards for the language
           - Maintain consistency with the project context and previous edits
           - Take into account the full context of all files in the project

        IMPORTANT: RETURN ONLY THE SEARCH/REPLACE BLOCKS. NO EXPLANATIONS OR COMMENTS.
        USE THE FOLLOWING FORMAT FOR EACH BLOCK:

        <SEARCH>
        Code to be replaced
        </SEARCH>
        <REPLACE>
        New code to insert
        </REPLACE>

        If no changes are needed, return an empty list.
        """

        # Make the API call to CODEEDITORMODEL (context is not maintained except for code_editor_memory)
        response = llm.invoke(
            [SystemMessage(content=system_prompt)]
            + [
                HumanMessage(
                    "Generate SEARCH/REPLACE blocks for the necessary changes."
                )
            ],
            config={"configurable": CODE_EDITOR_CONFIG},
        )
        # Update token usage for code editor
        code_editor_tokens["input"] += response.usage_metadata["input_tokens"]
        code_editor_tokens["output"] += response.usage_metadata["output_tokens"]
        if isinstance(response.content, list):
            content = "\n".join(
                item["text"] for item in response.content if "text" in item
            )
        else:
            content = response.content
        response.content = content

        # Parse the response to extract SEARCH/REPLACE blocks
        edit_instructions = parse_search_replace_blocks(response.content)

        # Update code editor memory (this is the only part that maintains some context between calls)
        code_editor_memory.append(
            f"Edit Instructions for {file_path}:\n{response.content}"
        )

        # Add the file to code_editor_files set
        code_editor_files.add(file_path)

        return edit_instructions

    except Exception as e:
        console.print(
            f"Error in generating edit instructions: {str(e)}", style="bold red"
        )
        return []  # Return empty list if any exception occurs


def parse_search_replace_blocks(response_text):
    blocks = []
    pattern = r"<SEARCH>\n(.*?)\n</SEARCH>\n<REPLACE>\n(.*?)\n</REPLACE>"
    matches = re.findall(pattern, response_text, re.DOTALL)

    for search, replace in matches:
        blocks.append({"search": search.strip(), "replace": replace.strip()})

    return json.dumps(blocks)  # Keep returning JSON string


@tool
async def edit_and_apply(
    *,
    instructions: str,
    project_context: str,
    path: str,
    is_automode: bool = False,
    max_retries: int = 3,
):
    """
    Apply AI-powered improvements to a file based on specific instructions and detailed project context. This function reads the file, processes it in batches using AI with conversation history and comprehensive code-related project context. It generates a diff and allows the user to confirm changes before applying them. The goal is to maintain consistency and prevent breaking connections between files. This tool should be used for complex code modifications that require understanding of the broader project context.

    Args:
        path: The absolute or relative path of the file to edit. Use forward slashes (/) for path separation, even on Windows systems.
        instructions: After completing the code review, construct a plan for the change between <PLANNING> tags. Ask for additional source files or documentation that may be relevant. The plan should avoid duplication (DRY principle), and balance maintenance and flexibility. Present trade-offs and implementation choices at this step. Consider available Frameworks and Libraries and suggest their use when relevant. STOP at this step if we have not agreed a plan.
                    Once agreed, produce code between <OUTPUT> tags. Pay attention to Variable Names, Identifiers and String Literals, and check that they are reproduced accurately from the original source files unless otherwise directed. When naming by convention surround in double colons and in ::UPPERCASE::. Maintain existing code style, use language appropriate idioms. Produce Code Blocks with the language specified after the first backticks
        project_context: Comprehensive context about the project, including recent changes, new variables or functions, interconnections between files, coding standards, and any other relevant information that might affect the edit.

    Returns:
    str: A message indicating the success or failure of applying changes to the file.
    """
    global file_contents
    print("In Edit and Apply Function")
    try:
        original_content = file_contents.get(path, "")
        if not original_content:
            with open(path, "r") as file:
                original_content = file.read()
            file_contents[path] = original_content

        for attempt in range(max_retries):
            edit_instructions_json = await generate_edit_instructions(
                path, original_content, instructions, project_context, file_contents
            )

            if edit_instructions_json:
                edit_instructions = json.loads(
                    edit_instructions_json
                )  # Parse JSON here
                console.print(
                    Panel(
                        f"Attempt {attempt + 1}/{max_retries}: The following SEARCH/REPLACE blocks have been generated:",
                        title="Edit Instructions",
                        style="cyan",
                    )
                )
                for i, block in enumerate(edit_instructions, 1):
                    console.print(f"Block {i}:")
                    console.print(
                        Panel(
                            f"SEARCH:\n{block['search']}\n\nREPLACE:\n{block['replace']}",
                            expand=False,
                        )
                    )

                edited_content, changes_made, failed_edits = await apply_edits(
                    path, edit_instructions, original_content
                )

                if changes_made:
                    file_contents[path] = (
                        edited_content  # Update the file_contents with the new content
                    )
                    console.print(
                        Panel(
                            f"File contents updated in system prompt: {path}",
                            style="green",
                        )
                    )

                    if failed_edits:
                        console.print(
                            Panel(
                                "Some edits could not be applied. Retrying...",
                                style="yellow",
                            )
                        )
                        instructions += f"\n\nPlease retry the following edits that could not be applied:\n{failed_edits}"
                        original_content = edited_content
                        continue

                    return f"Changes applied to {path}"
                elif attempt == max_retries - 1:
                    return f"No changes could be applied to {path} after {max_retries} attempts. Please review the edit instructions and try again."
                else:
                    console.print(
                        Panel(
                            f"No changes could be applied in attempt {attempt + 1}. Retrying...",
                            style="yellow",
                        )
                    )
            else:
                return f"No changes suggested for {path}"

        return f"Failed to apply changes to {path} after {max_retries} attempts."
    except Exception as e:
        return f"Error editing/applying to file: {str(e)}"


async def apply_edits(
    file_path: str, edit_instructions: str, original_content: str
) -> Tuple[str, bool, str]:
    """
    Apply a series of edits to a file based on the provided edit instructions.

    This function iterates through the edit instructions, searches for the specified content in the file,
    and replaces it with the new content. It uses a progress bar to display the progress of the edits.
    If a search content is not found in the file, it is considered a failed edit. The function returns
    the edited content, a boolean indicating whether any changes were made, and a string containing
    the details of any failed edits.

    Parameters:
    file_path (str): The path to the file to edit.
    edit_instructions (str): A list of edit instructions, each containing a 'search' and 'replace' field.
    original_content (str): The original content of the file.

    Returns:
    Tuple[str, bool, str]: A tuple containing the edited content, a boolean indicating whether any changes
                           were made, and a string containing the details of any failed edits.
    """
    changes_made = False
    edited_content = original_content
    total_edits = len(edit_instructions)
    failed_edits = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        edit_task = progress.add_task("[cyan]Applying edits...", total=total_edits)

        for i, edit in enumerate(edit_instructions, 1):
            search_content = edit["search"].strip()
            replace_content = edit["replace"].strip()

            # Use regex to find the content, ignoring leading/trailing whitespace
            pattern = re.compile(re.escape(search_content), re.DOTALL)
            match = pattern.search(edited_content)

            if match:
                # Replace the content, preserving the original whitespace
                start, end = match.span()
                # Strip <SEARCH> and <REPLACE> tags from replace_content
                replace_content_cleaned = re.sub(
                    r"</?SEARCH>|</?REPLACE>", "", replace_content
                )
                edited_content = (
                    edited_content[:start]
                    + replace_content_cleaned
                    + edited_content[end:]
                )
                changes_made = True

                # Display the diff for this edit
                diff_result = generate_diff(search_content, replace_content, file_path)
                console.print(
                    Panel(
                        diff_result,
                        title=f"Changes in {file_path} ({i}/{total_edits})",
                        style="cyan",
                    )
                )
            else:
                console.print(
                    Panel(
                        f"Edit {i}/{total_edits} not applied: content not found",
                        style="yellow",
                    )
                )
                failed_edits.append(f"Edit {i}: {search_content}")

            progress.update(edit_task, advance=1)

    if not changes_made:
        console.print(
            Panel(
                "No changes were applied. The file content already matches the desired state.",
                style="green",
            )
        )
    else:
        # Write the changes to the file
        with open(file_path, "w") as file:
            file.write(edited_content)
        console.print(Panel(f"Changes have been written to {file_path}", style="green"))

    return edited_content, changes_made, "\n".join(failed_edits)


def generate_diff(original, new, path):
    diff = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )

    diff_text = "".join(diff)
    highlighted_diff = highlight_diff(diff_text)

    return highlighted_diff


@tool
async def execute_code(code: str, timeout: int = 10):
    """
    Execute Python code in the 'code_execution_env' virtual environment and return the output. This tool should be used when you need to run code and see its output or check for errors. All code execution happens exclusively in this isolated environment. The tool will return the standard output, standard error, and return code of the executed code. Long-running processes will return a process ID for later management.

    Args:
        code: The Python code to execute in the 'code_execution_env' virtual environment. Include all necessary imports and ensure the code is complete and self-contained.
        timeout: The maximum time to wait for the code execution to complete. Defaults to 10 seconds.

    Returns:
        tuple: A tuple containing the process ID and the execution result. The execution result is a
           string that includes the process ID, standard output, standard error, and return code.
           If the code execution times out, the standard output will indicate that the process
           is still running in the background.
    """
    global running_processes
    venv_path, activate_script = setup_virtual_environment()

    # Generate a unique identifier for this process
    process_id = f"process_{len(running_processes)}"

    # Write the code to a temporary file
    with open(f"{process_id}.py", "w") as f:
        f.write(code)

    # Prepare the command to run the code
    if sys.platform == "win32":
        command = f'"{activate_script}" && python3 {process_id}.py'
    else:
        command = f'source "{activate_script}" && python3 {process_id}.py'

    # Create a process to run the command
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        shell=True,
        preexec_fn=None if sys.platform == "win32" else os.setsid,
    )

    # Store the process in our global dictionary
    running_processes[process_id] = process

    try:
        # Wait for initial output or timeout
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        stdout = stdout.decode()
        stderr = stderr.decode()
        return_code = process.returncode
    except asyncio.TimeoutError:
        # If we timeout, it means the process is still running
        stdout = "Process started and running in the background."
        stderr = ""
        return_code = "Running"

    execution_result = f"Process ID: {process_id}\n\nStdout:\n{stdout}\n\nStderr:\n{stderr}\n\nReturn Code: {return_code}"
    return process_id, execution_result


@tool
def read_file(path: str = ".") -> str:
    """
    Read the contents of a file at the specified path. This tool should be used when you need to examine the contents of an existing file. It will return the entire contents of the file as a string. If the file doesn't exist or can't be read, an appropriate error message will be returned.

    Args:
        The absolute or relative path of the file to read. Use forward slashes (/) for path separation, even on Windows systems.

    Returns:
        str: A string containing the contents of the file, or an error message if the file doesn't exist or can't be read.
    """
    global file_contents
    try:
        with open(path, "r") as f:
            content = f.read()
        file_contents[path] = content
        return f"File '{path}' has been read and stored in the system prompt."
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def read_multiple_files(paths: List[str]):
    """
    Read the contents of multiple files at the specified paths. This tool should be used when you need to examine the contents of multiple existing files at once. It will return the status of reading each file, and store the contents of successfully read files in the system prompt. If a file doesn't exist or can't be read, an appropriate error message will be returned for that file.

    Args:
        paths: An array of absolute or relative paths of the files to read. Use forward slashes (/) for path separation, even on Windows systems.

    Returns:
        str: A string containing the results of reading each file, separated by newlines.
    """
    global file_contents
    results = []
    for path in paths:
        try:
            with open(path, "r") as f:
                content = f.read()
            file_contents[path] = content
            results.append(
                f"File '{path}' has been read and stored in the system prompt."
            )
        except Exception as e:
            results.append(f"Error reading file '{path}': {str(e)}")
    return "\n".join(results)


@tool
def list_files(path: str):
    """
    List all files and directories in the specified folder. This tool should be used when you need to see the contents of a directory. It will return a list of all files and subdirectories in the specified path. If the directory doesn't exist or can't be read, an appropriate error message will be returned.

    Args:
        path: The absolute or relative path of the folder to list. Use forward slashes (/) for path separation, even on Windows systems. If not provided, the current working directory will be used.

    Returns:
        str: A string containing the names of all files and directories in the specified path, separated by newlines.
         If an error occurs, returns an error message.
    """
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {str(e)}"


@tool
def tavily_search(query: str):
    """
    Perform a web search using the Tavily API to get up-to-date information or additional context. This tool should be used when you need current information or feel a search could provide a better answer to the user's query. It will return a summary of the search results, including relevant snippets and source URLs.

    Args:
        query: The search query. Be as specific and detailed as possible to get the most relevant results.

    Returns:
        str: The response from the Tavily API, which includes a summary of the search results. If an error occurs,
         the function returns an error message.
    """
    try:
        response = tavily.qna_search(query=query, search_depth="advanced")
        return response
    except Exception as e:
        return f"Error performing search: {str(e)}"


@tool
def stop_process(process_id: int):
    """
    Stop a running process by its ID. This tool should be used to terminate long-running processes that were started by the execute_code tool. It will attempt to stop the process gracefully, but may force termination if necessary. The tool will return a success message if the process is stopped, and an error message if the process doesn't exist or can't be stopped.

    Args:
        process_id: The ID of the process to stop, as returned by the execute_code tool for long-running processes.

    Returns:
        str: A message indicating the success or failure of stopping the process.
    """
    global running_processes
    if process_id in running_processes:
        process = running_processes[process_id]
        if sys.platform == "win32":
            process.terminate()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        del running_processes[process_id]
        return f"Process {process_id} has been stopped."
    else:
        return f"No running process found with ID {process_id}."


tools = [
    create_folder,
    create_file,
    edit_and_apply,
    execute_code,
    stop_process,
    read_file,
    read_multiple_files,
    list_files,
    tavily_search,
    get_jira_issue_details,
]
tools_map = {t.name: t for t in tools}
gemai_tools = [tool_to_dict(convert_to_genai_function_declarations(tools))]

async def call_tools(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = None
        is_error = False
        if tool_name == "edit_and_apply":
            tool_input["is_automode"] = automode
            result = await tools_map[tool_name].ainvoke(tool_input)
        elif tool_name == "execute_code":
            process_id, execution_result = await tools_map[tool_name].ainvoke(
                tool_input
            )
            analysis_task = asyncio.create_task(
                send_to_ai_for_executing(tool_input["code"], execution_result)
            )
            analysis = await analysis_task
            result = f"{execution_result}\n\nAnalysis:\n{analysis}"
            if process_id in running_processes:
                result += "\n\nNote: The process is still running in the background."
        else:
            result = tools_map[tool_name].invoke(tool_input)
            is_error = False

        return {"content": result, "is_error": is_error}
    except KeyError as e:
        logging.error(f"Missing required parameter {str(e)} for tool {tool_name}")
        return {
            "content": f"Error: Missing required parameter {str(e)} for tool {tool_name}",
            "is_error": True,
        }
    except Exception as e:
        logging.error(f"Error executing tool {tool_name}: {str(e)}")
        return {
            "content": f"Error executing tool {tool_name}: {str(e)}",
            "is_error": True,
        }


def encode_image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            max_size = (1024, 1024)
            img.thumbnail(max_size, Image.DEFAULT_STRATEGY)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG")
            return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
    except Exception as e:
        return f"Error encoding image: {str(e)}"


async def send_to_ai_for_executing(code, execution_result):
    global code_execution_tokens

    try:
        system_prompt = f"""
        You are an AI code execution agent. Your task is to analyze the provided code and its execution result from the 'code_execution_env' virtual environment, then provide a concise summary of what worked, what didn't work, and any important observations. Follow these steps:

        1. Review the code that was executed in the 'code_execution_env' virtual environment:
        {code}

        2. Analyze the execution result from the 'code_execution_env' virtual environment:
        {execution_result}

        3. Provide a brief summary of:
           - What parts of the code executed successfully in the virtual environment
           - Any errors or unexpected behavior encountered in the virtual environment
           - Potential improvements or fixes for issues, considering the isolated nature of the environment
           - Any important observations about the code's performance or output within the virtual environment
           - If the execution timed out, explain what this might mean (e.g., long-running process, infinite loop)

        Be concise and focus on the most important aspects of the code execution within the 'code_execution_env' virtual environment.

        IMPORTANT: PROVIDE ONLY YOUR ANALYSIS AND OBSERVATIONS. DO NOT INCLUDE ANY PREFACING STATEMENTS OR EXPLANATIONS OF YOUR ROLE.
        """
        response = llm.invoke(
            [SystemMessage(content=system_prompt)]
            + [
                HumanMessage(
                    f"Analyze this code execution from the 'code_execution_env' virtual environment:\n\nCode:\n{code}\n\nExecution Result:\n{execution_result}",
                )
            ],
            config={"configurable": CODE_EXECUTION_CONFIG},
        )

        # Update token usage for code execution
        code_execution_tokens["input"] += response.usage_metadata["input_tokens"]
        code_execution_tokens["output"] += response.usage_metadata["output_tokens"]

        if isinstance(response.content, list):
            content = "\n".join(
                item["text"] for item in response.content if "text" in item
            )
        else:
            content = response.content
        response.content = content

        analysis = response.content

        return analysis

    except Exception as e:
        console.print(
            f"Error in AI code execution analysis: {str(e)}", style="bold red"
        )
        return f"Error analyzing code execution from 'code_execution_env': {str(e)}"


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
            formatted_chat += f"## AI Engineer\n\n{message.content}\n\n"
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

    # Filter conversation history to maintain context
    filtered_conversation_history = []
    for message in conversation_history:
        assert not isinstance(message.content, list)
        if isinstance(message.content, list):
            filtered_content = [
                content
                for content in message.content
                if content.get("type") != "tool_result"
                or (
                    content.get("type") == "tool_result"
                    and not any(
                        keyword in content.get("output", "")
                        for keyword in [
                            "File contents updated in system prompt",
                            "File created and added to system prompt",
                            "has been read and stored in the system prompt",
                        ]
                    )
                )
            ]
            if filtered_content:
                filtered_conversation_history.append(
                    {**message, "content": filtered_content}
                )
        else:
            filtered_conversation_history.append(message)

    # Combine filtered history with current conversation to maintain context
    messages = filtered_conversation_history + current_conversation
    current_prompt = update_system_prompt(current_iteration, max_iterations)
    # print("CURRENT PROMPT", current_prompt)
    # print("Before calling main model", messages)
    try:
        # MAINMODEL call, which maintains context
        response = llm.bind_tools(gemai_tools).invoke(
            [SystemMessage(content=current_prompt)] + messages,
            config={"configurable": MAIN_MODEL_CONFIG},
        )
        # Update token usage for MAINMODEL
        main_model_tokens["input"] += response.usage_metadata["input_tokens"]
        main_model_tokens["output"] += response.usage_metadata["output_tokens"]
    except APIStatusError as e:
        with open("main_model_error_log.txt", "a") as f:
            f.write(f"# {datetime.datetime.now()}\n\n")
            f.write(dump_message(messages))
        if e.status_code == 429:
            console.print(
                Panel(
                    "Rate limit exceeded. Retrying after a short delay...",
                    title="API Error",
                    style="bold yellow",
                )
            )
            time.sleep(5)
            return await chat_with_claude(
                user_input, image_path, current_iteration, max_iterations
            )
        else:
            console.print(
                Panel(f"API Error: {str(e)}", title="API Error", style="bold red")
            )
            return (
                "I'm sorry, there was an error communicating with the AI. Please try again.",
                False,
            )
    except APIError as e:
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

    assistant_response = ""
    exit_continuation = False
    tool_uses = [t for t in response.tool_calls]
    if isinstance(response.content, list):
        content = "\n".join(item["text"] for item in response.content if "text" in item)
    else:
        content = response.content
    response.content = content
    assistant_response += response.content
    assert isinstance(response.content, str)

    if CONTINUATION_EXIT_PHRASE in response.content:
        exit_continuation = True

    console.print(
        Panel(
            Markdown(assistant_response),
            title="Claude's Response",
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
    current_conversation.append(response)
    # print("TOOL USES", tool_uses)
    for tool_use in tool_uses:
        tool_name = tool_use["name"]
        tool_input = tool_use["args"]
        tool_use_id = tool_use["id"]
        # print("TOOL input", tool_input)
        console.print(Panel(f"Tool Used: {tool_name}", style="green"))
        console.print(
            Panel(f"Tool Input: {json.dumps(tool_input, indent=2)}", style="green")
        )

        tool_result = await call_tools(tool_name, tool_input)

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
        current_conversation.append(
            ToolMessage(tool_result["content"], tool_call_id=tool_use_id)
        )

        # Update the file_contents dictionary if applicable
        if (
            tool_name in ["create_file", "edit_and_apply", "read_file"]
            and not tool_result["is_error"]
        ):
            if "path" in tool_input:
                file_path = tool_input["path"]
                if (
                    "File contents updated in system prompt" in tool_result["content"]
                    or "File created and added to system prompt"
                    in tool_result["content"]
                    or "has been read and stored in the system prompt"
                    in tool_result["content"]
                ):
                    # The file_contents dictionary is already updated in the tool function
                    pass

        messages = filtered_conversation_history + current_conversation
        # print("Before calling tool checker model", messages)
        try:
            tool_response = llm.bind_tools(gemai_tools).invoke(
                [
                    SystemMessage(
                        content=update_system_prompt(current_iteration, max_iterations)
                    )
                ]
                + messages,
                config={"configurable": TOOL_CHECKER_CONFIG},
            )
            tool_checker_tokens["input"] += tool_response.usage_metadata["input_tokens"]
            tool_checker_tokens["output"] += tool_response.usage_metadata[
                "output_tokens"
            ]

            if isinstance(tool_response.content, list):
                content: str = "\n".join(
                    item["text"] for item in tool_response.content if "text" in item
                )
            else:
                content: str = tool_response.content
            tool_response.content = content
            # remove any request tool calls
            tool_response.tool_calls = []
            tool_checker_response = tool_response.content
            current_conversation.append(tool_response)
            console.print(
                Panel(
                    Markdown(tool_checker_response),
                    title="Claude's Response to Tool Result",
                    title_align="left",
                    border_style="blue",
                    expand=False,
                )
            )
            assistant_response += "\n\n" + tool_checker_response
        except APIError as e:
            error_message = f"Error in tool response: {str(e)}"
            with open("tool_checker_error_log.txt", "a") as f:
                f.write(f"# {datetime.datetime.now()}\n\n")
                f.write(dump_message(messages))
            console.print(Panel(error_message, title="Error", style="bold red"))
            assistant_response += f"\n\n{error_message}"
    # print("ASSISTANT RESPONSE VALUE1", assistant_response)

    conversation_history = messages

    # Display token usage at the end
    display_token_usage()

    return assistant_response, exit_continuation


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
