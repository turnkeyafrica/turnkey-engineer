import asyncio
import base64
import difflib
import io
import json
import logging
import re
import signal
import sys
import venv
from typing import Literal, Optional, Tuple, List, Dict, Any

from PIL import Image
from jira import JIRA, JIRAError
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain_core.tools import tool
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax

from config import *
from config import jira


def get_content(msg: AIMessage):
    if isinstance(msg.content, list):
        return "\n".join([c.get("text", "") for c in msg.content])
    else:
        return msg.content


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



@tool
def get_jira_info(info_type: Optional[Literal["whoami", "projects"]] = "whoami"):
    """

    Retrieves information from Jira based on the given info_type.

    Parameters:
        info_type (Optional[Literal["whoami", "projects"]]): The type of information to retrieve. Defaults to "whoami".

    Returns:
        Union[Dict[str, str], List[Dict[str, str]]]: The retrieved information. If info_type is "whoami", returns a dictionary with user information. If info_type is "projects", returns a list of dictionaries with project information.
    """
    if info_type == "whoami":
        return json.dumps(jira.myself())
    projects = jira.projects()
    return json.dumps([{"key": p.key, "name": p.name} for p in projects], indent=2)


@tool
def find_jira_issues_jql(jql_query: str):
    """

    Finds JIRA issues based on the provided JQL query.

    Args:
    - jql_query: The JQL query to search for issues.

    Returns:
    - list: A list of dictionaries containing the key and summary of each issue found.

    Examples:
        >>> find_jira_issues_jql("project=INN")
        [{'key': 'INN-46', 'summary': 'MemGPT Session Management Issue'},
         {'key': 'INN-45', 'summary': 'Brokerage Quotations AI Agent'},
         {'key': 'INN-44', 'summary': 'AI Routing Agent'}]
    """
    issues = jira.search_issues(jql_query)
    return json.dumps([{"key": i.key, "summary": i.fields.summary} for i in issues], indent=2)


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
def jira_issue_comment_crud(issue_key: str, action: Literal["add", "update", "delete"], comment_id: Optional[int] = None, comment: Optional[str] = None) -> str:
    """
    Perform CRUD operations on comments of a Jira issue.

    Args:
        issue_key: The key of the Jira issue.
        action: The action to perform on the comment. It can be "add", "update", or "delete".
        comment_id: The ID of the comment to update or delete. Required for "update" and "delete" actions.
        comment: The content of the comment to add or update. Required for "add" and "update" actions.

    Returns:
        str: A message indicating the success or failure of the operation.
    """
    if action == "add":
        if not comment:
            raise ValueError("Comment is required for 'add' action.")
        comment_obj = jira.add_comment(issue_key, comment)
        return f"Comment added to issue {issue_key}: {comment_obj.raw}."
    
    if action in ["update", "delete"]:
        if not comment_id:
            raise ValueError(f"Comment ID is required for '{action}' action.")
        comment_obj = jira.comment(issue_key, comment_id)
        if not comment_obj:
            return f"Comment {comment_id} not found for issue {issue_key}."
        
        if action == "update":
            if not comment:
                raise ValueError("Comment is required for 'update' action.")
            comment_obj.update(body=comment)
            return f"Comment {comment_id} updated for issue {issue_key}."
        
        if action == "delete":
            comment_obj.delete()
            return f"Comment {comment_id} deleted from issue {issue_key}."
    
    return "Invalid action. Please specify 'add', 'update', or 'delete'."

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
        # data = content.replace("\\\\", "")
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
    get_jira_info,
    find_jira_issues_jql,
    get_jira_issue_details,
    jira_issue_comment_crud
]
tools_map = {t.name: t for t in tools}
# gemai_tools = [tool_to_dict(convert_to_genai_function_declarations(tools))]


async def invoke_tool(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
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
