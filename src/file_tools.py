from pathlib import Path
from typing import Literal, TypedDict, Annotated, NotRequired

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain_core.messages import  ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from src.prompts import WRITE_TODOS_DESCRIPTION, TODO_USAGE_INSTRUCTIONS

from src.prompts import LS_DESCRIPTION,READ_FILE_DESCRIPTION,WRITE_FILE_DESCRIPTION
from src.utils import show_prompt, format_messages

load_dotenv(override=True)

class Todo(TypedDict):
    """A structured task item for tracking progress through complex workflows.

    Attributes:
        content: Short, specific description of the task
        status: Current state - pending, in_progress, or completed
    """

    content: str
    status: Literal["pending", "in_progress", "completed"]

def file_reducer(left, right):
    """Merge two file dictionaries, with right side taking precedence.

    Used as a reducer function for the files field in agent state,
    allowing incremental updates to the virtual file system.

    Args:
        left: Left side dictionary (existing files)
        right: Right side dictionary (new/updated files)

    Returns:
        Merged dictionary with right values overriding left values
    """
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return {**left, **right}


class DeepAgentState(AgentState):
    """Extended agent state that includes task tracking and virtual file system.

    Inherits from LangGraph's AgentState and adds:
    - todos: List of Todo items for task planning and progress tracking
    - files: Virtual file system stored as dict mapping filenames to content
    """

    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]

@tool(description=LS_DESCRIPTION)
def ls(state: Annotated[DeepAgentState, InjectedState]) -> list[str]:
    """List all files in the virtual filesystem."""
    print("List files...")
    return list(state.get("files", {}).keys())


@tool(description=READ_FILE_DESCRIPTION, parse_docstring=True)
def read_file(
    file_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file content from virtual filesystem with optional offset and limit.

    Args:
        file_path: Path to the file to read
        state: Agent state containing virtual filesystem (injected in tool node)
        offset: Line number to start reading from (default: 0)
        limit: Maximum number of lines to read (default: 2000)

    Returns:
        Formatted file content with line numbers, or error message if file not found
    """

    print(f"read_file.. path- {file_path} ")

    files = state.get("files", {})
    if file_path not in files:
        return f"Error: File '{file_path}' not found"

    content = files[file_path]
    if not content:
        return "System reminder: File exists but has empty contents"

    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    result_lines = []
    for i in range(start_idx, end_idx):
        line_content = lines[i][:2000]  # Truncate long lines
        result_lines.append(f"{i + 1:6d}\t{line_content}")

    return "\n".join(result_lines)

@tool(description=WRITE_FILE_DESCRIPTION, parse_docstring=True)
def write_file(
    file_path: str,
    content: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Write content to a file in the virtual filesystem.

    Args:
        file_path: Path where the file should be created/updated
        content: Content to write to the file
        state: Agent state containing virtual filesystem (injected in tool node)
        tool_call_id: Tool call identifier for message response (injected in tool node)

    Returns:
        Command to update agent state with new file content
    """
    print(f"write_file... file_path:- {file_path}. content {content}")

    files = state.get("files", {})
    files[file_path] = content
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )


# File usage instructions
FILE_USAGE_INSTRUCTIONS = """
You have access to a virtual file system.

MANDATORY WORKFLOW:

1. ALWAYS call ls() first

2. BEFORE doing any research, ALWAYS save the original user request
   using write_file().

3. Save the request in a file named:
   user_request.txt

4. The file content MUST follow this exact format:

User Request: <original user request>

The user wants: <short understanding of the request>

5. After saving the request, call web_search() exactly once.

6. Before generating the final response, call read_file().

If you skip any step, your response is incorrect.
"""


SIMPLE_RESEARCH_INSTRUCTIONS = """
Use web_search exactly once to gather information.
After searching, save the result using write_file().
Then use read_file() before answering.
"""

# Full prompt
INSTRUCTIONS = (
        FILE_USAGE_INSTRUCTIONS + "\n\n" + "=" * 80 + "\n\n" + SIMPLE_RESEARCH_INSTRUCTIONS
)

# Mock search result
search_result = """The Model Context Protocol (MCP) is an open standard protocol developed 
by Anthropic to enable seamless integration between AI models and external systems like 
tools, databases, and other services. It acts as a standardized communication layer, 
allowing AI models to access and utilize data from various sources in a consistent and 
efficient manner. Essentially, MCP simplifies the process of connecting AI assistants 
to external services by providing a unified language for data exchange. """


# Mock search tool
@tool(parse_docstring=True)
def web_search(
    query: str,
):
    """Search the web for information on a specific topic.

    This tool performs web searches and returns relevant results
    for the given query. Use this when you need to gather information from
    the internet about any topic.

    Args:
        query: The search query string. Be specific and clear about what
               information you're looking for.

    Returns:
        Search results from search engine.

    Example:
        web_search("machine learning applications in healthcare")
    """
    print("Mock search tool...")
    return search_result


# Create agent using create_react_agent directly
model = init_chat_model(model="anthropic:claude-haiku-4-5-20251001", temperature=0.0)
tools = [ls, read_file, write_file, web_search]

# Create agent with system prompt
agent = create_agent(
    model, tools, system_prompt=INSTRUCTIONS, state_schema=DeepAgentState
)
agent.get_graph(xray=1).print_ascii()
agent.get_graph(xray=1).draw_mermaid_png(
    output_file_path="src/graph_flow_diagram/" + Path(__file__).stem + ".png")

def main():
    #print(" list prompts...")
    #show_prompt(LS_DESCRIPTION)
    #show_prompt(READ_FILE_DESCRIPTION)
    #show_prompt(WRITE_FILE_DESCRIPTION)
    #show_prompt(INSTRUCTIONS)

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Give me an overview of Model Context Protocol (MCP).",
                }
            ],
            "files": {},
        }
    )
    format_messages(result["messages"])

    print(f"Saved file = {result['files']}")
    # Output
    # saved file = {'user_request.txt': 'User Request: Give me an overview of Model Context Protocol (MCP).\n\nThe user wants: A comprehensive overview of what Model Context Protocol (MCP) is, including its purpose, key features, how it works, and its significance in AI/LLM applications.',
    # 'mcp_research.txt': 'Model Context Protocol (MCP) - Research Results\n\nDefinition:\nThe Model Context Protocol (MCP) is an open standard protocol developed by Anthropic to enable seamless integration between AI models and external systems like tools, databases, and other services.\n\nKey Characteristics:\n- Acts as a standardized communication layer\n- Allows AI models to access and utilize data from various sources\n- Provides consistent and efficient manner of data exchange\n- Simplifies the process of connecting AI assistants to external services\n- Provides a unified language for data exchange\n\nPurpose:\n- Enable seamless integration between AI models and external systems\n- Standardize how AI assistants communicate with tools, databases, and services\n- Simplify data exchange between AI and external resources'}
    # (langgraph-deep-agent) PS C:\Gen_AI_workspace\langgraph-deep-agent>


# uv run -m src.file_tools
if __name__ == '__main__':
    main()
