
# ---------------Prompt-Driven Agent---------------------------------
# Current example is Prompt-Driven Agent
# Like telling employee:
#
# "Please follow process carefully."
#
# Maybe they do.
# Maybe they shortcut.
 # Right now:
#
# LLM
#   ↓
# choose any tool
#   ↓
# choose next step
#   ↓
# finish whenever it wants

# ---------------Graph-Enforced Version---------------------------------
#   START
#   ↓
# planner_node
#   ↓
# read_todos_node
#   ↓
# execution_node
#   ↓
# reflection_node
#   ↓
# update_todos_node
#   ↓
# should_continue?
#    ├── yes → execution_node
#    └── no  → END

# In Graph-Enforced Version:
#
# TODO reading guaranteed
# reflection guaranteed
# update guaranteed
# completion check guaranteed
# Even if model wants shortcut:  graph prevents it

#----------------------------------------------------------------------------------

# The biggest evolution in agent engineering is:
# from "Prompt engineering" to "Workflow engineering"
#-------------------------------------

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
from src.utils import format_messages, show_prompt

load_dotenv(override=True)

class Todo(TypedDict):
    """A structured task item for tracking progress through complex workflows.

    Attributes:
        content: Short, specific description of the task
        status: Current state - pending, in_progress, or completed
    """

    content: str
    status: Literal["pending", "in_progress", "completed"]
    #  Real Production Evolution
    #    id
    #    content
    #    status
    #    priority
    #    dependencies
    #    assigned_tool
    #    retries
    #    result
    #    created_at
    #    updated_at




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

@tool(description=WRITE_TODOS_DESCRIPTION,parse_docstring=True)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Create or update the agent's TODO list for task planning and tracking.

    Args:
        todos: List of Todo items with content and status
        tool_call_id: Tool call identifier for message response

    Returns:
        Command to update agent state with new TODO list
    """
    print(f"write_todos... :-{todos}")

    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    )


@tool(parse_docstring=True)
def read_todos(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Read the current TODO list from the agent state.

    This tool allows the agent to retrieve and review the current TODO list
    to stay focused on remaining tasks and track progress through complex workflows.

    Args:
        state: Injected agent state containing the current TODO list
        tool_call_id: Injected tool call identifier for message tracking

    Returns:
        Formatted string representation of the current TODO list
    """
    todos = state.get("todos", [])

    print(f"read_todos... :-{todos}")

    if not todos:
        return "No todos currently in the list."

    result = "Current TODO List:\n"
    for i, todo in enumerate(todos, 1):
        status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}
        emoji = status_emoji.get(todo["status"], "❓")
        result += f"{i}. {emoji} {todo['content']} ({todo['status']})\n"

    return result.strip()

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
    print(f"web_search query.... {query}")

    return search_result


# Create agent using create_react_agent directly
model = init_chat_model(model="anthropic:claude-haiku-4-5-20251001", temperature=0.0)
tools = [write_todos, web_search, read_todos]

# Add mock research instructions
SIMPLE_RESEARCH_INSTRUCTIONS = """IMPORTANT: Just make a single call to the web_search tool and use the result provided by the tool to answer the user's question."""

# Create agent
agent = create_agent(
    model,
    tools,
    system_prompt=TODO_USAGE_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SIMPLE_RESEARCH_INSTRUCTIONS,
    state_schema=DeepAgentState,
)



def main():

    agent.get_graph(xray=1).print_ascii()
    agent.get_graph(xray=1).draw_mermaid_png(
        output_file_path="src/graph_flow_diagram/" + Path(__file__).stem + ".png")

    # Example usage
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Research MCP, LangGraph, AutoGen, CrewAI, then compare them in a table",
                }
            ],
            "todos": [],
        }
    )

    format_messages(result["messages"])

# uv run -m src.planning_TODO_lists
if __name__ == '__main__':
    main()