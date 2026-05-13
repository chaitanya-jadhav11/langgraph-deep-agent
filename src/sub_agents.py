# flow
# User
#   ↓
# Main Agent
#   ↓
# task() tool
#   ↓
# Research Sub-Agent
#   ↓
# web_search()
#   ↓
# Result back to Main Agent
#   ↓
# User response
#--------------------------------------------------------------------------------
# Prompt-driven orchestration system (current example)
# And yes — it has scalability/reliability limitations.#

# In production, teams increasingly prefer:
# LangGraph-style controlled orchestration
# because it provides:

# determinism
# persistence
# observability
# retries
# parallelism
# human approval
# durable state
# -------------------------------------------------------------------------------------
from datetime import datetime
from pathlib import Path
from typing import TypedDict, NotRequired, Annotated, Required, Generic, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from src.utils import show_prompt, format_messages
from src.prompts import SUBAGENT_USAGE_INSTRUCTIONS
from langchain.agents import create_agent
from langchain.agents.middleware.types import ResponseT, JumpTo, OmitFromInput, PrivateStateAttr
from langchain_core.messages import ToolMessage, AnyMessage
from langchain_core.tools import InjectedToolCallId, BaseTool, tool
from langgraph.channels import EphemeralValue
from langgraph.graph import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from src.prompts import TASK_DESCRIPTION_PREFIX



# Limits
max_concurrent_research_units = 3
max_researcher_iterations = 3
load_dotenv(override=True)


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



class AgentState(TypedDict, Generic[ResponseT]):
    """State schema for the agent."""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]

class Todo(TypedDict):
    """A structured task item for tracking progress through complex workflows.

    Attributes:
        content: Short, specific description of the task
        status: Current state - pending, in_progress, or completed
    """

    content: str
    status: Literal["pending", "in_progress", "completed"]

class DeepAgentState(AgentState):
    """Extended agent state that includes task tracking and virtual file system.

    Inherits from LangGraph's AgentState and adds:
    - todos: List of Todo items for task planning and progress tracking
    - files: Virtual file system stored as dict mapping filenames to content
    """

    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]


class SubAgent(TypedDict):
    """Configuration for a specialized sub-agent."""

    name: str # name used by Main agent
    description: str
    prompt: str
    tools: NotRequired[list[str]]


def _create_task_tool(tools, subagents: list[SubAgent], model, state_schema):
    """Create a task delegation tool that enables context isolation through sub-agents.

    This function implements the core pattern for spawning specialized sub-agents with
    isolated contexts, preventing context clash and confusion in complex multi-step tasks.

    Args:
        tools: List of available tools that can be assigned to sub-agents
        subagents: List of specialized sub-agent configurations
        model: The language model to use for all agents
        state_schema: The state schema (typically DeepAgentState)

    Returns:
        A 'task' tool that can delegate work to specialized sub-agents
    """
    print(f"_create_task_tool... tools= {tools}. subagents:- {subagents}")
    # output
    # [StructuredTool(name='web_search', description='Search the web for information on a specific topic. This tool performs web searches and returns relevant results\nfor the given query.
    # Use this when you need to gather information from\nthe internet about any topic.', args_schema=<class 'langchain_core.utils.pydantic.web_search'>,
    # func=<function web_search at 0x0000028591CADD00>)].
    #
    # subagents:- [{'name': 'research-agent', 'description':
    # 'Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.',
    # 'prompt': 'You are a researcher. Research the topic provided to you. IMPORTANT: Just make a single call to the web_search tool and use the result provided by the tool to
    # answer the provided topic.',
    # 'tools': ['web_search']}]


    # Create agent registry
    agents = {}

    # Build tool name mapping for selective tool assignment
    tools_by_name = {}
    for tool_ in tools:
        if not isinstance(tool_, BaseTool):
            tool_ = tool(tool_)
        tools_by_name[tool_.name] = tool_

    print(f"_create_task_tool... tools_by_name= {tools_by_name}.")

    # Create specialized sub-agents based on configurations
    for _agent in subagents:
        print(f"_create_task_tool... _agent= {_agent}.")
        if "tools" in _agent:
            # Use specific tools if specified
            _tools = [tools_by_name[t] for t in _agent["tools"]]
        else:
            # Default to all tools
            _tools = tools
        agents[_agent["name"]] = create_agent(   # updated 1.0
            model, system_prompt=_agent["prompt"], tools=_tools, state_schema=state_schema
        )
        print(f"_create_task_tool... _agent-name= {agents[_agent["name"]]}.")

    # Generate description of available sub-agents for the tool description
    other_agents_string = [
        f"- {_agent['name']}: {_agent['description']}" for _agent in subagents
    ]
    print(f"_create_task_tool... other_agents_string= {other_agents_string}.")
    # _create_task_tool... other_agents_string= ['- research-agent: Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.'].

    @tool(description=TASK_DESCRIPTION_PREFIX.format(other_agents=other_agents_string))
    def task(
        description: str,
        subagent_type: str,
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Delegate a task to a specialized sub-agent with isolated context.

        This creates a fresh context for the sub-agent containing only the task description,
        preventing context pollution from the parent agent's conversation history.
        """
        print(f"Inside tool:agents ={agents}")

        # Validate requested agent type exists
        if subagent_type not in agents:
            return f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"

        # Get the requested sub-agent
        sub_agent = agents[subagent_type]
        print(f"task... sub_agent= {sub_agent}.")

        # Create isolated context with only the task description
        # This is the key to Context isolation - no parent history
        # "Research MCP overview"
        state["messages"] = [{"role": "user", "content": description}]

        # Execute the sub-agent in isolation
        result = sub_agent.invoke(state)
        print(f"task... result= {result}.")

        # Return results to parent agent via Command state update
        return Command(
            update={
                "files": result.get("files", {}),  # Merge any file changes
                "messages": [
                    # Sub-agent result becomes a ToolMessage in parent context
                    ToolMessage(
                        result["messages"][-1].content, tool_call_id=tool_call_id
                    )
                ],
            }
        )

    return task


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
        Search results from the search engine.

    Example:
        web_search("machine learning applications in healthcare")
    """
    print(f"web_search... .")
    return search_result


# Add mock research instructions
SIMPLE_RESEARCH_INSTRUCTIONS = """You are a researcher. Research the topic provided to you. IMPORTANT: Just make a single call to the web_search tool and use the result provided by the tool to answer the provided topic."""

# Create research sub-agent # Define Agent Configurations
research_sub_agent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "prompt": SIMPLE_RESEARCH_INSTRUCTIONS,
    "tools": ["web_search"],
}

# Create agent using create_react_agent directly
model = init_chat_model(model="anthropic:claude-haiku-4-5-20251001", temperature=0.0)

# Tools for sub-agent
sub_agent_tools = [web_search] # Create Tool Registry

# Create task tool to delegate tasks to sub-agents
# Dynamic Delegation Tool
task_tool = _create_task_tool(
    sub_agent_tools, [research_sub_agent], model, DeepAgentState
)

# Tools
delegation_tools = [task_tool]

# Create agent with system prompt
agent = create_agent(
    model,
    delegation_tools,
    system_prompt=SUBAGENT_USAGE_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
        date=datetime.now().strftime("%a %b %#d, %Y")
    ),
    state_schema=DeepAgentState,
)



def main():
    agent.get_graph(xray=1).print_ascii()
    agent.get_graph(xray=1).draw_mermaid_png(
        output_file_path="src/graph_flow_diagram/" + Path(__file__).stem + ".png")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Give me an overview of Model Context Protocol (MCP).",
                }
            ],
        }
    )

    print(f"result : {result}")
    format_messages(result["messages"])


# uv run -m src.sub_agents
if __name__ == '__main__':
    main()
