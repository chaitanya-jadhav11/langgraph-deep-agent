from pathlib import Path
from typing import Literal, TypedDict, Annotated, Sequence, NotRequired, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.graph import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from src.utils import format_messages

load_dotenv(override=True)


class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    #remaining_steps: NotRequired[RemainingSteps]

def reduce_list(left: list | None, right: list | None) -> list:
    """Safely combine two lists, handling cases where either or both inputs might be None.

    Args:
        left (list | None): The first list to combine, or None.
        right (list | None): The second list to combine, or None.

    Returns:
        list: A new list containing all elements from both input lists.
               If an input is None, it's treated as an empty list.
    """
    if not left:
        left = []
    if not right:
        right = []
    return left + right

class CalcState(AgentState):
    """Graph State."""
    ops: Annotated[List[str], reduce_list]


@tool
def calculator_wstate(
        operation: Literal["add", "subtract", "multiply", "divide"],
        a: float,
        b: float,

        # below parameter: The LLM NEVER sees this parameter.
        # The runtime injects it internally.
        state: Annotated[CalcState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId]
):
    """Define a two-input calculator tool.

    Arg:
        operation (str): The operation to perform ('add', 'subtract', 'multiply', 'divide').
        a (float or int): The first number.
        b (float or int): The second number.

    Returns:
        result (float or int): the result of the operation
    Example
        Divide: result   = a / b
        Subtract: result = a - b
    """
    # Access state from tools
    print(f"access State from tool \n: {state["messages"]}")
    # ------------------------

    if operation == 'divide' and b == 0:
        return {"error": "Division by zero is not allowed."}

    # Perform calculation
    if operation == 'add':
        result = a + b
    elif operation == 'subtract':
        result = a - b
    elif operation == 'multiply':
        result = a * b
    elif operation == 'divide':
        result = a / b
    else:
        result = "unknown operation"
    ops = [f"({operation}, {a}, {b}),"]
    return Command(
        update={
            "ops": ops, # State update from tools
            "messages": [
                ToolMessage(f"{result}", tool_call_id=tool_call_id)
            ],
        }
    )
    """
    Output after tool execution.
        {
        "messages": [
            HumanMessage("What is 3 * 4?"),
            ToolMessage("12")
        ],
        "ops": [
            "(multiply, 3, 4)"
        ]
        }
    
       This enables: 
    
        long-running agents
        multi-agent collaboration
        planning systems
        scratchpads
        memory accumulation
        event sourcing
        audit logs
        execution traces
        shared context between nodes/tools
    
    """

SYSTEM_PROMPT = """
You are a helpful arithmetic assistant who is an expert at using a calculator. 
Return all text as plain text without Markdown math delimiters.
"""
model = init_chat_model(model="openai:gpt-4o-mini", temperature=0.0)
tools = [calculator_wstate]  # new tool

# Create agent
agent = create_agent(
    model,
    tools,
    system_prompt=SYSTEM_PROMPT,
    state_schema=CalcState,  # now defining state scheme
).with_config({"recursion_limit": 20})  #recursion_limit limits the number of steps the agent will run


agent.get_graph(xray=1).print_ascii()
agent.get_graph(xray=1).draw_mermaid_png(
    output_file_path="src/graph_flow_diagram/" + Path(__file__).stem + ".png")

def main():
    # Example usage
    result2 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is 3.1 * 4.2?",
                }
            ],
        }
    )

    format_messages(result2["messages"])

# uv run -m src.stateful_tools
if __name__ == '__main__':
    main()