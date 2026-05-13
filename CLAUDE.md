# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project shape

Educational "deep agents from scratch" walkthrough built on LangGraph + LangChain 1.x. Each file in `src/` is a **standalone runnable example** demonstrating one agent-engineering primitive — they are not meant to import each other beyond the shared `prompts.py` and `utils.py`. Expect deliberate duplication: `DeepAgentState`, `file_reducer`, `Todo`, and the mock `web_search` are redefined in several modules so each example reads top-to-bottom on its own.

Python `>=3.12`, dependencies managed with `uv` (see `uv.lock`, `pyproject.toml`). No test suite, no lint config.

## Running examples

Each module is invoked as a script via its footer comment `# uv run -m src.<module>`:

```powershell
uv run -m src.file_tools
uv run -m src.planning_TODO_lists
uv run -m src.stateful_tools
uv run -m src.sub_agents
```

When run, every module also does two side effects: `agent.get_graph().print_ascii()` to stdout and `draw_mermaid_png()` to `src/graph_flow_diagram/<module>.png`. The PNG render requires network access (mermaid.ink) — failures there are unrelated to agent logic.

### Required env vars

Copy `.env.example` to `.env` and set:
- `OPENAI_API_KEY` — used by `stateful_tools.py` (`openai:gpt-4o-mini`)
- `ANTHROPIC_API_KEY` — used by every other module (`anthropic:claude-haiku-4-5-20251001`)
- `TAVILY_API_KEY` — referenced but the in-repo `web_search` tools are currently **mocked** (return a hardcoded MCP blurb); Tavily is wired in only as a dependency
- `LANGSMITH_*` — optional tracing

All modules call `load_dotenv(override=True)` at import time.

## What each module teaches

Treat the modules as a progression — later ones layer on earlier concepts.

- **`stateful_tools.py`** — Tools that read and write graph state via `InjectedState` / `InjectedToolCallId` and return `Command(update=...)`. Introduces the pattern of custom reducers (`reduce_list`) on a `TypedDict` state.
- **`file_tools.py`** — Adds a virtual filesystem to state (`files: dict[str, str]` with `file_reducer`) plus `ls` / `read_file` / `write_file` tools. The agent's "memory" lives in state, not on disk.
- **`planning_TODO_lists.py`** — Adds `todos: list[Todo]` to state with `write_todos` / `read_todos`. Header comments contrast a "prompt-driven" agent (current impl: ReAct loop guided by `TODO_USAGE_INSTRUCTIONS`) against a hypothetical "graph-enforced" version with explicit planner/reflection nodes.
- **`sub_agents.py`** — `_create_task_tool` factory spawns isolated sub-agents (`research-agent` here) and exposes them through a single `task(description, subagent_type)` tool to the parent. Sub-agent context is reset to just the task description — this is the **context isolation** pattern.

`DeepAgentState` extends `langchain.agents.AgentState` (or a local re-declaration in `sub_agents.py`) with `todos` and `files`. The `file_reducer` is the key shared mechanism that lets multiple tool calls accumulate file writes.

## Shared infrastructure

- **`src/prompts.py`** — Single source of truth for tool descriptions (`LS_DESCRIPTION`, `READ_FILE_DESCRIPTION`, `WRITE_FILE_DESCRIPTION`, `WRITE_TODOS_DESCRIPTION`) and instruction blocks (`TODO_USAGE_INSTRUCTIONS`, `FILE_USAGE_INSTRUCTIONS`, `SUBAGENT_USAGE_INSTRUCTIONS`, `TASK_DESCRIPTION_PREFIX`, `RESEARCHER_INSTRUCTIONS`). When editing prompts, prefer changing this file over inlining in modules.
- **`src/utils.py`** — Rich-formatted display helpers (`format_messages`, `show_prompt`) and `stream_agent(agent, query, config)` async runner that prints per-node updates while accumulating the final state.

## Conventions to preserve

- Modules use `create_agent` from `langchain.agents` (LangChain 1.x API), not the older `create_react_agent`. Pass `state_schema=DeepAgentState` so injected-state tools type-check.
- Tools that mutate state must return `Command(update={...})` and include a `ToolMessage` for the corresponding `tool_call_id` — see `write_file` and `write_todos` for the pattern.
- Tools using `parse_docstring=True` rely on Google-style docstrings to derive arg schemas; don't drop the docstrings.
- The `web_search` in each file is a mock returning a fixed MCP paragraph. If you swap in real Tavily, update each module independently (they don't share the tool).
- The header comments (ASCII flow diagrams, "Prompt-driven vs Graph-enforced" commentary) are intentional teaching material — keep them when refactoring.
