"""
Supabase Multi-Agent System
===========================

How it works:

    You type a request
          |
          v
    SUPERVISOR  -> asks the LLM: "which agent should handle this?"
          |
          v
    One of: CREATE / READ / UPDATE / DELETE agent  (or FINISH for chit-chat)
          |
          v
    The agent calls its database tool -> Supabase
          |
          v
    Answer is printed

Settings live in .env.
"""

import json
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from supabase import create_client


# =====================================================
# STEP 1: Load settings from .env
# =====================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "enterprise_customer_operations")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Please set SUPABASE_URL and SUPABASE_KEY in the .env file")


# =====================================================
# STEP 2: Connect to the LLM (Ollama) and the database (Supabase)
# =====================================================

# The local LLM shared by the supervisor and every agent.
# temperature=0 makes answers predictable (no randomness).
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

# Supabase client. table() gives us the table we read and write.
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def table():
    """Return the Supabase table configured in .env."""
    return supabase.table(SUPABASE_TABLE)


# =====================================================
# STEP 3: Database tools
#
# Tools are plain Python functions the agents are allowed to call.
# The @tool decorator lets the LLM see the function name, its
# arguments and its docstring, so it knows when and how to use it.
# =====================================================

# The only values the table should contain for these columns.
ALLOWED_VALUES = {
    "priority": ["low", "medium", "high", "critical"],
    "resolution_status": ["open", "in_progress", "resolved"],
    "account_status": ["active", "inactive"],
}


def check_allowed_values(fields: dict) -> str:
    """Return an error message if a field has a value that is not allowed, else ""."""
    for column, allowed in ALLOWED_VALUES.items():
        value = fields.get(column)
        if value and value not in allowed:
            return f"Error: {column} must be one of {allowed}."
    return ""


def remove_empty(fields: dict) -> dict:
    """Drop fields the user did not give (empty strings), so we don't overwrite data."""
    return {key: value for key, value in fields.items() if value != ""}


def success(operation: str, records: list) -> str:
    """Format a tool result as JSON text for the LLM to read."""
    return json.dumps({"success": True, "operation": operation, "records": records}, default=str)


@tool
def create_record(
    customer_name: str,
    company_name: str,
    email: str,
    request_description: str,
    industry: str = "",
    service_type: str = "",
    assigned_team: str = "",
    priority: str = "medium",
    resolution_status: str = "open",
    account_status: str = "active",
    contract_value: float = 0,
) -> str:
    """Insert a new customer operations request.

    priority: low, medium, high or critical.
    resolution_status: open, in_progress or resolved.
    account_status: active or inactive.
    """
    if not customer_name.strip():
        return "Error: customer_name cannot be empty."

    new_row = remove_empty({
        "customer_name": customer_name,
        "company_name": company_name,
        "email": email,
        "request_description": request_description,
        "industry": industry,
        "service_type": service_type,
        "assigned_team": assigned_team,
        "priority": priority,
        "resolution_status": resolution_status,
        "account_status": account_status,
        "contract_value": contract_value,
    })

    error = check_allowed_values(new_row)
    if error:
        return error

    try:
        result = table().insert(new_row).execute()
        return success("create", result.data)
    except Exception as exc:
        return f"Create failed: {exc}"


@tool
def read_records(
    record_id: int = 0,
    priority: str = "",
    resolution_status: str = "",
    assigned_team: str = "",
) -> str:
    """Read customer operations requests.

    record_id = 0 returns many records. You can also filter by
    priority (low, medium, high, critical), resolution_status
    (open, in_progress, resolved) or assigned_team.
    """
    try:
        query = table().select("*")

        # Add a filter only for the values the user asked about.
        if record_id > 0:
            query = query.eq("id", record_id)
        if priority:
            query = query.eq("priority", priority)
        if resolution_status:
            query = query.eq("resolution_status", resolution_status)
        if assigned_team:
            query = query.ilike("assigned_team", f"%{assigned_team}%")  # partial match

        result = query.order("id").limit(100).execute()
        return success("read", result.data)
    except Exception as exc:
        return f"Read failed: {exc}"


@tool
def update_record(
    record_id: int,
    priority: str = "",
    resolution_status: str = "",
    account_status: str = "",
    assigned_team: str = "",
    service_type: str = "",
    request_description: str = "",
    contract_value: float = -1,
) -> str:
    """Change some fields of one record, found by its ID.

    Leave a field empty (or contract_value at -1) to keep its current value.
    """
    if record_id <= 0:
        return "Error: a valid record ID is required."

    changes = remove_empty({
        "priority": priority,
        "resolution_status": resolution_status,
        "account_status": account_status,
        "assigned_team": assigned_team,
        "service_type": service_type,
        "request_description": request_description,
    })
    if contract_value >= 0:
        changes["contract_value"] = contract_value

    if not changes:
        return "Error: no fields to update were given."

    error = check_allowed_values(changes)
    if error:
        return error

    try:
        result = table().update(changes).eq("id", record_id).execute()
        return success("update", result.data)
    except Exception as exc:
        return f"Update failed: {exc}"


@tool
def delete_record(record_id: int) -> str:
    """Delete one record by its ID.

    The person at the terminal must confirm by typing the ID back.
    """
    if record_id <= 0:
        return "Error: a valid record ID is required."

    # 1. Find the record so we can show the user what will be deleted.
    try:
        found = (
            table()
            .select("id, customer_name, company_name, request_description")
            .eq("id", record_id)
            .execute()
        )
    except Exception as exc:
        return f"Delete failed: {exc}"

    if not found.data:
        return f"No record with ID {record_id} was found. Nothing was deleted."

    # 2. Ask the human. The LLM cannot approve a delete on its own.
    row = found.data[0]
    print(f"\n[APPROVAL REQUIRED] Delete record {row['id']}: "
          f"{row['customer_name']} ({row['company_name']}) - {row['request_description']}")
    answer = input(f"Type {record_id} to confirm, anything else to cancel: ").strip()

    if answer != str(record_id):
        return f"Deletion of record {record_id} was cancelled by the user."

    # 3. Delete it.
    try:
        result = table().delete().eq("id", record_id).execute()
        return success("delete", result.data)
    except Exception as exc:
        return f"Delete failed: {exc}"


# =====================================================
# STEP 4: Build the agents
#
# An agent = the LLM + instructions + the tools it is allowed to use.
# Each agent only gets the one tool it needs.
# =====================================================

# Rules every agent must follow, added to every agent's instructions.
COMMON_RULES = """
Rules:
- Only use your tool; never invent data or field values.
- Report only what the tool actually returned.
- If the tool returns an error, tell the user the error.
- If required information is missing, ask the user for it."""

AGENTS = {
    "CREATE": create_agent(
        model=llm,
        tools=[create_record],
        system_prompt="You add new customer operations requests using create_record." + COMMON_RULES,
    ),
    "READ": create_agent(
        model=llm,
        tools=[read_records],
        system_prompt="You find and show customer operations requests using read_records." + COMMON_RULES,
    ),
    "UPDATE": create_agent(
        model=llm,
        tools=[update_record],
        system_prompt=(
            "You change existing customer operations requests using update_record. "
            "You need the record ID and only change the fields the user asked for." + COMMON_RULES
        ),
    ),
    "DELETE": create_agent(
        model=llm,
        tools=[delete_record],
        system_prompt=(
            "You delete customer operations requests using delete_record. "
            "You need the exact record ID. The user confirms the deletion "
            "in the terminal; report whether it was deleted or cancelled." + COMMON_RULES
        ),
    ),
}


# =====================================================
# STEP 5: The LangGraph workflow (supervisor + agents)
# =====================================================

class State(TypedDict):
    """Data passed between the steps of the graph."""
    user_request: str  # what the user typed
    route: str         # which agent the supervisor picked
    result: str        # the final answer


SUPERVISOR_PROMPT = """You are the supervisor of a system that manages enterprise
customer operations requests stored in a database.

Pick exactly ONE route for the user's request:
- CREATE : add a new record
- READ   : show / list / find records
- UPDATE : change an existing record
- DELETE : remove a record
- FINISH : greetings or general questions

User request: {request}

Reply with only the route name."""


def supervisor(state: State) -> dict:
    """Ask the LLM which agent should handle the request."""
    reply = llm.invoke(SUPERVISOR_PROMPT.format(request=state["user_request"]))
    route = reply.content.strip().upper()

    # If the LLM answers something unexpected, treat it as general chat.
    if route not in AGENTS and route != "FINISH":
        route = "FINISH"
    return {"route": route}


def run_agent(state: State) -> dict:
    """Run the agent the supervisor picked and return its final answer."""
    agent = AGENTS[state["route"]]
    response = agent.invoke({"messages": [HumanMessage(content=state["user_request"])]})
    return {"result": str(response["messages"][-1].content)}


def general_chat(state: State) -> dict:
    """Answer greetings / general questions without touching the database."""
    reply = llm.invoke(
        "Answer this general question. Do not claim any database "
        "operation was performed.\n" + state["user_request"]
    )
    return {"result": str(reply.content)}


def choose_next_step(state: State) -> str:
    """Conditional edge: after the supervisor, go to 'agent' or 'finish'."""
    return "finish" if state["route"] == "FINISH" else "agent"


# Draw the graph:  START -> supervisor -> (agent | finish) -> END
workflow = StateGraph(State)
workflow.add_node("supervisor", supervisor)
workflow.add_node("agent", run_agent)
workflow.add_node("finish", general_chat)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges("supervisor", choose_next_step, {"agent": "agent", "finish": "finish"})
workflow.add_edge("agent", END)
workflow.add_edge("finish", END)

graph = workflow.compile()


# =====================================================
# STEP 6: Chat loop in the terminal
# =====================================================

def ask(request: str) -> dict:
    """Send one request through the graph and return {route, result}."""
    return graph.invoke({"user_request": request, "route": "", "result": ""})


def main():
    """Read requests from the keyboard until the user types 'exit'."""
    print("=" * 60)
    print("SUPABASE MULTI-AGENT SYSTEM")
    print(f"Table: {SUPABASE_TABLE}   |   LLM: {OLLAMA_MODEL}")
    print("=" * 60)
    print("\nTry:")
    print("  Show all open high priority requests")
    print("  Update record 3 resolution status to in_progress")
    print("  Delete record 16")
    print("\nType 'exit' to quit.\n")

    while True:
        request = input("You: ").strip()

        if request.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not request:
            continue

        try:
            answer = ask(request)
            print(f"\n[{answer['route']} agent]")
            print(answer["result"], "\n")
        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()
