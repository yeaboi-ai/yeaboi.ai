"""Per-tool risk classification — the source of truth for human-in-the-loop gating.

Every ``@tool`` registered in ``get_tools()`` MUST have a row here. WRITE
tools mutate external systems (Jira, Azure DevOps, Confluence, Notion) and are
routed through the ``human_review`` graph node for explicit user confirmation
before they execute; READ tools auto-execute via the ToolNode.

Why a plain dict and not LangChain tool metadata/tags: a central registry is
verifiable in one place (``tests/unit/tools/test_risk.py`` asserts two-way
equality against the AST-discovered ``@tool`` set, so a new tool cannot ship
unclassified), and this module stays importable with zero heavy dependencies —
preserving the lazy-import property of ``tools/__init__.py``.

# See docs: "Guardrails" — human-in-the-loop pattern (Tool layer)
# See docs: "Tools" — tool types, risk levels
"""

from __future__ import annotations

from yeaboi._compat import StrEnum


class ToolRisk(StrEnum):
    """Risk level of an agent tool.

    READ  — pure lookup; auto-executes via the ToolNode.
    WRITE — mutates an external system; the graph pauses in ``human_review``
            and only proceeds after the user confirms.
    """

    READ = "read"
    WRITE = "write"


TOOL_RISK: dict[str, ToolRisk] = {
    # GitHub — read-only surface (no write tools are registered)
    "github_read_repo": ToolRisk.READ,
    "github_read_file": ToolRisk.READ,
    "github_list_issues": ToolRisk.READ,
    "github_read_readme": ToolRisk.READ,
    # Azure DevOps
    "azdevops_read_repo": ToolRisk.READ,
    "azdevops_read_file": ToolRisk.READ,
    "azdevops_list_work_items": ToolRisk.READ,
    "azdevops_read_board": ToolRisk.READ,
    "azdevops_fetch_velocity": ToolRisk.READ,
    "azdevops_fetch_active_iteration": ToolRisk.READ,
    "azdevops_create_epic": ToolRisk.WRITE,
    "azdevops_create_story": ToolRisk.WRITE,
    "azdevops_create_iteration": ToolRisk.WRITE,
    # Local filesystem (read-only; path access is governed by the fs sandbox)
    "read_codebase": ToolRisk.READ,
    "read_local_file": ToolRisk.READ,
    "load_project_context": ToolRisk.READ,
    # Pure/LLM helpers
    "detect_bank_holidays": ToolRisk.READ,
    "estimate_complexity": ToolRisk.READ,
    "generate_acceptance_criteria": ToolRisk.READ,
    # Jira
    "jira_read_board": ToolRisk.READ,
    "jira_fetch_velocity": ToolRisk.READ,
    "jira_fetch_active_sprint": ToolRisk.READ,
    "jira_create_epic": ToolRisk.WRITE,
    "jira_create_story": ToolRisk.WRITE,
    "jira_create_sprint": ToolRisk.WRITE,
    # Linear
    "linear_read_board": ToolRisk.READ,
    "linear_fetch_velocity": ToolRisk.READ,
    "linear_fetch_active_sprint": ToolRisk.READ,
    "linear_create_epic": ToolRisk.WRITE,
    "linear_create_story": ToolRisk.WRITE,
    "linear_create_sprint": ToolRisk.WRITE,
    # Trello
    "trello_read_board": ToolRisk.READ,
    "trello_fetch_active_sprint": ToolRisk.READ,
    "trello_create_epic": ToolRisk.WRITE,
    "trello_create_story": ToolRisk.WRITE,
    "trello_create_sprint": ToolRisk.WRITE,
    # Confluence
    "confluence_search_docs": ToolRisk.READ,
    "confluence_read_page": ToolRisk.READ,
    "confluence_read_space": ToolRisk.READ,
    "confluence_create_page": ToolRisk.WRITE,
    "confluence_update_page": ToolRisk.WRITE,
    # Notion
    "notion_search_pages": ToolRisk.READ,
    "notion_read_page": ToolRisk.READ,
    "notion_read_database": ToolRisk.READ,
    "notion_create_page": ToolRisk.WRITE,
    "notion_update_page": ToolRisk.WRITE,
    # Team learning (DB reads)
    "analyze_team_history": ToolRisk.READ,
    "compare_plan_to_actuals": ToolRisk.READ,
}


#: Which connection a tool talks to — the key a plan's ``session_integrations``
#: names. ``""`` is a local or LLM helper that no integration gates. Every
#: TOOL_RISK row has one (tests/unit/tools/test_risk.py asserts it).
TOOL_INTEGRATION: dict[str, str] = {
    name: (
        "github"
        if name.startswith("github_")
        else "azdevops"
        if name.startswith("azdevops_")
        else "jira"
        if name.startswith("jira_")
        else "linear"
        if name.startswith("linear_")
        else "trello"
        if name.startswith("trello_")
        else "confluence"
        if name.startswith("confluence_")
        else "notion"
        if name.startswith("notion_")
        else ""
    )
    for name in TOOL_RISK
}


def integration_of(tool_name: str) -> str:
    """The connection key ``tool_name`` needs, or ``""`` when none does."""
    return TOOL_INTEGRATION.get(tool_name, "")


def tool_allowed(tool_name: str, integrations=None) -> bool:
    """Whether a plan restricted to ``integrations`` (None = unrestricted) may run ``tool_name``."""
    if integrations is None:
        return True
    key = integration_of(tool_name)
    return not key or key in set(integrations)


def allowed_tools(tools, integrations=None) -> list:
    """The subset of ``tools`` a plan restricted to ``integrations`` may bind; everything when None."""
    if integrations is None:
        return list(tools)
    return [tool for tool in tools if tool_allowed(tool.name, integrations)]


def high_risk_tool_names() -> frozenset[str]:
    """Return the names of all WRITE tools — the human-review gate set."""
    return frozenset(name for name, risk in TOOL_RISK.items() if risk is ToolRisk.WRITE)
