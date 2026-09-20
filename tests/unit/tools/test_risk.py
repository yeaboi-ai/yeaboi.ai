"""Risk-registry sync check — every @tool must be classified in tools/risk.py.

Mirrors test_tools_registry.py's AST discovery (no imports, no side effects)
and the surface-parity style of two-way set equality: a newly added ``@tool``
fails here until it gets an explicit READ/WRITE row, so a write tool can never
ship silently ungated; a removed tool rots the registry loudly.

# See docs: "Guardrails" — human-in-the-loop pattern (Tool layer)
"""

from yeaboi.tools.risk import TOOL_INTEGRATION, TOOL_RISK, ToolRisk, allowed_tools, high_risk_tool_names, tool_allowed

from .test_tools_registry import _discover_all_tools

_EXPECTED_WRITES = {
    "jira_create_epic",
    "jira_create_story",
    "jira_create_sprint",
    "confluence_create_page",
    "confluence_update_page",
    "notion_create_page",
    "notion_update_page",
    "azdevops_create_epic",
    "azdevops_create_story",
    "azdevops_create_iteration",
    "linear_create_epic",
    "linear_create_story",
    "linear_create_sprint",
    "trello_create_epic",
    "trello_create_story",
    "trello_create_sprint",
}


class TestRiskRegistryCoverage:
    def test_every_tool_is_classified(self):
        """Two-way equality: registry keys == AST-discovered @tool names."""
        discovered = _discover_all_tools()
        missing = set(discovered) - set(TOOL_RISK)
        stale = set(TOOL_RISK) - set(discovered)
        assert not missing, (
            f"@tool functions without a risk classification: {sorted(missing)} — "
            f"add each to TOOL_RISK in src/yeaboi/tools/risk.py "
            f"(source files: { {n: discovered[n] for n in sorted(missing)} })"
        )
        assert not stale, f"TOOL_RISK rows for tools that no longer exist: {sorted(stale)} — remove them from risk.py"

    def test_every_row_is_a_toolrisk(self):
        assert all(isinstance(risk, ToolRisk) for risk in TOOL_RISK.values())


class TestHighRiskDerivation:
    def test_write_tools_are_exactly_the_expected_set(self):
        """All external-system mutations are WRITE; adding one here is a conscious act."""
        assert high_risk_tool_names() == frozenset(_EXPECTED_WRITES)

    def test_reads_do_not_leak_into_the_gate(self):
        assert "read_codebase" not in high_risk_tool_names()
        assert "github_read_repo" not in high_risk_tool_names()
        assert "azdevops_read_board" not in high_risk_tool_names()


class TestIntegrationTable:
    def test_every_tool_names_its_integration(self):
        assert set(TOOL_INTEGRATION) == set(TOOL_RISK)
        assert TOOL_INTEGRATION["jira_create_epic"] == "jira"
        assert TOOL_INTEGRATION["github_read_repo"] == "github"
        assert TOOL_INTEGRATION["azdevops_read_board"] == "azdevops"
        assert TOOL_INTEGRATION["read_codebase"] == ""
        assert TOOL_INTEGRATION["estimate_complexity"] == ""

    def test_every_external_tool_has_an_integration(self):
        local = {n for n, key in TOOL_INTEGRATION.items() if not key}
        assert local == {
            "read_codebase",
            "read_local_file",
            "load_project_context",
            "detect_bank_holidays",
            "estimate_complexity",
            "generate_acceptance_criteria",
            "analyze_team_history",
            "compare_plan_to_actuals",
        }

    def test_tool_allowed(self):
        assert tool_allowed("jira_read_board", None)
        assert tool_allowed("jira_read_board", ["jira"])
        assert not tool_allowed("jira_read_board", ["notion"])
        assert tool_allowed("read_codebase", [])

    def test_allowed_tools_filters_by_name(self):
        class T:
            def __init__(self, name):
                self.name = name

        tools = [T("jira_read_board"), T("notion_read_page"), T("read_codebase")]
        assert [t.name for t in allowed_tools(tools, ["notion"])] == ["notion_read_page", "read_codebase"]
        assert allowed_tools(tools, None) == tools
