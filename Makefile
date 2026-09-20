# --- shared tooling (yeaboi-tooling, pinned by .tooling-rev) ------------------
#
# Copied verbatim from the tooling repo's bootstrap/Makefile.head. It clones the
# tooling repo to `.tooling/` at the pinned sha and includes the shared targets
# (wt-*, tooling-*, contracts-*). The clone happens at parse time and only when
# the pin and the checkout disagree, so the steady state is two file reads and
# no network — and a fresh `git worktree add`, which never populates a
# submodule, provisions itself on the first `make`.
#
# Bump the pin with `make tooling-bump` and commit `.tooling-rev`.

TOOLING      := .tooling
TOOLING_REV  := $(shell cat .tooling-rev 2>/dev/null | tr -d '[:space:]')
TOOLING_HAVE := $(shell cat $(TOOLING)/.git/tooling-rev 2>/dev/null | tr -d '[:space:]')

ifeq ($(TOOLING_REV),)
$(error missing .tooling-rev — this repo pins the shared tooling by commit sha)
endif
ifneq ($(TOOLING_REV),$(TOOLING_HAVE))
TOOLING_SYNC := $(shell bash scripts/tooling-sync.sh >&2 && echo ok)
ifneq ($(TOOLING_SYNC),ok)
$(error shared tooling could not be synced — see the [tooling] lines above)
endif
endif

# The include brings targets with it, and the first target in a makefile is the
# default goal. Name the goal explicitly so `make` with no arguments still
# prints help rather than cutting a worktree.
.DEFAULT_GOAL := help

include $(TOOLING)/mk/common.mk

# --- end shared tooling ------------------------------------------------------

UV := $(or $(shell command -v uv 2>/dev/null),$(HOME)/.local/bin/uv)

# `test` and `ship-gate` order their prerequisites deliberately (cheap checks
# first, unit before integration). `make -j` would run them concurrently, and
# two pytest processes in one worktree invent failures.
.NOTPARALLEL:

.PHONY: install dev test test-fast test-compat test-slow test-scoped test-v test-all lint format format-check security package-check preflight ship-gate run run-dry clean env pre-commit graph demo demo-render eval contract record smoke-test snapshot-update budget-report bump-patch bump-minor bump-major build publish help web-types dev-board dev-poker dev-deck dev-editable site-contract pr-feedback

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install uv (if missing) and project dependencies
	@command -v uv >/dev/null 2>&1 || (echo "Installing uv..." && curl -LsSf https://astral.sh/uv/install.sh | sh)
	$(UV) venv
	$(UV) pip install -e ".[dev]"

env: ## Copy .env.example to .env (won't overwrite existing)
	@if [ -f .env ]; then echo ".env already exists — skipping"; else cp .env.example .env && echo "Created .env from .env.example — fill in your keys"; fi

# Every unit lane runs through these three variables so the paths and the
# parallel flags cannot drift apart between the Makefile, CI and the hooks.
#
# UNIT_PATHS carries `tests/*.py` as well as `tests/unit/`. Those five files
# (215 tests) were run by no CI job at all for months — `test-fast` and `test`
# both name subdirectories, and only `test-all`, which nothing calls, picked
# them up. One of them had been failing that whole time.
UNIT_PATHS ?= tests/unit/ $(wildcard tests/test_*.py)
# A command-line `UNIT_PATHS=` beats `?=`, so CI handing over an empty selection
# would reach pytest with no paths at all — and pytest then falls through to
# pyproject's `testpaths = ["tests"]`, quietly collecting the integration,
# contract and golden suites as well. That is not the "degrades to the
# full unit lane" the callers assume, so resolve the empty case here instead.
UNIT_LANE = $(if $(strip $(UNIT_PATHS)),$(UNIT_PATHS),tests/unit/ $(wildcard tests/test_*.py))
SLOW_LANE = $(if $(strip $(SLOW_PATHS)),$(SLOW_PATHS),tests/integration/ tests/contract/)
SLOW_PATHS ?= tests/integration/ tests/contract/
# `--dist loadfile`, not the default `load`: it keeps every test in a file on one
# worker, so module-scoped fixtures, the shared `tmp_path` conventions and the
# handful of tests that bind a socket behave exactly as they do serially. The
# integration lane stays serial — `tests/integration/test_repl.py` monkeypatches
# ten-plus names and AGENTS.md forbids editing it.
PYTEST_PARALLEL ?= -n auto --dist loadfile

# The versions CI's non-required `compat` job covers — everything above the
# floor, which the required `unit` job already runs. Kept in step with
# ci.yml's matrix by tests/unit/test_python_floor.py.
COMPAT_PYTHONS ?= 3.11 3.12 3.13 3.14

test-fast: ## Unit tests only — the tight edit-test loop
	$(UV) run pytest $(UNIT_LANE) $(PYTEST_PARALLEL) --tb=short -q
	@echo "✓ Unit tests passed"

test-compat: ## Unit lane on every supported Python above the floor (what CI's `compat` job runs)
	@for v in $(COMPAT_PYTHONS); do \
		echo "── Python $$v ──"; \
		$(UV) run --isolated --extra dev --python $$v python -c "import sys; assert '.'.join(map(str,sys.version_info[:2])) == '$$v', sys.version" || exit 1; \
		$(UV) run --isolated --extra dev --python $$v pytest $(UNIT_LANE) $(PYTEST_PARALLEL) --tb=short -q || exit 1; \
	done
	@echo "✓ Unit lane passed on $(COMPAT_PYTHONS)"

test-slow: ## Integration + contract only — the half `test-fast` does not cover
	$(UV) run pytest $(SLOW_LANE) --tb=short
	@echo "✓ Integration & contract tests passed"

# One pytest process for 10,383 unit tests plus 562 integration ones took 408s,
# of which 310s was the unit half run serially — the same tests `test-fast`
# finishes in ~50s with `-n auto`. Expressed as prerequisites rather than two
# copied recipe lines so the paths and flags cannot drift from UNIT_LANE /
# SLOW_LANE / PYTEST_PARALLEL, which is the whole reason those variables exist.
# Same tests, same order, same split as CI's two test jobs.
test: test-fast test-slow ## Unit + integration + contract tests — full suite, no API keys needed
	@echo "✓ All tests passed"

# `python3`, not `$(UV) run`: scripts/test_scope.py imports the standard library
# only, and putting a dependency resolve in front of the hook that runs on every
# commit is how a fast gate becomes one people disable.
# The `||` is load-bearing: a non-zero exit inside `$$( )` is invisible to make,
# so a crash in the selector would silently become an empty path list. Fall back
# to the whole unit lane, the same direction CI falls back in.
test-scoped: ## Only the areas the working tree touches, plus the always-run guards
	@python3 scripts/test_scope.py --working-tree --explain || echo "scope failed — running the full unit lane"
	@paths=$$(python3 scripts/test_scope.py --working-tree --unit-paths) || paths=""; \
		[ -n "$$paths" ] || paths="$(UNIT_LANE)"; \
		$(UV) run pytest $$paths $(PYTEST_PARALLEL) --tb=short -q
	@echo "✓ Scoped unit tests passed"

test-v: ## Unit + integration + contract tests (verbose)
	$(UV) run pytest $(UNIT_LANE) $(SLOW_LANE) -v

test-all: ## Everything including golden evaluators (requires make eval separately for golden)
	$(UV) run pytest --ignore=tests/smoke --tb=short
	@echo "✓ Full test suite passed"

# `scripts/` is in the paths deliberately. It was not, and CI's required
# "Lint (ruff)" job runs this target — so a lint error in a script was caught by
# pre-commit and by nothing else, on a repo where the fleet, the release channel
# and the test selector all live in scripts/. Found the honest way: this very
# change tripped it.
lint: ## Lint with ruff
	$(UV) run ruff check src/ tests/ scripts/

format: ## Format with ruff
	$(UV) run ruff format src/ tests/ scripts/

# `make format` writes; this one asserts. CI's "Format check (ruff)" is a
# required status check and had no Makefile target, so the only way to fail it
# was to open the PR.
format-check: ## What CI's "Format check (ruff)" job runs — asserts, never writes
	$(UV) run ruff format --check src/ tests/ scripts/

# `security: lint`, not a second copy of `ruff check src/ tests/` — the SAST half
# of this target WAS that command, byte for byte, so `make lint security` linted
# twice. As a prerequisite it still runs standalone and make resolves it once.
security: lint ## Security scan — bandit (ruff S) SAST + dependency CVE audit
	@echo '→ SAST (ruff incl. flake8-bandit S rules; respects pyproject ignores) — ran above as make lint'
	@echo "→ Dependency CVE audit (pip-audit against the synced environment)"
	$(UV) run --with pip-audit pip-audit
	@echo "✓ Security scan passed"

# ── The ship gate ───────────────────────────────────────────────────────────
# `make test` proves the Python suite. It does not prove the other things
# CI checks — the format check above, the front-end bundles, the docs site,
# the golden evaluators and the wheel's contents. Those used to be discovered
# after the PR was already open.
#
# `preflight` runs only the ones this branch's diff actually needs, decided by
# scripts/test_scope.py — the same selector CI's `scope` job uses, whose third
# rule is that anything it cannot classify runs everything.

package-check: ## What CI's "Wheel declares its dependencies" job runs (uv build + assert)
	$(UV) build
	python3 scripts/check_wheel_deps.py

preflight: ## Run only the optional CI jobs this branch's diff needs (BASE=origin/main)
	python3 scripts/preflight.py --base $(BASE)

BASE ?= origin/main

# Fail-fast order: seconds-long checks first, then the suite, then the network
# audit, then the heavy optional jobs. This is what /ship runs, as ONE make
# invocation, so `lint` resolves once for both itself and `security`.
ship-gate: lint format-check test security preflight ## The full local gate /ship runs — everything CI will check
	@echo "✓ ship gate passed"

pre-commit: ## Point git at this worktree's own .githooks/ (repairs a stale shared hook)
	git config core.hooksPath .githooks
	@$(UV) run pre-commit uninstall >/dev/null 2>&1 || true
	@echo "hooks: .githooks/ per worktree (was one shared .git/hooks pinned to one venv)"

run: ## Run the yeaboi CLI (use ARGS="--flag" to pass arguments)
	$(UV) run yeaboi $(ARGS)

run-dry: ## Run the TUI with fake delays — no LLM calls
	$(UV) run yeaboi --dry-run $(ARGS)

eval: ## Run golden dataset evaluators
	$(UV) run pytest tests/golden/ -v

contract: ## Run contract tests (recorded API responses + LLM provider parsing)
	$(UV) run pytest tests/contract/ -v

smoke-test: ## Run smoke tests against live APIs (requires real credentials)
	$(UV) run pytest tests/smoke/ -v -m smoke

record: ## Re-record VCR cassettes against real APIs (requires API keys)
	$(UV) run pytest tests/ -v --record-mode=rewrite -m vcr

snapshot-update: ## Update syrupy snapshot baselines after intentional formatter changes
	$(UV) run pytest tests/unit/test_formatters.py --snapshot-update -v

budget-report: ## Show live prompt token counts for trend monitoring (runs token budget tests with -s)
	$(UV) run pytest tests/unit/test_token_budgets.py -v -s

# graph, demo and demo-render read this repo and write into a yeaboi-site
# checkout, which is where the website serves them from. Set YEABOI_SITE or keep
# the two repos side by side; scripts/_sibling_repos.py explains the resolution.
graph: ## Generate the agent graph PNG into the yeaboi-site checkout
	$(UV) run python scripts/generate_graph_png.py

demo: ## Re-record the terminal demo into the yeaboi-site checkout (requires agg: brew install agg)
	$(UV) run python scripts/record_demo.py

demo-render: ## Re-render the demo GIF from the committed cast (theme/size tweaks, no re-record)
	$(UV) run python scripts/record_demo.py --render-only

bump-patch: ## Bump the patch version in pyproject.toml (X.Y.Z -> X.Y.Z+1)
	$(UV) run python scripts/bump_version.py patch

bump-minor: ## Bump the minor version in pyproject.toml (X.Y.Z -> X.Y+1.0)
	$(UV) run python scripts/bump_version.py minor

bump-major: ## Bump the major version in pyproject.toml (X.Y.Z -> X+1.0.0)
	$(UV) run python scripts/bump_version.py major

# --- Front end — its own repo (yeaboi-frontend) -----------------------------
#
# There is no front-end build here any more. The bundles are built and published
# from yeaboi-frontend as the yeaboi-web-assets wheel, which is an ordinary
# dependency; what stays is the generation of the contracts that repo vendors.

web-types: ## Regenerate the contracts the front end and desktop vendor (enums + ui + connectors)
	uv run python scripts/gen_web_types.py
	uv run python scripts/gen_web_ui_contract.py
	uv run python scripts/gen_connectors_contract.py
	@echo "✓ commit contracts/web/ — yeaboi-frontend picks it up with 'make contracts-sync'"
	@echo "✓ commit contracts/v1/connectors.json — yeaboi-desktop picks it up the same way"

# The desktop app lives in yeaboi-desktop and installs a RELEASED wheel from
# PyPI; nothing about it is built here any more. What stays is one artefact it
# vendors: contracts/v1/routes_manifest.json, which the surface-parity suite
# reads to decide whether a capability reached the desktop. That repo
# regenerates it from its own renderer and fails when the two disagree.

# contracts/site.json is what the yeaboi-site repo vendors instead of reading
# this repo's pyproject.toml: the Python floor it advertises, the repo URL in its
# JSON-LD, the install target. tests/unit/test_site_contract.py asserts it is
# fresh on every lane. The website itself lives in yeaboi-site.
site-contract: ## Regenerate contracts/site.json from pyproject (the facts the website vendors)
	$(UV) run python scripts/gen_site_contract.py

dev-board: ## Seeded retro board for front-end development (prints the URL; :5173 unless the worktree has its own block)
	$(UV) run python scripts/dev_board.py

dev-poker: ## Seeded planning-poker board for front-end development (:5273 unless the worktree has its own block)
	$(UV) run python scripts/dev_poker.py

dev-deck: ## Seeded reporting slide deck for front-end development (:5373 unless the worktree has its own block)
	$(UV) run python scripts/dev_deck.py

dev-editable: ## Seeded correctable standup document for front-end development (:5473 unless the worktree has its own block)
	$(UV) run python scripts/dev_editable.py

build: ## Build sdist + wheel into dist/
	$(UV) build

publish: ## Publish to PyPI (use GitHub Actions for production releases)
	$(UV) publish

# --- PR feedback — the merge gate on unanswered review comments ---------------

# Five things comment on a PR here and nothing used to read any of it back. This
# reports what is still unanswered; the `pr-feedback` commit status posted by
# .github/workflows/pr-feedback.yml is what actually holds the merge.
pr-feedback: ## Report unanswered review feedback on a PR (PR=123, or the current branch's)
	@$(UV) run python scripts/pr_feedback.py $(if $(PR),--pr $(PR),)

.PHONY: fuzz
fuzz: ## Fuzz the live TUI in a pty (SEEDS=6 STEPS=120, or SEED=41 to replay one)
	@$(UV) run python scripts/tui_fuzz.py $(if $(SEED),--seed $(SEED),--seeds $(or $(SEEDS),6)) --steps $(or $(STEPS),120)


clean: ## Remove build artifacts and caches
	rm -rf .venv build dist .pytest_cache .ruff_cache *.egg-info src/*.egg-info bin
	find . -type d -name __pycache__ -exec rm -rf {} +
