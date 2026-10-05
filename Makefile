# Cost tiers mirror a CI pipeline: cheap checks on every commit, expensive
# model-backed suites on PRs and nightly. Everything is offline by default.

UV ?= uv
EVALH = $(UV) run evalh

.PHONY: test eval-retrieval eval-rag eval-agent eval-reference compare baseline demo-regression lint

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check
	$(UV) run ruff format --check

eval-retrieval eval-rag eval-agent eval-reference compare baseline demo-regression:
	@echo "$@: not implemented yet" && exit 1
