# Cost tiers mirror a CI pipeline: cheap checks on every commit, expensive
# model-backed suites on PRs and nightly. Everything is offline by default.

UV ?= uv
EVALH = $(UV) run evalh

.PHONY: test lint eval-retrieval eval-rag calibrate fixtures \
	eval-agent eval-reference compare baseline demo-regression

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check
	$(UV) run ruff format --check

eval-retrieval:
	$(EVALH) run rag --retrieval-only

eval-rag:
	$(EVALH) run rag

calibrate:
	$(EVALH) calibrate rag-judge

# Regenerate the synthetic stub fixtures from the plan in scripts/gen_fixtures.py.
fixtures:
	$(UV) run python scripts/gen_fixtures.py

eval-agent:
	$(EVALH) run agent

# The reference agent must score 100%. If it does not, a task or grader is wrong.
eval-reference:
	$(EVALH) run agent --agent reference --k 3 --require-pass-rate 1.0

compare baseline demo-regression:
	@echo "$@: not implemented yet" && exit 1
