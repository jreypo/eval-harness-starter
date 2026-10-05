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

LATEST = results/latest
BASELINE = results/baseline
DEMO = results/demo

# Run an eval only if its latest result is missing, so `make compare` works on
# a fresh clone without re-running suites you just ran.
$(LATEST)/rag.json:
	$(MAKE) eval-rag
$(LATEST)/agent.json:
	$(MAKE) eval-agent

# Canary analysis: latest vs baseline. Exits non-zero if either gate fails.
compare: $(LATEST)/rag.json $(LATEST)/agent.json
	@rc=0; \
	for s in rag agent; do \
		$(EVALH) compare $(BASELINE)/$$s.json $(LATEST)/$$s.json || rc=1; echo; \
	done; exit $$rc

# Promote the latest results to the versioned baseline. Manual on purpose.
baseline: $(LATEST)/rag.json $(LATEST)/agent.json
	cp $(LATEST)/rag.json $(LATEST)/agent.json $(BASELINE)/
	@echo "baseline updated; commit results/baseline/ to make it official"

# The payoff. Replay the "regressed" fixtures (same model name, new snapshot),
# then compare against the baseline. The aggregate barely moves; the gate must
# still FAIL on paired flips. This target succeeds only if both gates fail.
demo-regression:
	@echo "== Candidate: same model name, silently updated snapshot (fixtures/*/regressed) =="
	@EVAL_FIXTURES_VARIANT=regressed $(EVALH) run rag --out $(DEMO)/rag.json > /dev/null
	@EVAL_FIXTURES_VARIANT=regressed $(EVALH) run agent --out $(DEMO)/agent.json > /dev/null
	@for s in rag agent; do \
		echo; $(EVALH) compare $(BASELINE)/$$s.json $(DEMO)/$$s.json; rc=$$?; \
		if [ $$rc -ne 1 ]; then echo "demo-regression: expected gate FAIL (exit 1) for $$s, got exit $$rc"; exit 1; fi; \
	done
	@echo
	@echo "Gates failed as expected: aggregate pass rates are unchanged, but paired"
	@echo "flips name the regression cases that broke. That is what the gate is for."
