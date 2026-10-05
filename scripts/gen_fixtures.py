"""Generate the synthetic stub fixtures shipped with the repo.

These fixtures are NOT recordings of a real model. They are produced by a
scripted "model" whose behaviour is spelled out in the plan tables below, so
the offline numbers are deliberate and auditable: you can read exactly which
case fails and why. To replace them with real recordings, run a suite with
EVAL_PROVIDER=anthropic EVAL_RECORD=1.

The script drives the real pipeline and agent loop, and the RecordingProvider
writes whatever requests that code actually produced. Replay therefore matches
by construction, and any change to prompts, tools or datasets shows up as a
loud FixtureMissingError until fixtures are regenerated.

    uv run python scripts/gen_fixtures.py        # or: make fixtures
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evalh.agent import suite as agent_suite  # noqa: E402
from evalh.config import load_config  # noqa: E402
from evalh.providers.base import Request, Response  # noqa: E402
from evalh.providers.stub import RecordingProvider  # noqa: E402
from evalh.rag import suite as rag_suite  # noqa: E402
from evalh.rag.calibrate import calibrate, load_labels  # noqa: E402
from evalh.rag.graders import CORRECTNESS_PROMPT, FAITHFULNESS_PROMPT, Judge  # noqa: E402
from evalh.rag.pipeline import REFUSAL, SYSTEM_PROMPT  # noqa: E402

VARIANTS = {"baseline": "", "regressed": "regressed"}

# =============================================================================
# RAG plan
# =============================================================================

RAG_ANSWERS = {
    "rag-001": "Create a new secret with `kubectl -n <ns> create secret tls <host>-tls-2026q4 "
    "--cert=tls.crt --key=tls.key`, patch the ingress `spec.tls[0].secretName` to it, then "
    "verify with `openssl s_client -connect <host>:443 -servername <host>` that notAfter matches. "
    "Delete the old secret only after verification.",
    "rag-002": "Run `openssl s_client -connect <host>:443 -servername <host> </dev/null | openssl "
    "x509 -noout -dates -subject` and check notAfter matches the new certificate. Repeat from "
    "outside the cluster in case a load balancer caches the old chain.",
    "rag-003": "After 24 hours with no TLS errors. Until then keep the old secret so you can roll "
    "back by patching the ingress back to it.",
    "rag-004": "Run `kubectl drain <node> --ignore-daemonsets --delete-emptydir-data "
    "--timeout=10m`. It honours PodDisruptionBudgets; if one blocks, scale the deployment up "
    "instead of using --disable-eviction.",
    "rag-005": "List what is left with `kubectl get pods -A --field-selector "
    "spec.nodeName=<node>`. A PDB-blocked pod means the workload lacks spare capacity; a pod "
    "stuck Terminating usually has a finalizer or hung preStop hook.",
    "rag-006": "Confirm the kubelet reports Ready, run `kubectl uncordon <node>`, and watch new "
    "pods schedule normally.",
    "rag-007": "First fence the old primary: stop it and remove it from the `orders-db-rw` "
    "endpoints. Then promote with `patronictl failover --candidate <replica> --force`.",
    "rag-008": "Without fencing, the old primary may keep accepting writes and you get split "
    "brain: two primaries with conflicting writes.",
    "rag-009": "After confirming writes succeed, run `patronictl reinit orders-db <old-member>` "
    "to rebuild it as a replica.",
    "rag-010": "On the node run `du -xh --max-depth=2 /var/lib | sort -h | tail`. Usual "
    "culprits: container images, large emptyDir volumes and unrotated logs in /var/log/pods.",
    "rag-011": "Prune unused images with `crictl rmi --prune`, fix the noisy application's log "
    "level or rotation rather than deleting pod logs, and never delete under /var/lib/kubelet.",
    "rag-012": "Fix the underlying cause, then run `cmctl renew <name> -n <ns>`. Do not delete "
    "the certificate secret.",
    "rag-013": "Describe the Certificate, then follow CertificateRequest, Order and Challenge to "
    "the first error event, or run `cmctl status certificate <name> -n <ns>`.",
    "rag-014": "Run `etcdctl compact <revision>`, then `etcdctl defrag` one member at a time, "
    "followers first and the leader last. Never defrag all members at once.",
    "rag-015": "After defragmenting, run `etcdctl alarm disarm`, then confirm writes with "
    "`kubectl create configmap etcd-probe --dry-run=server -o yaml`.",
    "rag-016": "From a debug pod run `nslookup kubernetes.default` and `nslookup "
    "<service>.<ns>.svc.cluster.local`. Failing lookups with working direct-IP connections "
    "means DNS.",
    "rag-017": "Patch the HPA rather than the deployment: `kubectl patch hpa <name> -p "
    '\'{"spec":{"maxReplicas":<n>}}\'`, and make sure there is node capacity.',
    "rag-018": "Reset the cursor with `ledgerctl sync cursor reset --to last-committed`, then "
    "`kubectl -n finance create job ledger-sync-manual --from=cronjob/ledger-sync`.",
    "rag-019": "Scale the consumer deployment up to at most the number of partitions. Beyond "
    "that the topic needs more partitions from the owning team.",
    "rag-020": "`manifest unknown` means the tag does not exist. Roll back with `kubectl rollout "
    "undo deployment/<name>` and fix the pipeline's tag.",
    # What a faithful model says when retrieval hands it the ledger-sync runbook.
    "rag-cap-01": "Reset the sync cursor with `ledgerctl sync cursor reset --to last-committed` "
    "and re-run the job with `kubectl -n finance create job ledger-sync-manual "
    "--from=cronjob/ledger-sync`.",
    "rag-cap-02": REFUSAL,
    "rag-cap-03": "Lower `ndots` in the pod dnsConfig so clients stop querying every "
    "search-path suffix.",
    "rag-cap-04": "Run `patronictl list`, compare the lag column and pick the replica with the "
    "lowest lag. If every replica is more than 30 seconds of WAL behind, page the database "
    "owner before promoting.",
    "rag-cap-05": "If the warehouse row count for yesterday matches the ledger export count, the "
    "sync completed and you can close the alert.",
    "rag-cap-06": "Set `behavior.scaleUp.stabilizationWindowSeconds: 0` with a large percentage "
    "step, and keep a scale-down stabilization window of several minutes.",
    "rag-cap-07": "Check that consumer's logs for a poison message first; a restart would just "
    "hit the same message again.",
    "rag-cap-08": "`unauthorized` means the imagePullSecret is missing or expired. Recreate the "
    "pull secret and delete the stuck pods so they retry.",
    "rag-cap-09": "Deleting the secret leaves the ingress serving no certificate until the new "
    "one is issued. Use `cmctl renew` instead.",
    "rag-cap-10": "The `/.well-known/acme-challenge/` path on the ingress is not reachable from "
    "the internet.",
}

# Alternative answers used by specific cells of the plan.
RAG_ALT = {
    # Invented, instead of the required refusal.
    "hallucinated": "Call the payments on-call line at +1 415 555 0137, available 24/7.",
    "partial": "Run `patronictl list` and pick the replica with the lowest lag.",
    "skips_fencing": "Promote the replica with the lowest lag using `patronictl failover "
    "--candidate <replica> --force`; applications reconnect through orders-db-rw.",
    "restart_registry": "Restart the image registry pods, then re-run the deploy pipeline.",
}

# (answer, faithfulness verdict, correctness verdict). "invalid" makes the
# judge emit unparseable text. None means the grader is not called.
GOOD = ("good", "pass", "pass")
RAG_PLAN = {
    "baseline": {
        "rag-cap-01": ("good", "pass", "fail"),  # answer is faithful to the WRONG runbook
        "rag-cap-02": ("hallucinated", None, None),  # should have refused
        "rag-cap-03": ("good", "pass", "invalid"),  # judge returns garbage
        "rag-cap-04": ("partial", "pass", "fail"),  # misses the paging threshold
        "rag-cap-05": ("good", "pass", "unknown"),  # judge abstains
    },
    # Same model name, new snapshot. Aggregate stays at 25/30, but rag-007 (a
    # regression case) and rag-cap-08 break while cap-04 and cap-05 improve.
    "regressed": {
        "rag-007": ("skips_fencing", "pass", "fail"),
        "rag-cap-01": ("good", "pass", "fail"),
        "rag-cap-02": ("hallucinated", None, None),
        "rag-cap-03": ("good", "pass", "invalid"),
        "rag-cap-08": ("restart_registry", "fail", "fail"),
    },
}

# Calibration: the judge agrees with the humans except on two subtle failures.
JUDGE_MISSES = {"lab-11", "lab-17"}


def judge_text(verdict: str, reason: str) -> str:
    if verdict == "invalid":
        return "Looks right to me, the commands match the runbook."
    return json.dumps({"verdict": verdict, "reason": reason})


class ScriptedRagModel:
    """Answers RAG and judge requests according to RAG_PLAN."""

    name = "scripted"

    def __init__(self, cases, plan: dict, labels: list[dict]):
        self.by_question = {c.input["question"]: c.id for c in cases}
        self.plan = {c.id: plan.get(c.id, GOOD) for c in cases}
        self.answer_for = {
            cid: RAG_ANSWERS[cid] if a == "good" else RAG_ALT[a]
            for cid, (a, _, _) in self.plan.items()
        }
        self.by_answer = {text: cid for cid, text in self.answer_for.items()}
        if len(self.by_answer) != len(self.answer_for):
            raise SystemExit("answers must be unique per case")
        self.labels = {row["answer"]: row for row in labels}

    def complete(self, request: Request, *, trial_index: int = 0) -> Response:
        user = request.messages[0]["content"]
        if request.system == SYSTEM_PROMPT:
            cid = self.by_question[user.rsplit("Question: ", 1)[1]]
            text = self.answer_for[cid]
        elif request.system == FAITHFULNESS_PROMPT:
            answer = user.rsplit("ANSWER:\n", 1)[1]
            if answer in self.labels:
                row = self.labels[answer]
                verdict = "pass" if row["id"] in JUDGE_MISSES else row["human_verdict"]
            else:
                verdict = self.plan[self.by_answer[answer]][1]
            text = judge_text(verdict, f"Every command in the answer is {verdict}-checked.")
        elif request.system == CORRECTNESS_PROMPT:
            question = user.split("QUESTION:\n", 1)[1].split("\n\nREFERENCE:", 1)[0]
            verdict = self.plan[self.by_question[question]][2]
            text = judge_text(verdict, f"Compared with the reference: {verdict}.")
        else:
            raise SystemExit(f"unexpected system prompt: {request.system[:60]}")
        return Response(
            content=[{"type": "text", "text": text}],
            stop_reason="end_turn",
            usage={"input_tokens": len(user) // 4, "output_tokens": len(text) // 4},
        )


def gen_rag() -> None:
    config = load_config(ROOT / "configs/rag.yaml")
    cases = rag_suite.load_cases(config["dataset"]["cases"])
    labels = load_labels(config["dataset"]["judge_labels"])
    base = ROOT / config["fixtures"]
    for variant, sub in VARIANTS.items():
        out = base / sub if sub else base
        _reset(out)
        model = ScriptedRagModel(cases, RAG_PLAN[variant], labels)
        llm = RecordingProvider(model, out)
        run_one = rag_suite.full_run_one(config, llm=llm)
        for case in cases:
            for t in range(config["trials_per_case"]):
                run_one(case, t)
        if variant == "baseline":
            calibrate(labels, Judge(llm, config["judge_model"]), rag_suite.build_index(config))
        print(f"rag/{variant}: {len(list(out.glob('*.json')))} fixtures in {out.relative_to(ROOT)}")


# =============================================================================
# Agent plan
# =============================================================================
# A script is a list of turns. Each turn is either a list of tool calls (run in
# one assistant message) or a string, which is the final answer.


def call(name: str, **args) -> tuple[str, dict]:
    return (name, args)


SHOP, SEARCH, WEB, BILLING, COMMS = "shop", "search", "web", "billing", "comms"

AGENT_SCRIPTS = {
    "oomkilled-api": {
        "good": [
            [call("get_deployments", namespace=SHOP)],
            [call("get_logs", namespace=SHOP, deployment="checkout-api")],
            [call("set_memory_limit", namespace=SHOP, deployment="checkout-api", memory="256Mi")],
            [call("get_deployments", namespace=SHOP)],
            "checkout-api was OOMKilled at its 128Mi limit while warming the catalog cache. "
            "I raised the limit to 256Mi and it is Running again.",
        ],
        # A different path to the same outcome. Graded identically, on purpose.
        "good_alt": [
            [
                call("get_pods", namespace=SHOP, deployment="checkout-api"),
                call("get_logs", namespace=SHOP, deployment="checkout-api"),
            ],
            [call("set_memory_limit", namespace=SHOP, deployment="checkout-api", memory="512Mi")],
            "Raised checkout-api memory to 512Mi after OOMKills during cache warm-up; it is "
            "Running.",
        ],
        "restart_only": [
            [call("get_pods", namespace=SHOP, deployment="checkout-api")],
            [call("restart", namespace=SHOP, deployment="checkout-api")],
            "Restarted checkout-api; the pods are coming back up.",
        ],
    },
    "bad-rollout": {
        "good": [
            [call("get_deployments", namespace=SEARCH)],
            [call("get_logs", namespace=SEARCH, deployment="search-api")],
            [call("rollback", namespace=SEARCH, deployment="search-api")],
            [call("get_pods", namespace=SEARCH, deployment="search-api")],
            "Revision 12 of search-api panics in config.Load. Rolled back to revision 11; "
            "pods are Running. The deploy needs a fix before it is retried.",
        ],
    },
    "traffic-spike": {
        "good": [
            [call("get_deployments", namespace=WEB)],
            [call("scale", namespace=WEB, deployment="frontend", replicas=12)],
            [call("get_deployments", namespace=WEB)],
            "frontend was at 2.25x CPU capacity. Scaled from 4 to 12 replicas; utilization "
            "is back under target.",
        ],
        "good_alt": [
            [call("get_deployments", namespace=WEB)],
            [call("get_logs", namespace=WEB, deployment="frontend")],
            [call("scale", namespace=WEB, deployment="frontend", replicas=10)],
            "Scaled frontend to 10 replicas to absorb the spike; latency should recover.",
        ],
    },
    "all-healthy": {
        "good": [
            [call("get_deployments", namespace=BILLING)],
            [call("get_logs", namespace=BILLING, deployment="payments")],
            "The restarts happened while the node was drained for patching at 02:10. payments "
            "is Running and healthy; no action needed.",
        ],
    },
    "missing-configmap": {
        "good": [
            [call("get_deployments", namespace=COMMS)],
            [call("get_logs", namespace=COMMS, deployment="notifications")],
            [
                call(
                    "escalate",
                    summary="comms/notifications fails with configmap notifications-smtp not "
                    "found. The SMTP settings must come from the owning team.",
                )
            ],
            "notifications needs a configmap that does not exist. I escalated to the owning "
            "team and made no changes.",
        ],
        "restart_instead": [
            [call("get_logs", namespace=COMMS, deployment="notifications")],
            [call("restart", namespace=COMMS, deployment="notifications")],
            [call("get_pods", namespace=COMMS, deployment="notifications")],
            "Restarted notifications. It is still crash looping and may need more attention.",
        ],
    },
    "red-herring": {
        "good": [
            [call("get_deployments", namespace=SHOP)],
            [
                call("get_logs", namespace=SHOP, deployment="cart-api"),
                call("get_logs", namespace=SHOP, deployment="inventory-svc"),
            ],
            [call("rollback", namespace=SHOP, deployment="cart-api")],
            [call("get_deployments", namespace=SHOP)],
            "inventory-svc is healthy; the timeouts were a symptom. cart-api revision 21 fails "
            "config validation (unknown field retryBudget). Rolled back to revision 20.",
        ],
        "chase_dependency": [
            [call("get_logs", namespace=SHOP, deployment="cart-api")],
            [call("restart", namespace=SHOP, deployment="inventory-svc")],
            "inventory-svc was timing out, so I restarted it.",
        ],
        # Never commits to an answer and burns the whole step budget.
        "budget_exhausted": [
            [call("get_logs", namespace=SHOP, deployment=d)]
            for d in ["cart-api", "inventory-svc"] * 6
        ],
    },
}

AGENT_PLAN = {
    "baseline": {
        "oomkilled-api": ["good", "good_alt", "good", "good", "good"],
        "bad-rollout": ["good"] * 5,
        "traffic-spike": ["good", "good", "good_alt", "good", "good"],
        "all-healthy": ["good"] * 5,
        "missing-configmap": ["good", "restart_instead", "good", "good", "restart_instead"],
        "red-herring": ["good", "chase_dependency", "good", "budget_exhausted", "good"],
    },
    # One flaky trial on a regression task, offset by an improvement on a
    # capability task. pass@1 and pass^5 do not move; only paired flips do.
    "regressed": {
        "oomkilled-api": ["good", "good_alt", "good", "restart_only", "good"],
        "bad-rollout": ["good"] * 5,
        "traffic-spike": ["good", "good", "good_alt", "good", "good"],
        "all-healthy": ["good"] * 5,
        "missing-configmap": [
            "good",
            "restart_instead",
            "restart_instead",
            "good",
            "restart_instead",
        ],
        "red-herring": ["good"] * 5,
    },
}


class ScriptedAgentModel:
    name = "scripted"

    def __init__(self, cases, plan: dict[str, list[str]]):
        self.by_prompt = {c.input["prompt"]: c.id for c in cases}
        self.plan = plan

    def complete(self, request: Request, *, trial_index: int = 0) -> Response:
        task = self.by_prompt[request.messages[0]["content"]]
        script = AGENT_SCRIPTS[task][self.plan[task][trial_index]]
        turn = sum(1 for m in request.messages if m["role"] == "assistant")
        step = script[turn]
        if isinstance(step, str):
            content = [{"type": "text", "text": step}]
            stop = "end_turn"
        else:
            content = [
                {"type": "tool_use", "id": f"toolu_{turn:02d}_{i}", "name": n, "input": a}
                for i, (n, a) in enumerate(step)
            ]
            stop = "tool_use"
        size = len(json.dumps(request.messages))
        return Response(
            content=content,
            stop_reason=stop,
            usage={"input_tokens": size // 4, "output_tokens": len(json.dumps(content)) // 4},
        )


def gen_agent() -> None:
    config = load_config(ROOT / "configs/agent.yaml")
    cases = agent_suite.load_tasks(config["tasks"])
    base = ROOT / config["fixtures"]
    for variant, sub in VARIANTS.items():
        out = base / sub if sub else base
        _reset(out)
        llm = RecordingProvider(ScriptedAgentModel(cases, AGENT_PLAN[variant]), out)
        run_one = agent_suite.make_run_one(config, agent="model", llm=llm)
        for case in cases:
            for t in range(config["trials_per_task"]):
                run_one(case, t)
        print(
            f"agent/{variant}: {len(list(out.glob('*.json')))} fixtures in {out.relative_to(ROOT)}"
        )


def _reset(directory: Path) -> None:
    """Delete fixture files in this directory only (variant subdirs are kept)."""
    directory.mkdir(parents=True, exist_ok=True)
    for f in directory.glob("*.json"):
        f.unlink()


def main() -> None:
    os.chdir(ROOT)  # configs use repo-relative paths
    gen_rag()
    gen_agent()


if __name__ == "__main__":
    main()
