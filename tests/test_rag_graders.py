import json
from pathlib import Path

import pytest

from evalh.rag.calibrate import Calibration
from evalh.rag.graders import (
    grade_refusal,
    grade_retrieval,
    parse_verdict,
    recall_at_k,
    reciprocal_rank,
)
from evalh.rag.index import BM25Index, chunk_markdown, load_corpus
from evalh.rag.pipeline import REFUSAL

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def index():
    return BM25Index(load_corpus(ROOT / "datasets/runbooks"))


def cases(name):
    return [json.loads(x) for x in (ROOT / f"datasets/rag/{name}.jsonl").read_text().splitlines()]


# ---- retrieval metrics ----------------------------------------------------------


def test_recall_and_reciprocal_rank():
    assert recall_at_k(["a", "b", "c"], ["b", "z"]) == 0.5
    assert recall_at_k(["a"], []) == 1.0
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], ["z"]) == 0.0


def test_grade_retrieval_threshold():
    assert grade_retrieval(["a", "b"], ["a", "b"], 1.0).passed is True
    assert grade_retrieval(["a", "c"], ["a", "b"], 1.0).passed is False
    assert grade_retrieval(["a", "c"], ["a", "b"], 0.5).passed is True


def test_chunk_ids_follow_sections():
    chunks = chunk_markdown("x.md", "# T\nintro\n\n## One\nbody\n\n## Two\nmore\n")
    assert [c.id for c in chunks] == ["x.md#0", "x.md#1", "x.md#2"]
    assert chunks[2].title == "Two"


def test_every_labelled_chunk_exists(index):
    for row in cases("regression") + cases("capability"):
        for cid in row["relevant_chunks"]:
            assert cid in index.by_id, (row["id"], cid)


def test_regression_cases_retrieve_their_chunks(index):
    for row in cases("regression"):
        got = [c.id for c, _ in index.search(row["question"], 3)]
        assert set(row["relevant_chunks"]) <= set(got), (row["id"], got)


def test_similar_name_trap_retrieves_the_wrong_service(index):
    """Pins the silent-failure case: 'ledger service' pulls ledger-sync, not ledger-svc."""
    (trap,) = [r for r in cases("capability") if r["id"] == "rag-cap-01"]
    got = [c.id for c, _ in index.search(trap["question"], 3)]
    assert all(cid.startswith("ledger-sync.md") for cid in got), got
    assert "ledger-svc.md#2" in trap["relevant_chunks"]


# ---- judge parsing ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"verdict": "pass", "reason": "ok"}', True),
        ('{"verdict": "FAIL", "reason": "wrong"}', False),
        ('{"verdict": "unknown", "reason": "cannot tell"}', None),
        ('```json\n{"verdict": "pass", "reason": "ok"}\n```', True),
        ('Here you go: {"verdict": "fail", "reason": "x"}', False),
    ],
)
def test_parse_verdict_valid(text, expected):
    g = parse_verdict("faithfulness", text)
    assert g.error is None
    assert g.passed is expected


@pytest.mark.parametrize(
    "text",
    [
        "Looks right to me.",
        '{"verdict": "probably", "reason": "x"}',
        '{"verdict": "pass", "reason": ',
        "[1, 2, 3]",
        "",
    ],
)
def test_parse_verdict_invalid_is_grader_error_not_crash(text):
    g = parse_verdict("correctness", text)
    assert g.passed is None
    assert g.error == "judge returned invalid output"


def test_refusal_grader():
    assert grade_refusal(REFUSAL).passed is True
    assert grade_refusal("Call +1 415 555 0137.").passed is False


# ---- calibration math ---------------------------------------------------------------


def test_calibration_exposes_class_imbalance():
    # 17 good answers all passed, 1 of 3 bad answers caught.
    cal = Calibration(n=20, tp=1, fp=0, fn=2, tn=17)
    assert cal.agreement == pytest.approx(0.90)
    assert cal.always_pass_agreement == pytest.approx(0.85)
    assert cal.fail_recall == pytest.approx(1 / 3)
    assert cal.kappa == pytest.approx(0.459, abs=1e-3)


def test_kappa_is_zero_for_always_pass_judge():
    cal = Calibration(n=20, tp=0, fp=0, fn=3, tn=17)
    assert cal.agreement == pytest.approx(0.85)
    assert cal.kappa == pytest.approx(0.0)
