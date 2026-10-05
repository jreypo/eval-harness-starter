"""Calibrate the faithfulness judge against human labels.

A judge is a sensor, and you calibrate a sensor before you alert on it. The
trap is class imbalance: if 85% of answers are good, a judge that always says
"pass" scores 85% agreement while catching zero bad answers. So we report the
numbers that expose that: recall on the "fail" class and Cohen's kappa.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalh.rag.graders import Judge
from evalh.rag.index import BM25Index
from evalh.rag.pipeline import format_context


@dataclass
class Calibration:
    n: int = 0
    tp: int = 0  # judge fail, human fail (caught a bad answer)
    fp: int = 0  # judge fail, human pass (false alarm)
    fn: int = 0  # judge pass, human fail (missed a bad answer)
    tn: int = 0  # judge pass, human pass
    unknown: int = 0  # judge abstained
    errors: int = 0  # judge output unparseable
    disagreements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def decided(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def agreement(self) -> float:
        return (self.tp + self.tn) / self.decided if self.decided else 0.0

    @property
    def always_pass_agreement(self) -> float:
        # What a judge that never fails anything would score.
        return (self.fp + self.tn) / self.decided if self.decided else 0.0

    @property
    def fail_recall(self) -> float:
        bad = self.tp + self.fn
        return self.tp / bad if bad else 0.0

    @property
    def fail_precision(self) -> float:
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else 0.0

    @property
    def kappa(self) -> float:
        """Agreement corrected for what chance alone would produce."""
        n = self.decided
        if n == 0:
            return 0.0
        po = self.agreement
        judge_fail = (self.tp + self.fp) / n
        human_fail = (self.tp + self.fn) / n
        pe = judge_fail * human_fail + (1 - judge_fail) * (1 - human_fail)
        return (po - pe) / (1 - pe) if pe < 1 else 0.0


def load_labels(path: Path | str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def calibrate(labels: list[dict[str, Any]], judge: Judge, index: BM25Index) -> Calibration:
    cal = Calibration()
    for row in labels:
        context = format_context([index.by_id[c] for c in row["context_chunks"]])
        result = judge.faithfulness(context, row["answer"])
        human_pass = row["human_verdict"] == "pass"
        cal.n += 1
        if result.error:
            cal.errors += 1
            continue
        if result.passed is None:
            cal.unknown += 1
            continue
        judge_pass = result.passed
        if judge_pass and human_pass:
            cal.tn += 1
        elif judge_pass and not human_pass:
            cal.fn += 1
        elif not judge_pass and human_pass:
            cal.fp += 1
        else:
            cal.tp += 1
        if judge_pass != human_pass:
            cal.disagreements.append(
                {
                    "id": row["id"],
                    "human": row["human_verdict"],
                    "judge": "pass" if judge_pass else "fail",
                    "note": row.get("note", ""),
                }
            )
    return cal


def format_calibration(cal: Calibration, judge_model: str) -> str:
    human_fail = cal.tp + cal.fn
    lines = [
        f"Judge calibration: faithfulness judge ({judge_model}) vs human labels",
        f"labels: {cal.n}  (human pass={cal.tn + cal.fp}, human fail={human_fail})"
        f"  unknown={cal.unknown}  judge_errors={cal.errors}",
        "",
        f"  raw agreement                 {cal.agreement:.2f}",
        f"  'always pass' baseline        {cal.always_pass_agreement:.2f}   <- never-fail judge",
        f"  recall on fail class          {cal.fail_recall:.2f}   <- bad answers caught",
        f"  precision on fail class       {cal.fail_precision:.2f}",
        f"  Cohen's kappa                 {cal.kappa:.2f}",
        "",
        "confusion (rows = human, cols = judge):",
        "              judge pass  judge fail",
        f"  human pass  {cal.tn:>10}  {cal.fp:>10}",
        f"  human fail  {cal.fn:>10}  {cal.tp:>10}",
    ]
    if cal.disagreements:
        lines += ["", "disagreements:"]
        for d in cal.disagreements:
            lines.append(f"  {d['id']:<10} human={d['human']:<5} judge={d['judge']:<5} {d['note']}")
    return "\n".join(lines)
