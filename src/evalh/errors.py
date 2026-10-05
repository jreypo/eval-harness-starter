"""Errors that mean the harness is broken, as opposed to the system under test.

The runner records ordinary exceptions on the trial and keeps going. A
HarnessError instead aborts the whole run: a missing fixture or a malformed
dataset says nothing about the model, and averaging it into a pass rate would
hide the real problem.
"""


class HarnessError(RuntimeError):
    pass
