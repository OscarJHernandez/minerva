"""A counterexample holds the user's expressions and the points where they fail: it is returned to the caller
(in the user's own conversation) and must never reach a log line, a trace event or an identifier field."""

import dataclasses

from minerva_api.agent.tracing import IDENTIFIER_FIELDS
from minerva_api.logging import ALLOWED_FIELDS
from minerva_engine.counterexample import Counterexample

CONTENT_FIELDS = {
    "expression_a",
    "expression_b",
    "assumptions",
    "substitution",
    "value_a",
    "value_b",
    "residual",
    "values",
}


def test_every_content_field_of_a_counterexample_is_kept_out_of_logs_and_traces() -> None:
    names = {f.name for f in dataclasses.fields(Counterexample)}
    assert names >= CONTENT_FIELDS  # the guard below must be kept in step with the record
    assert not CONTENT_FIELDS & set(ALLOWED_FIELDS)
    assert not CONTENT_FIELDS & set(IDENTIFIER_FIELDS)
