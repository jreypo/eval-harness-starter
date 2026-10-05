import json

import pytest

from evalh.config import expand_env
from evalh.providers.base import Request, Response
from evalh.providers.stub import FixtureMissingError, RecordingProvider, StubProvider


class Scripted:
    name = "scripted"

    def complete(self, request, *, trial_index=0):
        return Response(
            content=[{"type": "text", "text": f"answer {trial_index}"}],
            stop_reason="end_turn",
            usage={"input_tokens": 3, "output_tokens": 2},
        )


def req(text="hi", model="m"):
    return Request(model=model, system="sys", messages=[{"role": "user", "content": text}])


def test_key_is_stable_and_sensitive_to_every_field():
    base = req()
    assert base.key() == req().key()
    assert base.key() != req(text="bye").key()
    assert base.key() != req(model="other").key()
    assert base.key() != Request("m", "other sys", base.messages).key()
    assert base.key() != Request("m", "sys", base.messages, tools=[{"name": "t"}]).key()
    # max_tokens is deliberately excluded.
    assert base.key() == Request("m", "sys", base.messages, max_tokens=99).key()


def test_record_then_replay_per_trial(tmp_path):
    rec = RecordingProvider(Scripted(), tmp_path)
    for i in range(3):
        rec.complete(req(), trial_index=i)
    stub = StubProvider(tmp_path)
    assert stub.complete(req(), trial_index=0).text == "answer 0"
    assert stub.complete(req(), trial_index=2).text == "answer 2"
    record = json.loads((tmp_path / f"{req().key()}.json").read_text())
    assert record["request"]["system"] == "sys"


def test_missing_fixture_is_loud_and_names_the_hash(tmp_path):
    stub = StubProvider(tmp_path)
    with pytest.raises(FixtureMissingError, match=req().key()):
        stub.complete(req())


def test_missing_trial_index_is_loud(tmp_path):
    RecordingProvider(Scripted(), tmp_path).complete(req(), trial_index=0)
    with pytest.raises(FixtureMissingError, match="no response for trial 4"):
        StubProvider(tmp_path).complete(req(), trial_index=4)


def test_missing_fixture_dir_is_loud(tmp_path):
    with pytest.raises(FixtureMissingError):
        StubProvider(tmp_path / "nope")


def test_env_expansion(monkeypatch):
    monkeypatch.delenv("EVAL_PROVIDER", raising=False)
    assert expand_env("p: ${EVAL_PROVIDER:-stub}") == "p: stub"
    monkeypatch.setenv("EVAL_PROVIDER", "anthropic")
    assert expand_env("p: ${EVAL_PROVIDER:-stub}") == "p: anthropic"


def test_anthropic_sdk_not_imported_by_default():
    import sys

    from evalh.providers.base import make_provider  # noqa: F401

    assert "anthropic" not in sys.modules
