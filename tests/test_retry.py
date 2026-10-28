import pytest

from src.core.retry import retry_transient


def test_retries_transient_then_succeeds():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("Server disconnected")
        return "ok"

    assert retry_transient(flaky, delay=0) == "ok"
    assert len(calls) == 3


def test_does_not_retry_other_errors():
    calls = []

    def broken():
        calls.append(1)
        raise ValueError("violates unique constraint")

    with pytest.raises(ValueError):
        retry_transient(broken, delay=0)
    assert len(calls) == 1


def test_gives_up_after_attempts():
    def always():
        raise RuntimeError("Server disconnected")

    with pytest.raises(RuntimeError):
        retry_transient(always, attempts=2, delay=0)
