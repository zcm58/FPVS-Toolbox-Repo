import importlib.util
import threading
import pytest

if importlib.util.find_spec("customtkinter") is None:
    pytest.skip("customtkinter not available", allow_module_level=True)

from Main_App.tk_logging_mixin import TkLoggingMixin


class DummyWidget(TkLoggingMixin):
    def __init__(self):
        self.messages = []
        super().__init__()

    def after(self, delay, callback, *args):
        callback(*args)

    def _append_log(self, message: str) -> None:  # type: ignore[override]
        self.messages.append(message)


def test_log_from_thread_no_exception():
    widget = DummyWidget()
    exc: list[Exception] = []

    def worker():
        try:
            widget.log("hi")
        except Exception as e:  # pragma: no cover - should not happen
            exc.append(e)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert not exc
    assert widget.messages == ["hi"]
