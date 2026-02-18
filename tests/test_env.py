from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from progress_bar.env import is_notebook


def test_returns_false_when_ipython_not_installed() -> None:
    """When IPython is not available, should return False."""
    with patch("IPython.core.getipython.get_ipython", side_effect=ImportError):
        assert is_notebook() is False


def test_returns_false_when_get_ipython_is_none() -> None:
    """When get_ipython() returns None (not in IPython), should return False."""
    with patch("IPython.core.getipython.get_ipython", return_value=None):
        assert is_notebook() is False


@pytest.mark.parametrize(
    ("shell_class_name", "expected"),
    [
        ("ZMQInteractiveShell", True),
        ("TerminalInteractiveShell", False),
        ("SomeOtherShell", False),
        ("InteractiveShell", False),
    ],
)
def test_shell_type_detection(shell_class_name: str, expected: bool) -> None:
    """Test different IPython shell types are correctly identified."""
    mock_ip = Mock()
    mock_ip.__class__.__name__ = shell_class_name
    
    with patch("IPython.core.getipython.get_ipython", return_value=mock_ip):
        assert is_notebook() is expected


@pytest.mark.parametrize(
    "exception",
    [
        RuntimeError("Unexpected error"),
        ValueError("Bad value"),
        TypeError("Type issue"),
        AttributeError("Missing attribute"),
    ],
)
def test_returns_false_on_unexpected_exception(exception: Exception) -> None:
    """When unexpected exception occurs, should return False gracefully."""
    with patch("IPython.core.getipython.get_ipython", side_effect=exception):
        assert is_notebook() is False
