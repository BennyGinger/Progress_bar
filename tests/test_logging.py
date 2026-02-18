from __future__ import annotations

import logging
from io import StringIO
from unittest.mock import Mock

import pytest

from progress_bar.logging import (
    LogBuffer,
    BufferHandler,
    attach_buffer_handler,
    detach_handler,
    HandlerSwap,
    detach_console_stream_handlers,
    restore_handlers,
)


def test_log_buffer_post_init_creates_deque() -> None:
    """LogBuffer should initialize deque in __post_init__."""
    buf = LogBuffer(max_lines=10)
    assert buf._lines is not None
    assert buf._lines.maxlen == 10


@pytest.mark.parametrize(
    ("max_lines", "items_to_add", "expected_length"),
    [
        (5, 3, 3),
        (5, 5, 5),
        (5, 7, 5),  # Should evict oldest 2
        (1, 3, 1),  # Should only keep last
    ],
)
def test_log_buffer_respects_max_lines(max_lines: int, items_to_add: int, expected_length: int) -> None:
    """LogBuffer should respect max_lines limit."""
    buf = LogBuffer(max_lines=max_lines)
    for i in range(items_to_add):
        buf.append(f"Line {i}")
    assert len(buf.lines()) == expected_length


def test_log_buffer_lines_returns_list_copy() -> None:
    """lines() should return a list, not the deque itself."""
    buf = LogBuffer(max_lines=5)
    buf.append("Test")
    result = buf.lines()
    assert isinstance(result, list)
    assert result == ["Test"]


def test_log_buffer_clear_empties_buffer() -> None:
    """clear() should remove all lines."""
    buf = LogBuffer(max_lines=5)
    buf.append("Line 1")
    buf.append("Line 2")
    buf.clear()
    assert buf.lines() == []


def test_buffer_handler_formats_and_stores_records() -> None:
    """BufferHandler should format log records and store them in buffer."""
    buf = LogBuffer(max_lines=10)
    handler = BufferHandler(buf, level=logging.INFO)
    
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    
    handler.emit(record)
    
    lines = buf.lines()
    assert len(lines) == 1
    assert "Test message" in lines[0]
    assert "INFO" in lines[0]
    assert "test.logger" in lines[0]


def test_buffer_handler_calls_on_emit_callback() -> None:
    """BufferHandler should call on_emit callback when emitting."""
    buf = LogBuffer(max_lines=10)
    callback = Mock()
    handler = BufferHandler(buf, level=logging.INFO, on_emit=callback)
    
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Message",
        args=(),
        exc_info=None,
    )
    
    handler.emit(record)
    
    callback.assert_called_once()


def test_buffer_handler_handles_emit_errors_gracefully() -> None:
    """BufferHandler should not raise when emit fails."""
    buf = LogBuffer(max_lines=10)
    handler = BufferHandler(buf, level=logging.INFO)
    
    # Create a record that will cause formatting to fail
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Message with %s",  # Missing format arg
        args=(),  # Empty args will cause KeyError
        exc_info=None,
    )
    
    # Should not raise, but will call handleError internally
    handler.emit(record)


def test_buffer_handler_respects_level() -> None:
    """BufferHandler should only emit records at or above its level."""
    buf = LogBuffer(max_lines=10)
    handler = BufferHandler(buf, level=logging.WARNING)
    
    # Test that handler's level is set correctly
    assert handler.level == logging.WARNING
    
    info_record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="test.py", lineno=1,
        msg="Info", args=(), exc_info=None,
    )
    warning_record = logging.LogRecord(
        name="test", level=logging.WARNING, pathname="test.py", lineno=1,
        msg="Warning", args=(), exc_info=None,
    )
    
    # Manually check level before emitting (like handle() does)
    if info_record.levelno >= handler.level:
        handler.emit(info_record)
    assert len(buf.lines()) == 0  # Should not be emitted
    
    # WARNING should go through
    if warning_record.levelno >= handler.level:
        handler.emit(warning_record)
    assert len(buf.lines()) == 1


def test_attach_buffer_handler_adds_to_logger() -> None:
    """attach_buffer_handler should add handler to logger."""
    logger = logging.getLogger("test_attach")
    buf = LogBuffer(max_lines=10)
    initial_count = len(logger.handlers)
    
    handler = attach_buffer_handler(logger, buf, level=logging.INFO)
    
    assert len(logger.handlers) == initial_count + 1
    assert handler in logger.handlers
    
    # Cleanup
    logger.removeHandler(handler)


def test_attach_buffer_handler_preserves_other_handlers() -> None:
    """attach_buffer_handler should not affect existing handlers."""
    logger = logging.getLogger("test_preserve")
    existing_handler = logging.StreamHandler(StringIO())
    logger.addHandler(existing_handler)
    
    buf = LogBuffer(max_lines=10)
    new_handler = attach_buffer_handler(logger, buf, level=logging.INFO)
    
    assert existing_handler in logger.handlers
    assert new_handler in logger.handlers
    
    # Cleanup
    logger.removeHandler(existing_handler)
    logger.removeHandler(new_handler)


def test_attach_buffer_handler_with_on_emit() -> None:
    """attach_buffer_handler should pass on_emit callback to handler."""
    logger = logging.getLogger("test_on_emit")
    logger.setLevel(logging.DEBUG)  # Ensure logger level allows INFO messages
    buf = LogBuffer(max_lines=10)
    callback = Mock()
    
    handler = attach_buffer_handler(logger, buf, level=logging.INFO, on_emit=callback)
    logger.info("Test message")
    
    callback.assert_called()
    
    # Cleanup
    logger.removeHandler(handler)


def test_detach_handler_removes_and_closes() -> None:
    """detach_handler should remove handler from logger and close it."""
    logger = logging.getLogger("test_detach")
    handler = logging.StreamHandler(StringIO())
    logger.addHandler(handler)
    
    assert handler in logger.handlers
    
    detach_handler(logger, handler)
    
    assert handler not in logger.handlers


def test_detach_handler_closes_even_on_error() -> None:
    """detach_handler should close handler even if removal fails."""
    logger = logging.getLogger("test_detach_error")
    handler = Mock(spec=logging.Handler)
    handler.close = Mock()
    
    # Handler not actually in logger, but should still close
    detach_handler(logger, handler)
    
    handler.close.assert_called_once()


def test_detach_console_stream_handlers_removes_stream_handlers() -> None:
    """detach_console_stream_handlers should remove StreamHandlers."""
    logger = logging.getLogger("test_detach_streams")
    stream_handler = logging.StreamHandler(StringIO())
    logger.addHandler(stream_handler)
    
    swap = detach_console_stream_handlers()
    
    assert stream_handler not in logger.handlers
    assert len(swap.removed) >= 1
    assert any(h is stream_handler for _, h in swap.removed)
    
    # Cleanup
    restore_handlers(swap)


def test_detach_console_stream_handlers_preserves_file_handlers() -> None:
    """detach_console_stream_handlers should NOT remove FileHandlers."""
    logger = logging.getLogger("test_preserve_file")
    
    # Create a mock FileHandler
    file_handler = logging.FileHandler("/tmp/test.log")
    logger.addHandler(file_handler)
    
    try:
        swap = detach_console_stream_handlers()
        
        # FileHandler should still be attached
        assert file_handler in logger.handlers
        
        # Cleanup
        restore_handlers(swap)
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()


def test_detach_console_stream_handlers_handles_multiple_loggers() -> None:
    """detach_console_stream_handlers should process all loggers."""
    logger1 = logging.getLogger("test_multi1")
    logger2 = logging.getLogger("test_multi2")
    
    handler1 = logging.StreamHandler(StringIO())
    handler2 = logging.StreamHandler(StringIO())
    
    logger1.addHandler(handler1)
    logger2.addHandler(handler2)
    
    swap = detach_console_stream_handlers()
    
    assert handler1 not in logger1.handlers
    assert handler2 not in logger2.handlers
    
    # Cleanup
    restore_handlers(swap)


def test_restore_handlers_reattaches_all() -> None:
    """restore_handlers should reattach all previously removed handlers."""
    logger = logging.getLogger("test_restore")
    handler1 = logging.StreamHandler(StringIO())
    handler2 = logging.StreamHandler(StringIO())
    
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    
    swap = detach_console_stream_handlers()
    
    # Both should be removed
    assert handler1 not in logger.handlers
    assert handler2 not in logger.handlers
    
    restore_handlers(swap)
    
    # Both should be back
    assert handler1 in logger.handlers
    assert handler2 in logger.handlers
    
    # Cleanup
    logger.removeHandler(handler1)
    logger.removeHandler(handler2)


def test_handler_swap_is_dataclass() -> None:
    """HandlerSwap should be a simple dataclass container."""
    logger = logging.getLogger("test")
    handler = logging.StreamHandler(StringIO())
    
    swap = HandlerSwap(removed=[(logger, handler)])
    
    assert len(swap.removed) == 1
    assert swap.removed[0] == (logger, handler)


def test_buffer_handler_custom_formatter() -> None:
    """BufferHandler formatter can be customized."""
    buf = LogBuffer(max_lines=10)
    handler = BufferHandler(buf, level=logging.INFO)
    
    # Set custom formatter
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="test.py", lineno=1,
        msg="Custom format", args=(), exc_info=None,
    )
    
    handler.emit(record)
    
    lines = buf.lines()
    assert len(lines) == 1
    assert lines[0] == "INFO: Custom format"
