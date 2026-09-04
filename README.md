# Progress Bar

A simple, explicit progress and logging package for Python applications. It
wraps Rich in a small context-managed API, supports determinate and indeterminate
work, reports rates and estimated completion, and keeps noisy worker logs from
destroying the terminal display.

FITS uses it for command-line workflow feedback, but the package has no
knowledge of microscopy data or FITS state and can be reused by any iterable or
task loop.

## Features

- **Simple API**: One import, context-managed progress bars
- **Log Control**: Explicit control over how logs behave while the progress bar is active
- **Three Log Modes**:
  - `"off"`: Suppress logs to keep the progress bar clean
  - `"buffered"`: Show logs in a bounded, rolling panel below the progress bar
  - `"live"`: Let logs flow normally (for debugging)
- **Generic**: No application-specific assumptions
- **Rich-based**: Beautiful terminal UI using [Rich](https://github.com/Textualize/rich)

## Installation

```bash
pip install -e .
```

Or if you're using uv:

```bash
uv pip install -e .
```

## Quick Start

### Basic Usage

```python
from progress_bar import pbar

# Simple progress bar
with pbar(total=100, desc="Processing") as pb:
    for item in items:
        process(item)
        pb.advance()
```

### Buffered Logs

Capture logs into a fixed rolling panel below the progress bar:

```python
from progress_bar import pbar
import logging

logger = logging.getLogger(__name__)

with pbar(total=len(states), desc="Segment", logs="buffered", log_lines=6) as pb:
    for state in states:
        logger.info(f"Processing {state}")
        process(state)
        pb.advance()
```

### Quiet Mode

Suppress logs completely for a clean progress bar:

```python
with pbar(total=tasks, desc="Pipeline", logs="off") as pb:
    for task in tasks:
        process(task)  # logs are muted while bar is active
        pb.advance()
```

### Live Logs

Let logs flow normally (useful for debugging):

```python
with pbar(total=10, desc="Debug Mode", logs="live") as pb:
    for i in range(10):
        logger.info(f"Step {i}")  # logs appear normally
        pb.advance()
```

## API Reference

### `pbar()`

Create a context-managed progress bar.

```python
pbar(
    total: int | None = None,
    desc: str = "Working",
    logs: Literal["off", "buffered", "live"] = "off",
    log_lines: int = 6,
    transient: bool = False,
) -> ProgressBar
```

**Parameters:**

- `total`: Total number of items to process (None = indeterminate)
- `desc`: Description text to show next to the progress bar
- `logs`: Log display mode:
  - `"off"`: Suppress console logs while bar is active (default)
  - `"buffered"`: Capture logs into a fixed rolling panel below the bar
  - `"live"`: Let logs flow normally (may interfere with bar display)
- `log_lines`: Number of log lines to show in buffered mode (default: 6)
- `transient`: If True, remove the progress bar when done (default: False)

**Returns:** A `ProgressBar` context manager

### `ProgressBar` Methods

#### `advance(n: int = 1)`

Advance the progress bar by n steps.

```python
pb.advance()      # advance by 1
pb.advance(5)     # advance by 5
```

#### `update(**kwargs)`

Update progress bar properties dynamically.

```python
pb.update(total=200)                    # change total
pb.update(completed=50)                 # set completed directly
pb.update(description="New Phase")      # change description
pb.update(description="Phase 2", total=150)  # multiple updates
```

#### `refresh()`

Manually refresh the display (usually not needed).

```python
pb.refresh()
```

#### `close()`

Manually close the progress bar. Normally you should use the context manager (`with` statement) instead.

## Examples

### Simple Loop

```python
from progress_bar import pbar

with pbar(total=len(items), desc="Processing") as pb:
    for item in items:
        process(item)
        pb.advance()
```

### Batch Processing with Logs

```python
import logging
from progress_bar import pbar

logger = logging.getLogger(__name__)

with pbar(total=len(batches), desc="Batches", logs="buffered", log_lines=8) as pb:
    for batch in batches:
        logger.info(f"Starting batch {batch.id}")
        result = process_batch(batch)
        logger.info(f"Batch {batch.id} completed: {result}")
        pb.advance()
```

### Dynamic Progress

```python
with pbar(total=10, desc="Phase 1", logs="buffered") as pb:
    for i in range(5):
        work()
        pb.advance()
    
    # Switch to phase 2 with new total
    pb.update(description="Phase 2", total=20)
    
    for i in range(15):
        work()
        pb.advance()
```

### Event-Driven Progress (e.g., Scheduler)

```python
with pbar(total=total_tasks, desc="Pipeline", logs="off") as pb:
    while not done:
        event = scheduler.next_event()
        if event.type == "task_completed":
            pb.advance()
        done = scheduler.is_done()
```

## Log Behavior Details

### `logs="off"` (default)

While the progress bar is active:
- Console log handlers are temporarily muted
- The progress bar remains clean and readable
- File handlers continue to work normally
- Logging handlers are restored when the progress bar exits

Use this when you want a clean progress display without log noise.

### `logs="buffered"`

While the progress bar is active:
- Console log handlers are muted
- Log records are captured into a bounded buffer
- Only the latest N log lines are displayed in a fixed panel below the progress bar
- Old log lines are automatically removed as new ones arrive
- File handlers continue to work normally
- Logging handlers are restored when the progress bar exits

Use this when you want to see recent logs without flooding the terminal.

### `logs="live"`

While the progress bar is active:
- Logs flow normally through console handlers
- This may cause the progress bar to become less stable or readable
- Useful primarily for debugging

Use this when you need to see all log output in real-time, even if it interferes with the progress bar.

## Design Philosophy

This package provides a **generic progress UI with optional bounded log rendering**.

It does **not** encode application-specific execution patterns. Instead, it gives you:
- One simple import
- Explicit control over when to advance progress
- Explicit control over how logs behave

This makes it suitable for:
- Simple loops
- Batch processing
- Event-driven systems (schedulers, async tasks, etc.)
- Any context where you want to show progress

## Advanced Usage

For advanced use cases, you can access the lower-level `ProgressManager` class directly:

```python
from progress_bar import ProgressManager

with ProgressManager.create(show_logs=True, max_log_lines=10) as pm:
    task1 = pm.add_task("Task 1", total=100)
    task2 = pm.add_task("Task 2", total=50)
    
    # ... work and call pm.advance(task1) or pm.advance(task2)
```

However, for most use cases, the simple `pbar()` API is recommended.

## Migration from Decorator-Based API

The previous decorator-based API has been removed in favor of this explicit API.

**Old (decorator-based):**
```python
@pbar(desc="Convert")
def run_convert(settings, exp_states):
    for state in exp_states:
        yield [convert(state)]
```

**New (explicit):**
```python
def run_convert(settings, exp_states):
    results = []
    with pbar(total=len(exp_states), desc="Convert", logs="buffered") as pb:
        for state in exp_states:
            results.append(convert(state))
            pb.advance()
    return results
```

Benefits of the new API:
- No hidden assumptions about function signatures
- No hidden assumptions about yielded values
- Works with any loop structure (including event-driven)
- Explicit control over progress and logging
- Simpler to understand and debug

## Requirements

- Python 3.10+
- rich

## License

MIT
