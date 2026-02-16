

def is_notebook() -> bool:
    """
    Best-effort check whether we're running in a notebook (Jupyter/Lab).
    """
    try:
        from IPython.core.getipython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        # Jupyter uses ZMQInteractiveShell; terminal IPython uses TerminalInteractiveShell
        return ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False