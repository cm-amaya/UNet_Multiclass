"""Training utilities."""

import os
from typing import Any, Dict


def parse_args(args: list[str] | None = None) -> Dict[str, Any]:
    """Parse command-line arguments."""
    if args is None:
        args = os.environ.get("TRAIN_ARGS", "").split()
    if not args:
        return {}
    kwargs = {}
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            key = args[i][2:].replace("-", "_")
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                kwargs[key] = args[i + 1]
            else:
                kwargs[key] = True
            i += 1
        else:
            kwargs["positional"] = args[i]
            i += 1
    return kwargs


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
