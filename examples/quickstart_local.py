"""Run the packaged local Cachet quickstart from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path


def _prefer_checkout_imports() -> None:
    """Let `python examples/quickstart_local.py` work from a source checkout."""

    repo_src = Path(__file__).resolve().parents[1] / "src"
    if repo_src.is_dir():
        sys.path.insert(0, str(repo_src))


_prefer_checkout_imports()

from cachet.quickstart_local import main  # noqa: E402


if __name__ == "__main__":
    main()
