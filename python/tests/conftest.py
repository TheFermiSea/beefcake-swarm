"""Shared pytest fixtures for the python/ test suite.

Adds the python/ directory to sys.path so tests can `from architect import ...`
and `from sepl import ...` without requiring an installed package.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

# Ensure python/ is importable when pytest is launched from the python/ dir.
_PY_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_PY_DIR) not in sys.path:
    sys.path.insert(0, str(_PY_DIR))


@pytest.fixture
def fresh_worktree(tmp_path: pathlib.Path) -> pathlib.Path:
    """A tiny single-file git repo with one commit, suitable for testing
    SEPL operators against a real worktree."""
    wt = tmp_path / "wt"
    wt.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(wt)], check=True)
    (wt / "foo.rs").write_text("#[derive(Debug)]\npub struct Foo;\n")
    subprocess.run(["git", "-C", str(wt), "add", "."], check=True)
    subprocess.run(["git", "-C", str(wt), "config", "user.email", "t@x"], check=True)
    subprocess.run(["git", "-C", str(wt), "config", "user.name", "T"], check=True)
    subprocess.run(["git", "-C", str(wt), "commit", "-qm", "init"], check=True)
    return wt


@pytest.fixture
def green_verifier():
    def _fn(_wt):
        return {"all_green": True,
                "gates": {"cargo-check": {"passed": True, "stderr_tail": ""}}}
    return _fn


@pytest.fixture
def red_verifier():
    def _fn(_wt):
        return {"all_green": False, "gates": {"cargo-check": {
            "passed": False,
            "stderr_tail": "error[E0308]: mismatched types\n  --> foo.rs:1:1",
        }}}
    return _fn
