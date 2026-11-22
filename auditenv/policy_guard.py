"""Interpreter-level restrictions for the policy process (installed by agent_runner).

A Python audit hook cannot be removed once installed. It denies the policy process:

* reading closed data: any ``private/`` directory next to ``public/``, ``truth.json``,
  ``provenance.json``, evaluator logs and the repository's episodes, splits, trajectories, reports;
* importing evaluator-side modules (grader, generator, environment, server, harness, ...);
* writing or changing files, starting processes, opening sockets and loading native code via ctypes.

This is defence in depth and not an operating-system sandbox: closed runs should still start the
policy under a separate user or container without a mount of the episode directory.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROTECTED_DIRS = tuple(os.path.normcase(os.path.join(ROOT, d)) for d in
                        ("episodes", "configs", "trajectories", "reports"))
_CLOSED_NAMES = {"truth.json", "provenance.json"}
_DENIED_MODULES = {"judge_adapter"} | {f"auditenv.{m}" for m in (
    "calculator", "checker", "cli", "config", "env", "episode", "generator", "grader", "harness", "leakscan",
    "remote", "server", "tools", "transforms", "validate_data")}
_DENIED_EVENTS = {"subprocess.Popen", "os.system", "os.exec", "os.fork", "os.forkpty", "os.posix_spawn", "os.spawn",
                  "os.startfile", "os.startfile/2", "pty.spawn", "os.link", "os.symlink", "os.rename", "os.remove",
                  "os.rmdir", "os.mkdir", "os.truncate", "os.chmod", "os.chown", "os.putenv", "os.unsetenv"}
_DENIED_PREFIXES = ("socket.", "ctypes.", "shutil.", "winreg.")
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_state = threading.local()
_installed = False


def is_closed_path(path: Any) -> bool:
    if isinstance(path, int):
        return False
    text = os.fsdecode(os.fspath("." if path is None else path))
    real = os.path.normcase(os.path.realpath(text))
    name = os.path.basename(real)
    if name in _CLOSED_NAMES or "evaluator" in name:
        return True
    if any(real == d or real.startswith(d + os.sep) for d in _PROTECTED_DIRS):
        return True
    current = real
    while True:
        head, tail = os.path.split(current)
        if not tail:
            return False
        if tail == "private" and os.path.isdir(os.path.join(head, "public")):
            return True
        current = head


def _deny(event: str) -> None:
    raise PermissionError(f"policy process: {event} is not allowed")


def _hook(event: str, args: Tuple[Any, ...]) -> None:
    if getattr(_state, "busy", False):
        return
    _state.busy = True
    try:
        if event == "import":
            name = args[0] or ""
            if any(name == m or name.startswith(m + ".") for m in _DENIED_MODULES):
                _deny(f"import {name}")
        elif event == "open":
            path, _mode, flags = args
            if is_closed_path(path):
                _deny("reading closed data")
            if isinstance(flags, int) and flags & _WRITE_FLAGS and path != os.devnull:
                _deny("writing files")
        elif event in ("os.listdir", "os.scandir", "_winapi.CreateFile"):
            if is_closed_path(args[0]):
                _deny("reading closed data")
        elif event in _DENIED_EVENTS or event.startswith(_DENIED_PREFIXES):
            _deny(event)
    finally:
        _state.busy = False


def install() -> None:
    """Install the guard for the rest of this process. Call only in a dedicated policy process."""
    global _installed
    if _installed:
        return
    sys.dont_write_bytecode = True
    for name in list(sys.modules):
        if name in _DENIED_MODULES:
            del sys.modules[name]
    sys.addaudithook(_hook)
    _installed = True
