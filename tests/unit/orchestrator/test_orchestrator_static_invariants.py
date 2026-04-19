"""Static invariants on ``orchestrate`` that can't run end-to-end here.

The orchestrator function pulls in vLLM, scheduler, weight broadcast, and
torch — none of which load from this venv. These tests instead parse the
source AST and assert structural properties that a smoke run would catch
in production but only after burning hours of GPU time.

P1 reproducer: ``use_rae`` is read in the VLM-MA guard before its
assignment from ``isinstance(env.rubric, MultiAgentRubric)`` block. Python
local-scope rule: any function-level assignment makes the name local
throughout the function body, so the earlier read raises
``UnboundLocalError`` — for EVERY orchestrate invocation, single-agent or
multi-agent.

P2 reproducer: the final ``ckpt_manager.save`` (after the loop ends) does
not pass ``rae_state=``, so a multi-agent run that finishes on a
non-interval step writes a checkpoint without ``rae_state.pt``. Resuming
from that checkpoint then hits the load-side ``FileNotFoundError`` that
ckpt.py raises by design.
"""

from __future__ import annotations

import ast
from pathlib import Path

ORCHESTRATOR_SRC = Path(__file__).resolve().parents[3] / "src" / "prime_rl" / "orchestrator" / "orchestrator.py"


def _orchestrate_fn() -> ast.AsyncFunctionDef:
    tree = ast.parse(ORCHESTRATOR_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "orchestrate":
            return node
    raise AssertionError("`async def orchestrate` not found in orchestrator.py")


def test_use_rae_assigned_before_first_use():
    """P1: ``use_rae`` first Load (by source line) must come no earlier than
    first Store. Python's local scoping rule: any function-level assignment
    makes the name local throughout the function, so an earlier read raises
    ``UnboundLocalError``. ``ast.walk`` is BFS — collect all linenumbers
    and compare ``min`` to get true source order, not walk order."""
    fn = _orchestrate_fn()
    store_lines: list[int] = []
    load_lines: list[int] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == "use_rae":
            if isinstance(node.ctx, ast.Store):
                store_lines.append(node.lineno)
            elif isinstance(node.ctx, ast.Load):
                load_lines.append(node.lineno)
    assert store_lines, "use_rae never assigned in orchestrate()"
    assert load_lines, "use_rae never read in orchestrate()"
    first_store = min(store_lines)
    first_load = min(load_lines)
    assert first_store <= first_load, (
        f"P1: use_rae first read at line {first_load} but first assigned at "
        f"line {first_store} → UnboundLocalError on every orchestrate() call"
    )


def test_rae_state_assigned_before_first_use():
    """Companion check: same scoping trap applies to ``rae_state`` (assigned
    after the MA detection block, used in ckpt save/load + the per-step
    branching). Catch it the same way."""
    fn = _orchestrate_fn()
    store_lines: list[int] = []
    load_lines: list[int] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == "rae_state":
            if isinstance(node.ctx, ast.Store):
                store_lines.append(node.lineno)
            elif isinstance(node.ctx, ast.Load):
                load_lines.append(node.lineno)
    if not store_lines and not load_lines:
        return  # name not present, nothing to check
    assert store_lines, "rae_state read but never assigned in orchestrate()"
    assert load_lines, "rae_state assigned but never read in orchestrate()"
    first_store = min(store_lines)
    first_load = min(load_lines)
    assert first_store <= first_load, (
        f"rae_state first read at line {first_load} but first assigned at "
        f"line {first_store} → UnboundLocalError"
    )


def test_all_ckpt_saves_pass_rae_state_kwarg():
    """P2: every ``ckpt_manager.save(...)`` call in orchestrate() must pass
    ``rae_state=`` so multi-agent runs don't write incomplete final
    checkpoints. Single-agent runs pass ``rae_state=None`` (the default) and
    ckpt.py omits the file accordingly."""
    fn = _orchestrate_fn()
    save_call_lines: list[int] = []
    save_calls_with_rae: list[int] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "save"
            and isinstance(func.value, ast.Name)
            and func.value.id == "ckpt_manager"
        ):
            continue
        save_call_lines.append(node.lineno)
        if any(kw.arg == "rae_state" for kw in node.keywords):
            save_calls_with_rae.append(node.lineno)
    missing = sorted(set(save_call_lines) - set(save_calls_with_rae))
    assert not missing, (
        f"P2: ckpt_manager.save call(s) missing rae_state= kwarg at line(s) "
        f"{missing}; multi-agent runs ending on those paths will write "
        f"incomplete final checkpoints"
    )


def test_all_ckpt_loads_pass_rae_state_kwarg():
    """Symmetry with P2: ``ckpt_manager.load(...)`` calls must also pass
    rae_state so resume populates the in-place state."""
    fn = _orchestrate_fn()
    load_call_lines: list[int] = []
    load_calls_with_rae: list[int] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "load"
            and isinstance(func.value, ast.Name)
            and func.value.id == "ckpt_manager"
        ):
            continue
        load_call_lines.append(node.lineno)
        if any(kw.arg == "rae_state" for kw in node.keywords):
            load_calls_with_rae.append(node.lineno)
    missing = sorted(set(load_call_lines) - set(load_calls_with_rae))
    assert not missing, (
        f"ckpt_manager.load call(s) missing rae_state= kwarg at line(s) "
        f"{missing}"
    )
