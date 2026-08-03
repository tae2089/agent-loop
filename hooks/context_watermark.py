#!/usr/bin/env python3
"""Stop/PreCompact watermark: block context loss until the agent writes a
handoff artifact.

Detection is deterministic (token usage from the transcript); writing the
handoff is creative work left to the agent. Context size is the last assistant
message's input_tokens + cache_read_input_tokens + cache_creation_input_tokens.

Satisfaction is stateless: any Write call this session whose file_path
contains "handoff" counts. Same fail-open policy as skill_invocation_verifier:
this tool's action is blocking, so its own failure must never lock a session.

Audit mode: `--check <transcript.jsonl>` prints current usage percentage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from transcript import (  # noqa: E402
    context_tokens,
    context_window,
    last_prompt_index,
    parse_transcript,
    read_hook_input,
    run_fail_open,
    successful_tool_results,
    tool_calls,
)
from handoff_state import HANDOFF_NAME, record_session_handoff, resolve_handoff  # noqa: E402
from artifact_lint import lint_file  # noqa: E402

LABEL = "context-watermark"
DEFAULT_WINDOW = 200_000
# Long-context recall degrades well before the window fills (Chroma "Context
# Rot", 2025), so evacuate at 0.85 rather than waiting for a near-full window.
DEFAULT_THRESHOLD = 0.85
# Auto compaction fires mid-turn and resets usage before Stop runs, so Stop's
# watermark can never see a full window in an auto-compacting session. Sample
# mid-turn too, low enough to leave room to write the handoff first.
DEFAULT_MIDTURN_THRESHOLD = 0.75
MIDTURN_EVENTS = frozenset({"PostToolUse"})


def _current_turn(entries: list[dict]) -> list[dict]:
    index = last_prompt_index(entries)
    return entries[index + 1:] if index is not None else []


def handoff_written(entries: list[dict], cwd: Path) -> tuple[Path | None, list[str]]:
    """Return a successful current-turn handoff and any structural lint notes.

    An allowed handoff must be a completed Write in the current turn and an
    existing regular file inside cwd. Unreadable files keep lint's fail-open
    behavior, while readable files must pass its structural floor. A returned
    path guarantees the notes list is empty.
    """
    turn_entries = _current_turn(entries)
    candidates = [
        call
        for entry in turn_entries
        for call in tool_calls(entry)
        if call.name == "Write"
        and Path(str(call.input.get("file_path", ""))).name == HANDOFF_NAME
    ]
    if not candidates:
        return None, []
    try:
        root = cwd.resolve(strict=True)
    except (OSError, RuntimeError):
        return None, []
    successful = successful_tool_results(turn_entries)
    notes = []
    seen: set[Path] = set()
    for call in candidates:
        if call.tool_use_id not in successful:
            continue
        path = resolve_handoff(root, str(call.input.get("file_path", "")))
        if path is None or path in seen:
            continue
        seen.add(path)
        result = lint_file(path, "handoff")
        if result is None or result["passed"]:
            return path, []
        notes.append(f"{path}: score={result['score']}, "
                     f"missing critical sections: {', '.join(result['floor_failures']) or 'none'}")
    return None, notes


def run_hook(window: int, threshold: float, midturn_threshold: float) -> int:
    hook_input = read_hook_input(LABEL)
    if hook_input is None:
        return 0
    return run_fail_open(
        LABEL, lambda: _run_hook(hook_input, window, threshold, midturn_threshold)
    )


def _print_block(hook_input: dict, reason: str) -> None:
    """Continue a Stop/PostToolUse turn, but stop compaction itself at PreCompact."""
    # decision:block feeds reason back to both Claude and Codex, and is the only
    # shape Claude Code accepts to cancel compaction:
    # https://code.claude.com/docs/en/hooks
    # Codex instead stops the lifecycle event via continue/stopReason, so
    # PreCompact carries both contracts.
    # https://learn.chatgpt.com/docs/hooks
    verdict = {"decision": "block", "reason": reason}
    if hook_input.get("hook_event_name") == "PreCompact":
        verdict.update({
            "continue": False,
            "stopReason": reason,
            "systemMessage": reason,
        })
    print(json.dumps(verdict, ensure_ascii=False))


def _run_hook(hook_input: dict, window: int, threshold: float,
              midturn_threshold: float) -> int:
    entries = parse_transcript(Path(hook_input.get("transcript_path")))
    event = hook_input.get("hook_event_name")
    compaction_due = event == "PreCompact"
    midturn = event in MIDTURN_EVENTS
    if midturn:
        threshold = midturn_threshold
    tokens = context_tokens(entries)
    if tokens is None and not compaction_due:
        return 0
    if tokens is not None:
        window = context_window(entries) or window
        pct = tokens / window
    else:
        pct = None
    if pct is not None and pct < threshold and not compaction_due:
        return 0
    cwd = Path(hook_input.get("cwd"))
    handoff, lint_notes = handoff_written(entries, cwd)
    if handoff is not None:
        record_session_handoff(cwd, hook_input.get("session_id"), handoff)
        return 0
    lint_suffix = ""
    if lint_notes:
        lint_suffix = (" A handoff was written but failed the structural lint — fix it: "
                       + "; ".join(lint_notes) + ".")
    usage = (
        f"Context is {pct:.0%} full ({tokens}/{window} tokens)."
        if pct is not None
        else "Compaction is starting."
    )
    reason = (
        f"[agent-loop] {usage}{lint_suffix} "
        "Before finishing, write a handoff artifact so work survives compaction: "
        "create _workspace/<current-task>/handoff.md (or handoff.md in cwd if no task folder) "
        "covering: current goal, work completed with file paths, key decisions and why, "
        "user corrections and value judgments (which work was deemed low-value, wasteful, or "
        "skippable — preserve these first and quote them verbatim rather than paraphrasing, "
        "they are the most expensive to rediscover and lose the most in summary), "
        "verified state (tests/commands run), and exact next steps. "
        + ("Write it now, before the context fills further, then carry on with the task."
           if midturn else "Then finish the turn.")
    )
    _print_block(hook_input, reason)
    return 0


def run_check(window: int, transcript: str) -> int:
    try:
        entries = parse_transcript(Path(transcript))
    except OSError as exc:
        print(f"cannot read transcript: {exc}", file=sys.stderr)
        return 2
    tokens = context_tokens(entries)
    if tokens is None:
        print("no usage data found")
        return 0
    window = context_window(entries) or window
    print(f"context: {tokens}/{window} tokens ({tokens / window:.1%}), "
          "handoff status requires hook-mode cwd and current-turn result data")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="context window in tokens; used only when the transcript reports none")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="block threshold ratio")
    parser.add_argument("--midturn-threshold", type=float, default=DEFAULT_MIDTURN_THRESHOLD,
                        help=f"block threshold ratio for mid-turn events ({', '.join(sorted(MIDTURN_EVENTS))})")
    parser.add_argument("--check", metavar="TRANSCRIPT", help="report usage instead of running as a hook")
    args = parser.parse_args()
    if args.check:
        return run_check(args.window, args.check)
    return run_hook(args.window, args.threshold, args.midturn_threshold)


if __name__ == "__main__":
    sys.exit(main())
