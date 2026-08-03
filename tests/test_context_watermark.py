"""Contract tests for hooks/context_watermark.py."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hooks"))
from handoff_state import marker_path, record_session_handoff  # noqa: E402
from transcript_helpers import assistant_usage, tool_result, user_text, write_call  # noqa: E402

WATERMARK = Path(__file__).resolve().parent.parent / "hooks" / "context_watermark.py"

GOOD_HANDOFF = """# handoff

## 목표
Agent Loop의 context watermark 보강

## 완료 작업
- hooks/context_watermark.py 수정

## 결정
- 현재 턴의 성공한 Write만 인정

## 검증 상태
- 회귀 테스트 실행

## 다음 단계
- 전체 테스트를 실행하고 결과 확인
"""


class WatermarkHarness(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.transcript = self.dir / "session.jsonl"

    def write_transcript(self, entries):
        self.transcript.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")

    def run_hook(self, hook_input=None, stdin_raw=None, extra_args=None, threshold="0.9"):
        args = [sys.executable, str(WATERMARK), "--window", "200000"]
        if threshold is not None:  # None exercises the built-in default threshold
            args += ["--threshold", threshold]
        if extra_args:
            args += extra_args
        data = stdin_raw if stdin_raw is not None else json.dumps(hook_input)
        return subprocess.run(args, input=data, capture_output=True, text=True, timeout=30)

    def hook_input(self, **over):
        base = {"transcript_path": str(self.transcript), "stop_hook_active": False,
                "cwd": str(self.dir), "session_id": "session-1"}
        base.update(over)
        return base


class TestWatermark(WatermarkHarness):
    def test_t1_below_threshold_passes(self):
        self.write_transcript([assistant_usage(100_000)])
        proc = self.run_hook(self.hook_input())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "")

    def test_t2_above_threshold_without_handoff_blocks(self):
        self.write_transcript([assistant_usage(185_000)])
        proc = self.run_hook(self.hook_input())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        verdict = json.loads(proc.stdout)
        self.assertEqual(verdict["decision"], "block")
        self.assertIn("handoff", verdict["reason"])
        self.assertIn("92", verdict["reason"])  # 185k/200k = 92.5%

    def test_t3_above_threshold_with_handoff_written_passes(self):
        handoff = self.dir / "handoff.md"
        handoff.write_text(GOOD_HANDOFF, encoding="utf-8")
        self.write_transcript([
            user_text("handoff를 작성해"),
            write_call(str(handoff)),
            tool_result(),
            assistant_usage(185_000),
        ])
        proc = self.run_hook(self.hook_input())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "")
        markers = list((self.dir / "_workspace" / ".handoff-sessions").glob("*.json"))
        self.assertEqual(len(markers), 1)
        self.assertEqual(json.loads(markers[0].read_text(encoding="utf-8"))["path"], "handoff.md")

    def test_t7_previous_turn_handoff_does_not_satisfy(self):
        handoff = self.dir / "handoff.md"
        handoff.write_text(GOOD_HANDOFF, encoding="utf-8")
        self.write_transcript([
            user_text("이전 턴"),
            write_call(str(handoff)),
            tool_result(),
            user_text("새 턴"),
            assistant_usage(185_000),
        ])
        proc = self.run_hook(self.hook_input())
        self.assertEqual(json.loads(proc.stdout)["decision"], "block")

    def test_t8_failed_write_does_not_satisfy(self):
        handoff = self.dir / "handoff.md"
        handoff.write_text(GOOD_HANDOFF, encoding="utf-8")
        self.write_transcript([
            user_text("handoff를 작성해"),
            write_call(str(handoff)),
            tool_result(is_error=True),
            assistant_usage(185_000),
        ])
        proc = self.run_hook(self.hook_input())
        self.assertEqual(json.loads(proc.stdout)["decision"], "block")

    def test_t9_similar_filename_does_not_satisfy(self):
        wrong = self.dir / "not-a-handoff.txt"
        wrong.write_text(GOOD_HANDOFF, encoding="utf-8")
        self.write_transcript([
            user_text("handoff를 작성해"),
            write_call(str(wrong)),
            tool_result(),
            assistant_usage(185_000),
        ])
        proc = self.run_hook(self.hook_input())
        self.assertEqual(json.loads(proc.stdout)["decision"], "block")

    def test_t10_handoff_outside_project_does_not_satisfy(self):
        outside = Path(tempfile.mkdtemp()) / "handoff.md"
        outside.write_text(GOOD_HANDOFF, encoding="utf-8")
        self.write_transcript([
            user_text("handoff를 작성해"),
            write_call(str(outside)),
            tool_result(),
            assistant_usage(185_000),
        ])
        proc = self.run_hook(self.hook_input())
        self.assertEqual(json.loads(proc.stdout)["decision"], "block")

    def test_t13_malformed_usage_fails_open(self):
        self.write_transcript([assistant_usage("not-a-number")])
        proc = self.run_hook(self.hook_input())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "")

    def test_t14_missing_cwd_fails_open(self):
        self.write_transcript([assistant_usage(185_000)])
        proc = self.run_hook(self.hook_input(cwd=None))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "")

    def test_t15_non_utf8_handoff_fails_open(self):
        handoff = self.dir / "handoff.md"
        handoff.write_bytes(b"\xff\xfe")
        self.write_transcript([
            user_text("handoff를 작성해"),
            write_call(str(handoff)),
            tool_result(),
            assistant_usage(185_000),
        ])
        proc = self.run_hook(self.hook_input())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "")

    def test_t16_session_marker_symlink_is_not_followed(self):
        handoff = self.dir / "handoff.md"
        handoff.write_text(GOOD_HANDOFF, encoding="utf-8")
        marker = marker_path(self.dir, "session-1")
        marker.parent.mkdir(parents=True)
        outside = Path(tempfile.mkdtemp()) / "outside.json"
        outside.write_text("DO_NOT_OVERWRITE", encoding="utf-8")
        marker.symlink_to(outside)
        self.write_transcript([
            user_text("handoff를 작성해"),
            write_call(str(handoff)),
            tool_result(),
            assistant_usage(185_000),
        ])
        proc = self.run_hook(self.hook_input())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(outside.read_text(encoding="utf-8"), "DO_NOT_OVERWRITE")

    def test_t4_stop_hook_active_without_handoff_still_blocks(self):
        self.write_transcript([assistant_usage(185_000)])
        proc = self.run_hook(self.hook_input(stop_hook_active=True))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["decision"], "block")

    def test_t5_no_usage_fail_open(self):
        self.write_transcript([{"type": "user", "message": {"role": "user", "content": "hi"}}])
        proc = self.run_hook(self.hook_input())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "")

    def test_t6_check_mode_reports_percentage(self):
        self.write_transcript([assistant_usage(185_000)])
        proc = self.run_hook(extra_args=["--check", str(self.transcript)], stdin_raw="")
        self.assertIn("92.5%", proc.stdout)

    def test_t17_default_threshold_blocks_below_old_ninety(self):
        # A4: degradation starts well before the window fills, so the default
        # gate is 0.8. 0.85 of 200k must block without an explicit --threshold.
        self.write_transcript([assistant_usage(170_000)])
        proc = self.run_hook(self.hook_input(), threshold=None)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["decision"], "block")

    def test_t18_block_reason_demands_verbatim_user_quotes(self):
        # A5: paraphrased judgments degrade recall; the reason must ask for quotes.
        self.write_transcript([assistant_usage(190_000)])
        reason = json.loads(self.run_hook(self.hook_input()).stdout)["reason"]
        self.assertRegex(reason, r"(verbatim|원문|그대로|quote)")

    def test_t19_marker_replace_failure_preserves_previous_handoff(self):
        original = self.dir / "_workspace" / "original" / "handoff.md"
        replacement = self.dir / "_workspace" / "replacement" / "handoff.md"
        original.parent.mkdir(parents=True)
        replacement.parent.mkdir(parents=True)
        original.write_text(GOOD_HANDOFF, encoding="utf-8")
        replacement.write_text(GOOD_HANDOFF, encoding="utf-8")
        marker = marker_path(self.dir, "session-1")
        marker.parent.mkdir(parents=True)
        marker.write_text(
            json.dumps({"path": original.relative_to(self.dir).as_posix()}),
            encoding="utf-8",
        )

        with patch("session_marker.os.replace", side_effect=OSError("replace failed")):
            self.assertFalse(
                record_session_handoff(self.dir, "session-1", replacement.resolve())
            )

        self.assertEqual(
            json.loads(marker.read_text(encoding="utf-8")),
            {"path": "_workspace/original/handoff.md"},
        )
        self.assertEqual(list(marker.parent.glob(".session-marker-*.tmp")), [])

    def test_t20_codex_stop_continues_with_handoff_feedback(self):
        self.write_transcript([assistant_usage(185_000)])

        proc = self.run_hook(self.hook_input(
            hook_event_name="Stop",
            turn_id="turn-1",
            model="gpt-5.6-sol",
        ))

        self.assertEqual(proc.returncode, 0, proc.stderr)
        verdict = json.loads(proc.stdout)
        self.assertEqual(verdict["decision"], "block")
        self.assertIn("handoff", verdict["reason"])
        self.assertNotIn("continue", verdict)

    def test_t21_claude_stop_keeps_claude_block_contract(self):
        self.write_transcript([assistant_usage(185_000)])

        proc = self.run_hook(self.hook_input(hook_event_name="Stop"))

        self.assertEqual(proc.returncode, 0, proc.stderr)
        verdict = json.loads(proc.stdout)
        self.assertEqual(verdict["decision"], "block")
        self.assertIn("handoff", verdict["reason"])
        self.assertNotIn("continue", verdict)

    def test_t22_codex_precompact_blocks_even_below_threshold(self):
        self.write_transcript([assistant_usage(100_000)])

        proc = self.run_hook(self.hook_input(
            hook_event_name="PreCompact",
            trigger="auto",
            turn_id="turn-1",
            model="gpt-5.6-sol",
        ))

        self.assertEqual(proc.returncode, 0, proc.stderr)
        verdict = json.loads(proc.stdout)
        self.assertEqual(verdict["continue"], False)
        self.assertIn("compaction", verdict["stopReason"])

    def test_t23_codex_precompact_allows_valid_current_turn_handoff(self):
        handoff = self.dir / "handoff.md"
        handoff.write_text(GOOD_HANDOFF, encoding="utf-8")
        self.write_transcript([
            user_text("handoff를 작성해"),
            write_call(str(handoff)),
            tool_result(),
            assistant_usage(100_000),
        ])

        proc = self.run_hook(self.hook_input(
            hook_event_name="PreCompact",
            trigger="manual",
            turn_id="turn-1",
            model="gpt-5.6-sol",
        ))

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "")

    def test_t24_codex_precompact_blocks_when_usage_is_missing(self):
        self.write_transcript([
            {"type": "user", "message": {"role": "user", "content": "계속"}},
        ])

        proc = self.run_hook(self.hook_input(
            hook_event_name="PreCompact",
            trigger="auto",
            turn_id="turn-1",
            model="gpt-5.6-sol",
        ))

        self.assertEqual(proc.returncode, 0, proc.stderr)
        verdict = json.loads(proc.stdout)
        self.assertEqual(verdict["continue"], False)
        self.assertIn("Compaction is starting", verdict["stopReason"])

    def test_t25_claude_precompact_blocks_with_claude_decision_contract(self):
        # Claude Code cancels compaction on a top-level decision:block only;
        # continue/stopReason alone lets auto compaction through.
        self.write_transcript([assistant_usage(100_000)])

        proc = self.run_hook(self.hook_input(
            hook_event_name="PreCompact",
            trigger="auto",
        ))

        self.assertEqual(proc.returncode, 0, proc.stderr)
        verdict = json.loads(proc.stdout)
        self.assertEqual(verdict["decision"], "block")
        self.assertIn("handoff", verdict["reason"])


class TestMidTurnWatermark(WatermarkHarness):
    """Auto compaction fires mid-turn, so Stop is too late to be the only gate."""

    def test_t26_midturn_above_threshold_blocks_before_auto_compaction(self):
        self.write_transcript([assistant_usage(160_000)])  # 80%: under Stop's 0.9

        proc = self.run_hook(
            self.hook_input(hook_event_name="PostToolUse"),
            extra_args=["--midturn-threshold", "0.75"],
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        verdict = json.loads(proc.stdout)
        self.assertEqual(verdict["decision"], "block")
        self.assertIn("handoff", verdict["reason"])
        self.assertNotIn("continue", verdict)  # must not abort the turn

    def test_t27_midturn_below_threshold_passes(self):
        self.write_transcript([assistant_usage(100_000)])  # 50%

        proc = self.run_hook(
            self.hook_input(hook_event_name="PostToolUse"),
            extra_args=["--midturn-threshold", "0.75"],
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "")

    def test_t28_midturn_with_valid_handoff_passes(self):
        handoff = self.dir / "handoff.md"
        handoff.write_text(GOOD_HANDOFF, encoding="utf-8")
        self.write_transcript([
            user_text("handoff를 작성해"),
            write_call(str(handoff)),
            tool_result(),
            assistant_usage(160_000),
        ])

        proc = self.run_hook(
            self.hook_input(hook_event_name="PostToolUse"),
            extra_args=["--midturn-threshold", "0.75"],
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "")

    def test_t29_default_midturn_threshold_leaves_headroom_for_auto_compaction(self):
        self.write_transcript([assistant_usage(156_000)])  # 78%

        proc = self.run_hook(self.hook_input(hook_event_name="PostToolUse"))

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["decision"], "block")

    def test_t30_midturn_without_usage_data_passes(self):
        self.write_transcript([
            {"type": "user", "message": {"role": "user", "content": "계속"}},
        ])

        proc = self.run_hook(self.hook_input(hook_event_name="PostToolUse"))

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()


class TestWatermarkLintIntegration(WatermarkHarness):
    def test_t11_empty_shell_handoff_still_blocks_with_lint_reason(self):
        handoff = self.dir / "handoff.md"
        handoff.write_text("# handoff\n\n## 목표\nx\n", encoding="utf-8")  # missing floors
        self.write_transcript([
            user_text("handoff를 작성해"),
            write_call(str(handoff)),
            tool_result(),
            assistant_usage(185_000),
        ])
        proc = self.run_hook(self.hook_input())
        verdict = json.loads(proc.stdout)
        self.assertEqual(verdict["decision"], "block")
        self.assertIn("failed the structural lint", verdict["reason"])

    def test_t12_good_handoff_on_disk_passes(self):
        handoff = self.dir / "handoff.md"
        handoff.write_text(GOOD_HANDOFF, encoding="utf-8")
        self.write_transcript([
            user_text("handoff를 작성해"),
            write_call(str(handoff)),
            tool_result(),
            assistant_usage(185_000),
        ])
        proc = self.run_hook(self.hook_input())
        self.assertEqual(proc.stdout.strip(), "")
