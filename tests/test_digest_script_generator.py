"""Tests for src/digest_script_generator.py."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.digest_script_generator import (
    _load_daily_script,
    _log_usage,
    _summarise_daily,
    collect_week_scripts,
    generate_digest_script,
    save_for_digest,
)
from src.script_generator import VideoScript

# ── helpers ────────────────────────────────────────────────────────────────────

def _make_daily_data(title: str = "GPT-5 Breaks Records") -> dict:
    return {
        "title": title,
        "scenes": [
            {"heading": "Hook", "narration": "This is scene 0. First sentence. More text."},
            {"heading": "Analysis", "narration": "Scene 1 narration here. Extra."},
        ],
    }


def _mock_openai_client(payload: dict) -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.choices[0].message.content = json.dumps(payload)
    resp.usage = None
    client.chat.completions.create.return_value = resp
    return client


# ── _log_usage ─────────────────────────────────────────────────────────────────

class TestLogUsage:
    def test_no_op_when_none(self):
        _log_usage(None)

    def test_calls_record_llm(self):
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 30
        usage.prompt_tokens_details = None
        with patch("src.digest_script_generator.get_ledger") as mock_ledger:
            _log_usage(usage, "digest")
        mock_ledger.return_value.record_llm.assert_called_once()

    def test_zero_prompt_tokens_no_error(self):
        usage = MagicMock()
        usage.prompt_tokens = 0
        usage.completion_tokens = 5
        usage.prompt_tokens_details = None
        with patch("src.digest_script_generator.get_ledger") as mock_ledger:
            _log_usage(usage)
        mock_ledger.return_value.record_llm.assert_called_once()

    def test_handles_cached_tokens(self):
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 20
        details = MagicMock()
        details.cached_tokens = 50
        usage.prompt_tokens_details = details
        with patch("src.digest_script_generator.get_ledger") as mock_ledger:
            _log_usage(usage)
        mock_ledger.return_value.record_llm.assert_called_once()


# ── _load_daily_script ─────────────────────────────────────────────────────────

class TestLoadDailyScript:
    def test_returns_none_when_no_file(self, tmp_path):
        result = _load_daily_script(tmp_path, tmp_path, date(2026, 1, 1))
        assert result is None

    def test_loads_from_data_dir(self, tmp_path):
        day = date(2026, 5, 10)
        digest_dir = tmp_path / "digest_scripts"
        digest_dir.mkdir()
        (digest_dir / f"{day.isoformat()}.json").write_text(json.dumps({"title": "Test"}))
        result = _load_daily_script(tmp_path, tmp_path, day)
        assert result == {"title": "Test"}

    def test_loads_from_output_dir(self, tmp_path):
        day = date(2026, 5, 11)
        day_dir = tmp_path / day.isoformat()
        day_dir.mkdir()
        (day_dir / "script.json").write_text(json.dumps({"title": "Output Title"}))
        result = _load_daily_script(tmp_path, tmp_path / "data", day)
        assert result == {"title": "Output Title"}

    def test_data_dir_takes_priority_over_output_dir(self, tmp_path):
        day = date(2026, 5, 12)
        digest_dir = tmp_path / "digest_scripts"
        digest_dir.mkdir()
        (digest_dir / f"{day.isoformat()}.json").write_text(json.dumps({"title": "From data"}))
        day_dir = tmp_path / day.isoformat()
        day_dir.mkdir()
        (day_dir / "script.json").write_text(json.dumps({"title": "From output"}))
        result = _load_daily_script(tmp_path, tmp_path, day)
        assert result["title"] == "From data"

    def test_returns_none_on_invalid_json(self, tmp_path):
        day = date(2026, 5, 13)
        digest_dir = tmp_path / "digest_scripts"
        digest_dir.mkdir()
        (digest_dir / f"{day.isoformat()}.json").write_text("invalid json{")
        result = _load_daily_script(tmp_path, tmp_path, day)
        assert result is None


# ── _summarise_daily ───────────────────────────────────────────────────────────

class TestSummariseDaily:
    def test_includes_day_name(self):
        day = date(2026, 5, 11)
        data = _make_daily_data("AI News")
        result = _summarise_daily(day, data)
        assert "Monday" in result or "May 11" in result

    def test_includes_title(self):
        day = date(2026, 5, 11)
        data = _make_daily_data("GPT Breakthrough")
        result = _summarise_daily(day, data)
        assert "GPT Breakthrough" in result

    def test_includes_scene_headings(self):
        day = date(2026, 5, 11)
        data = _make_daily_data()
        result = _summarise_daily(day, data)
        assert "Hook" in result
        assert "Analysis" in result

    def test_includes_first_sentence_of_narration(self):
        day = date(2026, 5, 11)
        data = _make_daily_data()
        result = _summarise_daily(day, data)
        assert "This is scene 0." in result

    def test_skips_scene_without_heading(self):
        day = date(2026, 5, 11)
        data = {"title": "Test", "scenes": [{"heading": "", "narration": "text"}]}
        result = _summarise_daily(day, data)
        assert "text" not in result

    def test_handles_empty_scenes(self):
        day = date(2026, 5, 11)
        data = {"title": "No Scenes", "scenes": []}
        result = _summarise_daily(day, data)
        assert "No Scenes" in result


# ── save_for_digest ────────────────────────────────────────────────────────────

class TestSaveForDigest:
    def test_creates_digest_scripts_dir(self, tmp_path):
        script_path = tmp_path / "script.json"
        script_path.write_text(json.dumps({"title": "Today"}))
        save_for_digest(tmp_path, date(2026, 5, 11), script_path)
        assert (tmp_path / "digest_scripts").exists()

    def test_copies_script_content(self, tmp_path):
        data = {"title": "Test Script", "scenes": []}
        script_path = tmp_path / "script.json"
        script_path.write_text(json.dumps(data))
        save_for_digest(tmp_path, date(2026, 5, 11), script_path)
        dest = tmp_path / "digest_scripts" / "2026-05-11.json"
        assert json.loads(dest.read_text()) == data

    def test_filename_uses_date(self, tmp_path):
        script_path = tmp_path / "script.json"
        script_path.write_text("{}")
        day = date(2026, 3, 15)
        save_for_digest(tmp_path, day, script_path)
        assert (tmp_path / "digest_scripts" / "2026-03-15.json").exists()


# ── collect_week_scripts ───────────────────────────────────────────────────────

class TestCollectWeekScripts:
    def test_returns_empty_when_no_scripts(self, tmp_path):
        dates, blocks = collect_week_scripts(tmp_path, tmp_path, sunday=date(2026, 5, 17))
        assert dates == []
        assert blocks == []

    def test_collects_up_to_6_scripts(self, tmp_path):
        sunday = date(2026, 5, 17)
        digest_dir = tmp_path / "digest_scripts"
        digest_dir.mkdir()
        for i in range(1, 8):
            day = sunday - timedelta(days=i)
            (digest_dir / f"{day.isoformat()}.json").write_text(
                json.dumps(_make_daily_data(f"Title {i}"))
            )
        dates, blocks = collect_week_scripts(tmp_path, tmp_path, sunday=sunday)
        assert len(dates) == 6
        assert len(blocks) == 6

    def test_dates_are_in_chronological_order(self, tmp_path):
        sunday = date(2026, 5, 17)
        digest_dir = tmp_path / "digest_scripts"
        digest_dir.mkdir()
        for i in range(1, 4):
            day = sunday - timedelta(days=i)
            (digest_dir / f"{day.isoformat()}.json").write_text(
                json.dumps(_make_daily_data(f"T{i}"))
            )
        dates, _ = collect_week_scripts(tmp_path, tmp_path, sunday=sunday)
        assert dates == sorted(dates)

    def test_uses_today_when_no_sunday_given(self, tmp_path):
        # Should not raise
        dates, blocks = collect_week_scripts(tmp_path, tmp_path)
        assert isinstance(dates, list)


# ── generate_digest_script ─────────────────────────────────────────────────────

def _make_digest_payload(n_scenes: int = 8) -> dict:
    return {
        "title": "Week in AI Digest",
        "description": "Best AI news of the week.",
        "tags": ["AI", "digest"],
        "hook": "This week was huge.",
        "hook_variants": ["This week was huge.", "Big AI week recap"],
        "scenes": [
            {
                "heading": f"Scene {i}",
                "narration": f"Scene {i} narration content here.",
                "visual_prompt": f"visual {i}",
            }
            for i in range(n_scenes)
        ],
    }


class TestGenerateDigestScript:
    def test_raises_when_no_week_blocks(self):
        with pytest.raises(ValueError, match="No daily scripts"):
            generate_digest_script([], "2026-05-11")

    def test_returns_video_script(self):
        payload = _make_digest_payload()
        with patch("src.digest_script_generator.make_openai_client",
                   return_value=_mock_openai_client(payload)):
            result = generate_digest_script(["Week block content"], "2026-05-11")
        assert isinstance(result, VideoScript)

    def test_title_from_payload(self):
        payload = _make_digest_payload()
        with patch("src.digest_script_generator.make_openai_client",
                   return_value=_mock_openai_client(payload)):
            result = generate_digest_script(["Block"], "2026-05-11")
        assert result.title == "Week in AI Digest"

    def test_scene_count_from_payload(self):
        payload = _make_digest_payload(n_scenes=8)
        with patch("src.digest_script_generator.make_openai_client",
                   return_value=_mock_openai_client(payload)):
            result = generate_digest_script(["Block"], "2026-05-11")
        assert len(result.scenes) == 8

    def test_picks_hook_from_variants_with_data_dir(self, tmp_path):
        payload = _make_digest_payload()
        with patch("src.digest_script_generator.make_openai_client",
                   return_value=_mock_openai_client(payload)):
            with patch("src.hook_selector.pick_hook", return_value="This week was huge.") as mock_pick:
                result = generate_digest_script(["Block"], "2026-05-11", data_dir=tmp_path)
        mock_pick.assert_called_once()
        assert result.hook == "This week was huge."

    def test_uses_first_hook_when_single_variant(self):
        payload = _make_digest_payload()
        payload["hook_variants"] = ["Only hook"]
        with patch("src.digest_script_generator.make_openai_client",
                   return_value=_mock_openai_client(payload)):
            result = generate_digest_script(["Block"], "2026-05-11")
        assert result.hook == "Only hook"

    def test_uses_hook_field_when_no_variants(self):
        payload = _make_digest_payload()
        payload["hook_variants"] = []
        payload["hook"] = "Default hook"
        with patch("src.digest_script_generator.make_openai_client",
                   return_value=_mock_openai_client(payload)):
            result = generate_digest_script(["Block"], "2026-05-11")
        assert result.hook == "Default hook"

    def test_multiple_week_blocks(self):
        payload = _make_digest_payload()
        blocks = [f"Block {i}" for i in range(4)]
        with patch("src.digest_script_generator.make_openai_client",
                   return_value=_mock_openai_client(payload)):
            result = generate_digest_script(blocks, "2026-05-11")
        assert isinstance(result, VideoScript)
