"""Tests for DualTranscriptLogger modes, WAL DB, flat files, export, search, and pruning."""

import csv
import json
import os
import sqlite3
import time

from rtt.logs.export import export_entries, export_session, list_sessions, search
from rtt.logs.transcript_logger import DualTranscriptLogger, format_srt_time, prune_older_than


def test_full_logging_writes_wal_files_merged_status_and_exports(tmp_path):
    base = tmp_path / "logs"
    logger = DualTranscriptLogger(base, session_id="2026-11-24_00-00-00_test1", target_lang="en", mode="full")
    t0 = logger.session_start
    logger.log_source(1, t0 + 1.0, t0 + 2.5, "WhisperEngine(small/cuda)", "ja", "おはよう", 110)
    logger.log_source(2, t0 + 2.6, t0 + 3.8, "WhisperEngine(small/cuda)", "ja", "ございます", 95)
    logger.log_translation(1, [1, 2], "おはよう ございます", "Good morning", "LLM(gpt-4o-mini)", 320, "TRANSLATED")
    logger.flush()
    logger.shutdown()

    assert logger.db_path.is_file()
    with sqlite3.connect(logger.db_path) as conn:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() == "wal"
        rows = conn.execute("SELECT seq_id, status, translated_text FROM utterances ORDER BY seq_id").fetchall()
        assert rows == [(1, "TRANSLATED", "Good morning"), (2, "MERGED_INTO_1", None)]

    assert "おはよう" in logger.src_only_path.read_text(encoding="utf-8")
    assert "Good morning" in logger.txt_path.read_text(encoding="utf-8")
    srt_text = logger.srt_path.read_text(encoding="utf-8")
    assert "-->" in srt_text
    assert format_srt_time(1.9999) == "00:00:02,000"
    jsonl_lines = logger.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl_lines) == 1
    assert json.loads(jsonl_lines[0])["merged_seq_ids"] == [1, 2]

    # Search & list_sessions
    hits = search(logger.db_path, "morning")
    assert len(hits) == 1 and hits[0]["seq_id"] == 1
    sessions = list_sessions(logger.db_path)
    assert len(sessions) == 1 and sessions[0]["count"] == 2

    # Export formats
    out_srt = export_session(logger.db_path, logger.session_id, "srt", tmp_path / "out.srt")
    assert "Good morning" in out_srt.read_text(encoding="utf-8")
    out_csv = export_session(logger.db_path, logger.session_id, "csv", tmp_path / "out.csv")
    with out_csv.open(encoding="utf-8-sig") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 1 and csv_rows[0]["translated_text"] == "Good morning"
    out_json = export_session(logger.db_path, logger.session_id, "json", tmp_path / "out.json")
    assert len(json.loads(out_json.read_text(encoding="utf-8"))) == 1


def test_source_only_and_ephemeral_modes(tmp_path):
    base_src = tmp_path / "src_logs"
    src_logger = DualTranscriptLogger(base_src, session_id="2026-11-24_00-00-01_src", target_lang="en", mode="source_only")
    t0 = src_logger.session_start
    src_logger.log_source(1, t0, t0 + 1.0, "WhisperEngine", "ja", "テスト", 50)
    src_logger.log_translation(1, [1], "テスト", "Test", "LLM", 100, "TRANSLATED")
    src_logger.shutdown()

    assert src_logger.src_only_path.is_file()
    assert not src_logger.txt_path.exists()
    assert not src_logger.srt_path.exists()
    with sqlite3.connect(src_logger.db_path) as conn:
        row = conn.execute("SELECT translated_text, status FROM utterances WHERE seq_id=1").fetchone()
        assert row == (None, "SOURCE_ONLY")

    base_eph = tmp_path / "eph_logs"
    eph_logger = DualTranscriptLogger(base_eph, session_id="ephemeral_test", target_lang="en", mode="ephemeral")
    eph_logger.log_source(1, t0, t0 + 1.0, "WhisperEngine", "ja", "秘密", 40)
    eph_logger.log_translation(1, [1], "秘密", "Secret", "LLM", 80, "TRANSLATED")
    eph_logger.shutdown()

    assert not base_eph.exists()
    entries = eph_logger.snapshot_entries()
    assert len(entries) == 1 and entries[0]["translated_text"] == "Secret"
    exported = export_entries(entries, "txt", tmp_path / "eph.txt", session_start=t0)
    assert "Secret" in exported.read_text(encoding="utf-8")


def test_prune_older_than_removes_old_rows_and_folders(tmp_path):
    base = tmp_path / "prune_logs"
    old_logger = DualTranscriptLogger(base, session_id="2026-10-01_00-00-00_old", target_lang="en", mode="full")
    old_ts = time.time() - (45 * 86400)
    old_logger.log_source(1, old_ts, old_ts + 1.0, "Whisper", "en", "old line", 10)
    old_logger.shutdown()
    os.utime(old_logger.session_dir, (old_ts, old_ts))

    new_logger = DualTranscriptLogger(base, session_id="2026-11-24_00-00-00_new", target_lang="en", mode="full")
    new_ts = time.time()
    new_logger.log_source(1, new_ts, new_ts + 1.0, "Whisper", "en", "new line", 10)
    new_logger.shutdown()

    prune_older_than(base, days=30)
    assert not old_logger.session_dir.exists()
    assert new_logger.session_dir.exists()
    assert len(search(base / "sessions.db", "old line")) == 0
    assert len(search(base / "sessions.db", "new line")) == 1
