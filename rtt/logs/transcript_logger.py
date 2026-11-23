"""Non-blocking two-phase transcript & translation logger (SQLite WAL + streaming flat files)."""

from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import shutil
import sqlite3
import threading
import time
import uuid


def make_session_id() -> str:
    """Create a unique session identifier prefixed with YYYY-MM-DD_HH-MM-SS."""
    return f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4().hex[:6]}"


def format_srt_time(offset_sec: float) -> str:
    """Format seconds since session start as `HH:MM:SS,mmm`."""
    total_ms = int(round(max(0.0, float(offset_sec)) * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def prune_older_than(base_dir: Path | str, days: int) -> int:
    """Delete SQLite utterance rows and session folders older than `days` days."""
    if days <= 0:
        return 0
    base = Path(base_dir)
    if not base.exists():
        return 0

    cutoff = time.time() - (days * 86400.0)
    removed = 0

    db_path = base / "sessions.db"
    if db_path.is_file():
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            cur = conn.execute("DELETE FROM utterances WHERE start_ts < ?", (cutoff,))
            removed += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            conn.commit()

    for entry in base.iterdir():
        if entry.is_dir():
            try:
                if entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed += 1
            except OSError:
                pass
    return removed


class DualTranscriptLogger:
    """Two-phase background logger writing SQLite WAL + JSONL/TXT/SRT outputs."""

    def __init__(
        self,
        base_dir: Path | str,
        session_id: str | None = None,
        target_lang: str = "en",
        mode: str = "full",
    ):
        if mode not in {"full", "source_only", "ephemeral"}:
            raise ValueError(f"Unsupported logging mode: {mode}")

        self.base_dir = Path(base_dir)
        self.session_id = session_id or make_session_id()
        self.target_lang = target_lang
        self.mode = mode
        self.session_start = time.time()

        self._entries_lock = threading.Lock()
        self.entries: dict[int, dict] = {}
        self._srt_index = 0

        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

        self.db_path = self.base_dir / "sessions.db"
        folder_prefix = self.session_id[:19] if len(self.session_id) >= 19 else self.session_id
        candidate_dir = self.base_dir / folder_prefix
        if candidate_dir.exists():
            candidate_dir = self.base_dir / self.session_id
        self.session_dir = candidate_dir
        self.jsonl_path = self.session_dir / "transcript.jsonl"
        self.txt_path = self.session_dir / "bilingual.txt"
        self.srt_path = self.session_dir / "subtitles_bilingual.srt"
        self.src_only_path = self.session_dir / "source_only.txt"

        if self.mode != "ephemeral":
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._worker = threading.Thread(
                target=self._run_writer,
                name="DualTranscriptWriter",
                daemon=True,
            )
            self._worker.start()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS utterances (
                    session_id TEXT NOT NULL,
                    seq_id INTEGER NOT NULL,
                    merged_seq_ids TEXT,
                    start_ts REAL,
                    end_ts REAL,
                    stt_engine TEXT,
                    source_lang TEXT,
                    source_text TEXT,
                    target_lang TEXT,
                    translated_text TEXT,
                    translator_provider TEXT,
                    stt_latency_ms INTEGER,
                    translation_latency_ms INTEGER,
                    status TEXT,
                    PRIMARY KEY (session_id, seq_id)
                )
                """
            )
            conn.commit()

    def snapshot_entries(self) -> list[dict]:
        with self._entries_lock:
            return [dict(self.entries[k]) for k in sorted(self.entries)]

    def log_source(
        self,
        seq_id: int,
        start_ts: float,
        end_ts: float,
        stt_engine: str,
        source_lang: str,
        source_text: str,
        stt_latency_ms: int,
    ) -> None:
        payload = {
            "session_id": self.session_id,
            "seq_id": int(seq_id),
            "merged_seq_ids": [int(seq_id)],
            "start_ts": float(start_ts),
            "end_ts": float(end_ts),
            "stt_engine": stt_engine,
            "source_lang": source_lang,
            "source_text": source_text,
            "target_lang": self.target_lang,
            "translated_text": None,
            "translator_provider": None,
            "stt_latency_ms": int(stt_latency_ms),
            "translation_latency_ms": None,
            "status": "PENDING",
        }
        with self._entries_lock:
            self.entries[int(seq_id)] = dict(payload)

        if self.mode != "ephemeral":
            self._queue.put(("SOURCE", payload))

    def log_translation(
        self,
        primary_seq_id: int,
        merged_seq_ids: list[int],
        combined_source_text: str,
        translated_text: str | None,
        provider: str,
        translation_latency_ms: int,
        status: str = "TRANSLATED",
    ) -> None:
        stored_translation = None if self.mode == "source_only" else translated_text
        payload = {
            "seq_id": int(primary_seq_id),
            "merged_seq_ids": [int(i) for i in merged_seq_ids],
            "combined_source_text": combined_source_text,
            "translated_text": stored_translation,
            "translator_provider": None if self.mode == "source_only" else provider,
            "translation_latency_ms": None if self.mode == "source_only" else int(translation_latency_ms),
            "status": "SOURCE_ONLY" if self.mode == "source_only" else status,
        }
        with self._entries_lock:
            entry = self._apply_translation_to_ram(payload)

        if self.mode != "ephemeral":
            self._queue.put(("TRANSLATION", payload, dict(entry)))

    def _apply_translation_to_ram(self, payload: dict) -> dict:
        seq = payload["seq_id"]
        entry = self.entries.get(seq)
        if entry is None:
            now = time.time()
            entry = {
                "session_id": self.session_id,
                "seq_id": seq,
                "start_ts": now,
                "end_ts": now,
                "stt_engine": "unknown",
                "source_lang": "auto",
                "source_text": payload["combined_source_text"],
                "target_lang": self.target_lang,
                "stt_latency_ms": 0,
            }
            self.entries[seq] = entry

        entry["merged_seq_ids"] = list(payload["merged_seq_ids"])
        entry["source_text"] = payload["combined_source_text"]
        entry["translated_text"] = payload["translated_text"]
        entry["translator_provider"] = payload["translator_provider"]
        entry["translation_latency_ms"] = payload["translation_latency_ms"]
        entry["status"] = payload["status"]

        for sub_seq in entry["merged_seq_ids"]:
            if sub_seq != seq and sub_seq in self.entries:
                self.entries[sub_seq]["status"] = f"MERGED_INTO_{seq}"
        return entry

    def _run_writer(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    first_item = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                batch = [first_item]
                while len(batch) < 64:
                    try:
                        batch.append(self._queue.get_nowait())
                    except queue.Empty:
                        break

                try:
                    for item in batch:
                        event_type = item[0]
                        if event_type == "SOURCE":
                            self._write_source(conn, item[1])
                        elif event_type == "TRANSLATION":
                            self._write_translation(conn, item[1], item[2])
                    conn.commit()
                finally:
                    for _ in batch:
                        self._queue.task_done()
        finally:
            conn.close()

    def _write_source(self, conn: sqlite3.Connection, payload: dict) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO utterances VALUES
            (:session_id, :seq_id, :merged_seq_ids, :start_ts, :end_ts,
             :stt_engine, :source_lang, :source_text, :target_lang,
             :translated_text, :translator_provider, :stt_latency_ms,
             :translation_latency_ms, :status)
            """,
            {**payload, "merged_seq_ids": json.dumps(payload["merged_seq_ids"])},
        )
        ts_str = (
            datetime.fromtimestamp(payload["start_ts"], tz=timezone.utc)
            .astimezone()
            .strftime("%H:%M:%S")
        )
        with self.src_only_path.open("a", encoding="utf-8") as f_src:
            f_src.write(f"[{ts_str}] ({payload['source_lang']}) {payload['source_text']}\n")

    def _write_translation(
        self,
        conn: sqlite3.Connection,
        payload: dict,
        entry: dict,
    ) -> None:
        seq = payload["seq_id"]
        conn.execute(
            """
            UPDATE utterances SET
                merged_seq_ids = ?,
                source_text = ?,
                translated_text = ?,
                translator_provider = ?,
                translation_latency_ms = ?,
                status = ?
            WHERE session_id = ? AND seq_id = ?
            """,
            (
                json.dumps(entry["merged_seq_ids"]),
                entry["source_text"],
                entry["translated_text"],
                entry["translator_provider"],
                entry["translation_latency_ms"],
                entry["status"],
                self.session_id,
                seq,
            ),
        )
        for sub_seq in entry["merged_seq_ids"]:
            if sub_seq != seq:
                conn.execute(
                    "UPDATE utterances SET status = ? WHERE session_id = ? AND seq_id = ?",
                    (f"MERGED_INTO_{seq}", self.session_id, sub_seq),
                )

        if self.mode == "full":
            self._append_flat_files(entry)

    def _append_flat_files(self, entry: dict) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f_json:
            f_json.write(json.dumps(entry, ensure_ascii=False) + "\n")

        ts_str = (
            datetime.fromtimestamp(entry["start_ts"], tz=timezone.utc)
            .astimezone()
            .strftime("%H:%M:%S")
        )
        trn_text = entry.get("translated_text") or ""
        with self.txt_path.open("a", encoding="utf-8") as f_txt:
            f_txt.write(
                f"[{ts_str}] ({entry['source_lang']} -> {entry['target_lang']})\n"
                f"  SRC: {entry['source_text']}\n"
                f"  TRN: {trn_text}\n\n"
            )

        self._srt_index += 1
        start_srt = format_srt_time(entry["start_ts"] - self.session_start)
        end_srt = format_srt_time(entry["end_ts"] - self.session_start)
        with self.srt_path.open("a", encoding="utf-8") as f_srt:
            lines = [
                str(self._srt_index),
                f"{start_srt} --> {end_srt}",
                entry["source_text"],
            ]
            if trn_text:
                lines.append(trn_text)
            f_srt.write("\n".join(lines) + "\n\n")

    def flush(self) -> None:
        """Block until all currently queued log events are committed to SQLite and disk."""
        if self._worker is not None:
            self._queue.join()

    def shutdown(self) -> None:
        """Drain pending writes and cleanly stop the background writer thread."""
        self.flush()
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=3.0)
            self._worker = None
