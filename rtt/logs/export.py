"""Session querying and export utilities (SRT, TXT, CSV, JSON, JSONL)."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from rtt.logs.transcript_logger import format_srt_time

CSV_COLUMNS = [
    "seq_id",
    "merged_seq_ids",
    "start_ts",
    "end_ts",
    "stt_engine",
    "source_lang",
    "source_text",
    "target_lang",
    "translated_text",
    "translator_provider",
    "stt_latency_ms",
    "translation_latency_ms",
    "status",
]


def _row_to_dict(row: sqlite3.Row) -> dict:
    item = dict(row)
    raw_merged = item.get("merged_seq_ids")
    if isinstance(raw_merged, str):
        try:
            item["merged_seq_ids"] = json.loads(raw_merged)
        except json.JSONDecodeError:
            item["merged_seq_ids"] = [item.get("seq_id")]
    return item


def export_entries(
    entries: list[dict],
    fmt: str,
    out_path: Path | str,
    session_start: float | None = None,
) -> Path:
    """Export in-memory or queried utterance dicts to srt/txt/csv/json/jsonl."""
    fmt_norm = fmt.lower().lstrip(".")
    if fmt_norm not in {"srt", "txt", "csv", "json", "jsonl"}:
        raise ValueError(f"Unsupported export format: {fmt}")

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    active = [
        dict(e)
        for e in sorted(entries, key=lambda x: int(x.get("seq_id", 0)))
        if not str(e.get("status") or "").startswith("MERGED_INTO_")
    ]

    if session_start is None and active:
        session_start = float(active[0].get("start_ts") or 0.0)
    base_ts = float(session_start or 0.0)

    if fmt_norm == "json":
        target.write_text(json.dumps(active, indent=2, ensure_ascii=False), encoding="utf-8")
    elif fmt_norm == "jsonl":
        lines = [json.dumps(e, ensure_ascii=False) for e in active]
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    elif fmt_norm == "csv":
        with target.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for e in active:
                row = dict(e)
                if isinstance(row.get("merged_seq_ids"), list):
                    row["merged_seq_ids"] = json.dumps(row["merged_seq_ids"])
                writer.writerow(row)
    elif fmt_norm == "txt":
        chunks: list[str] = []
        for e in active:
            ts = (
                datetime.fromtimestamp(float(e.get("start_ts") or 0.0), tz=timezone.utc)
                .astimezone()
                .strftime("%H:%M:%S")
            )
            src_lang = e.get("source_lang") or "auto"
            tgt_lang = e.get("target_lang") or ""
            src_txt = e.get("source_text") or ""
            trn_txt = e.get("translated_text") or ""
            chunks.append(
                f"[{ts}] ({src_lang} -> {tgt_lang})\n  SRC: {src_txt}\n  TRN: {trn_txt}\n"
            )
        target.write_text("\n".join(chunks), encoding="utf-8")
    elif fmt_norm == "srt":
        blocks: list[str] = []
        for idx, e in enumerate(active, start=1):
            start_s = format_srt_time(float(e.get("start_ts") or base_ts) - base_ts)
            end_s = format_srt_time(float(e.get("end_ts") or base_ts) - base_ts)
            lines = [str(idx), f"{start_s} --> {end_s}", str(e.get("source_text") or "")]
            if e.get("translated_text"):
                lines.append(str(e["translated_text"]))
            blocks.append("\n".join(lines) + "\n")
        target.write_text("\n".join(blocks), encoding="utf-8")

    return target


def export_session(
    db_path: Path | str,
    session_id: str,
    fmt: str,
    out_path: Path | str,
) -> Path:
    """Load utterances for `session_id` from SQLite and export to `out_path`."""
    db = Path(db_path)
    if not db.is_file():
        raise FileNotFoundError(f"Database not found: {db}")

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM utterances WHERE session_id = ? ORDER BY seq_id ASC",
            (session_id,),
        ).fetchall()
    entries = [_row_to_dict(r) for r in rows]
    return export_entries(entries, fmt=fmt, out_path=out_path)


def search(db_path: Path | str, query: str, limit: int = 200) -> list[dict]:
    """Search past utterances in SQLite across source and translated text."""
    db = Path(db_path)
    if not db.is_file():
        return []
    q = f"%{(query or '').strip()}%"
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT * FROM utterances
                WHERE (source_text LIKE ? OR translated_text LIKE ?)
                  AND COALESCE(status, '') NOT LIKE 'MERGED_INTO_%'
                ORDER BY start_ts DESC, seq_id DESC
                LIMIT ?
                """,
                (q, q, int(limit)),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [_row_to_dict(r) for r in rows]


def list_sessions(db_path: Path | str) -> list[dict]:
    """Return summary metadata for all logged sessions in SQLite."""
    db = Path(db_path)
    if not db.is_file():
        return []
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT session_id,
                       MIN(start_ts) AS started_at,
                       MAX(end_ts) AS ended_at,
                       COUNT(*) AS count
                FROM utterances
                GROUP BY session_id
                ORDER BY started_at DESC
                """
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [dict(r) for r in rows]
