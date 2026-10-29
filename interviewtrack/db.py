"""SQLite persistence with per-user isolation.

Design notes:
  * Every row is scoped by `user_id`; every query filters on it.
  * Stable IDs + UNIQUE constraints make re-synchronization idempotent: the same
    Gmail message can be stored repeatedly without creating duplicates.
  * Email bodies are stored locally only (never logged, never sent anywhere), and
    `delete_user_data` removes everything for a user.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from .config import get_settings
from .models import Application, Correction, InterviewEvent, RawMessage, RecruiterContact

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    email         TEXT NOT NULL,
    display_name  TEXT DEFAULT '',
    picture       TEXT DEFAULT '',
    created_at    TEXT NOT NULL,
    last_login_at TEXT
);

CREATE TABLE IF NOT EXISTS oauth_tokens (
    user_id       TEXT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    token_json    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    user_id         TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    gmail_id        TEXT NOT NULL,
    thread_id       TEXT NOT NULL,
    rfc_message_id  TEXT,
    in_reply_to     TEXT,
    references_json TEXT DEFAULT '[]',
    from_addr       TEXT DEFAULT '',
    from_name       TEXT DEFAULT '',
    to_json         TEXT DEFAULT '[]',
    cc_json         TEXT DEFAULT '[]',
    subject         TEXT DEFAULT '',
    date            TEXT,
    snippet         TEXT DEFAULT '',
    labels_json     TEXT DEFAULT '[]',
    body_text       TEXT DEFAULT '',
    body_html       TEXT DEFAULT '',
    body_fetched    INTEGER DEFAULT 0,
    PRIMARY KEY (user_id, gmail_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(user_id, thread_id);
CREATE INDEX IF NOT EXISTS idx_messages_date   ON messages(user_id, date);

CREATE TABLE IF NOT EXISTS applications (
    user_id              TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    app_key              TEXT NOT NULL,
    employer             TEXT NOT NULL,
    employer_canonical   TEXT DEFAULT '',
    job_title            TEXT,
    job_title_canonical  TEXT,
    role_family          TEXT,
    requisition_id       TEXT,
    current_stage        TEXT DEFAULT '',
    status               TEXT DEFAULT '',
    latest_activity_at   TEXT,
    latest_activity_label TEXT DEFAULT '',
    last_milestone       TEXT DEFAULT '',
    match_confidence     REAL DEFAULT 0,
    needs_review         INTEGER DEFAULT 0,
    history_incomplete   INTEGER DEFAULT 0,
    notes_json           TEXT DEFAULT '[]',
    recruiters_json      TEXT DEFAULT '[]',
    PRIMARY KEY (user_id, app_key)
);

CREATE TABLE IF NOT EXISTS interview_events (
    user_id        TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    app_key        TEXT NOT NULL,
    gmail_id       TEXT NOT NULL,
    thread_id      TEXT NOT NULL,
    rfc_message_id TEXT,
    occurred_at    TEXT,
    event_type     TEXT NOT NULL,
    occurrence     TEXT NOT NULL,
    sender         TEXT DEFAULT '',
    sender_name    TEXT DEFAULT '',
    subject        TEXT DEFAULT '',
    preview        TEXT DEFAULT '',
    evidence       TEXT DEFAULT '',
    confidence     REAL DEFAULT 0,
    event_datetime TEXT,
    -- one message appears at most once per application
    PRIMARY KEY (user_id, app_key, gmail_id)
);
CREATE INDEX IF NOT EXISTS idx_events_app ON interview_events(user_id, app_key);

CREATE TABLE IF NOT EXISTS match_decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    gmail_id      TEXT NOT NULL,
    app_key       TEXT,
    kind          TEXT NOT NULL,
    score         REAL DEFAULT 0,
    features_json TEXT DEFAULT '{}',
    reasons_json  TEXT DEFAULT '[]',
    blockers_json TEXT DEFAULT '[]',
    source        TEXT DEFAULT '',
    created_at    TEXT NOT NULL,
    UNIQUE (user_id, gmail_id, app_key, kind)
);

CREATE TABLE IF NOT EXISTS corrections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    gmail_id    TEXT,
    app_key     TEXT,
    target_key  TEXT,
    payload_json TEXT DEFAULT '{}',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corrections_user ON corrections(user_id, created_at);

CREATE TABLE IF NOT EXISTS sync_state (
    user_id        TEXT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    last_sync_at   TEXT,
    history_id     TEXT,
    message_count  INTEGER DEFAULT 0,
    window_start   TEXT,
    notes          TEXT DEFAULT ''
);
"""


def _iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class Database:
    """Thin, explicit SQLite wrapper. No ORM, no magic."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else get_settings().db_path
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # check_same_thread=False: Streamlit reruns across threads.
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        if str(self.path) != ":memory:":
            self.path.chmod(0o600)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # --- users ---------------------------------------------------------------

    def upsert_user(self, user_id: str, email: str, display_name: str = "",
                    picture: str = "") -> None:
        now = _iso(datetime.now(timezone.utc))
        with self.tx() as c:
            c.execute(
                """INSERT INTO users (user_id, email, display_name, picture,
                                      created_at, last_login_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(user_id) DO UPDATE SET
                     email=excluded.email,
                     display_name=excluded.display_name,
                     picture=excluded.picture,
                     last_login_at=excluded.last_login_at""",
                (user_id, email, display_name, picture, now, now),
            )

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    # --- oauth tokens --------------------------------------------------------

    def save_token(self, user_id: str, token: dict[str, Any]) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO oauth_tokens (user_id, token_json, updated_at)
                   VALUES (?,?,?)
                   ON CONFLICT(user_id) DO UPDATE SET
                     token_json=excluded.token_json, updated_at=excluded.updated_at""",
                (user_id, json.dumps(token), _iso(datetime.now(timezone.utc))),
            )

    def load_token(self, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT token_json FROM oauth_tokens WHERE user_id = ?", (user_id,)
            ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["token_json"])
        except json.JSONDecodeError:
            return None

    def delete_token(self, user_id: str) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM oauth_tokens WHERE user_id = ?", (user_id,))

    # --- messages ------------------------------------------------------------

    def upsert_messages(self, user_id: str, messages: Sequence[RawMessage]) -> int:
        """Store raw messages. Idempotent on (user_id, gmail_id)."""
        rows = [
            (
                user_id, m.gmail_id, m.thread_id, m.rfc_message_id, m.in_reply_to,
                json.dumps(m.references), m.from_addr, m.from_name,
                json.dumps(m.to_addrs), json.dumps(m.cc_addrs), m.subject,
                _iso(m.date), m.snippet, json.dumps(m.labels),
                m.body_text, m.body_html, int(m.body_fetched),
            )
            for m in messages
        ]
        with self.tx() as c:
            c.executemany(
                """INSERT INTO messages (
                     user_id, gmail_id, thread_id, rfc_message_id, in_reply_to,
                     references_json, from_addr, from_name, to_json, cc_json,
                     subject, date, snippet, labels_json, body_text, body_html,
                     body_fetched)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(user_id, gmail_id) DO UPDATE SET
                     thread_id=excluded.thread_id,
                     rfc_message_id=COALESCE(excluded.rfc_message_id, messages.rfc_message_id),
                     subject=excluded.subject,
                     snippet=excluded.snippet,
                     labels_json=excluded.labels_json,
                     -- never overwrite a fetched body with an empty metadata-only one
                     body_text=CASE WHEN excluded.body_fetched=1
                                    THEN excluded.body_text ELSE messages.body_text END,
                     body_html=CASE WHEN excluded.body_fetched=1
                                    THEN excluded.body_html ELSE messages.body_html END,
                     body_fetched=MAX(excluded.body_fetched, messages.body_fetched)""",
                rows,
            )
        return len(rows)

    def load_messages(self, user_id: str) -> list[RawMessage]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE user_id = ? ORDER BY date ASC", (user_id,)
            ).fetchall()
        out: list[RawMessage] = []
        for r in rows:
            out.append(
                RawMessage(
                    gmail_id=r["gmail_id"], thread_id=r["thread_id"],
                    rfc_message_id=r["rfc_message_id"], in_reply_to=r["in_reply_to"],
                    references=json.loads(r["references_json"] or "[]"),
                    from_addr=r["from_addr"] or "", from_name=r["from_name"] or "",
                    to_addrs=json.loads(r["to_json"] or "[]"),
                    cc_addrs=json.loads(r["cc_json"] or "[]"),
                    subject=r["subject"] or "", date=_parse_dt(r["date"]),
                    snippet=r["snippet"] or "",
                    labels=json.loads(r["labels_json"] or "[]"),
                    body_text=r["body_text"] or "", body_html=r["body_html"] or "",
                    body_fetched=bool(r["body_fetched"]),
                )
            )
        return out

    def known_gmail_ids(self, user_id: str) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT gmail_id FROM messages WHERE user_id = ?", (user_id,)
            ).fetchall()
        return {r["gmail_id"] for r in rows}

    # --- applications --------------------------------------------------------

    def replace_applications(self, user_id: str, apps: Sequence[Application]) -> None:
        """Persist the reconstruction output for a user.

        Reconstruction is deterministic and derived from stored messages plus stored
        corrections, so the derived tables are rewritten wholesale. Messages and
        corrections (the sources of truth) are never touched here.
        """
        with self.tx() as c:
            c.execute("DELETE FROM interview_events WHERE user_id = ?", (user_id,))
            c.execute("DELETE FROM applications WHERE user_id = ?", (user_id,))
            for app in apps:
                c.execute(
                    """INSERT INTO applications (
                         user_id, app_key, employer, employer_canonical, job_title,
                         job_title_canonical, role_family, requisition_id,
                         current_stage, status, latest_activity_at,
                         latest_activity_label, last_milestone, match_confidence,
                         needs_review, history_incomplete, notes_json, recruiters_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        user_id, app.key, app.employer, app.employer_canonical,
                        app.job_title, app.job_title_canonical, app.role_family,
                        app.requisition_id, app.current_stage.value, app.status.value,
                        _iso(app.latest_activity_at), app.latest_activity_label,
                        app.last_supported_milestone, app.match_confidence,
                        int(app.needs_review), int(app.history_possibly_incomplete),
                        json.dumps(app.notes),
                        json.dumps([r.model_dump() for r in app.recruiters]),
                    ),
                )
                c.executemany(
                    """INSERT OR REPLACE INTO interview_events (
                         user_id, app_key, gmail_id, thread_id, rfc_message_id,
                         occurred_at, event_type, occurrence, sender, sender_name,
                         subject, preview, evidence, confidence, event_datetime)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    [
                        (
                            user_id, app.key, e.gmail_id, e.thread_id, e.rfc_message_id,
                            _iso(e.occurred_at), e.event_type.value, e.occurrence.value,
                            e.sender, e.sender_name, e.subject, e.preview, e.evidence,
                            e.confidence, _iso(e.event_datetime),
                        )
                        for e in app.events
                    ],
                )

    def load_applications(self, user_id: str) -> list[Application]:
        from .enums import ApplicationStatus, EventType, Occurrence, Stage

        with self._lock:
            app_rows = self._conn.execute(
                "SELECT * FROM applications WHERE user_id = ?", (user_id,)
            ).fetchall()
            ev_rows = self._conn.execute(
                "SELECT * FROM interview_events WHERE user_id = ?", (user_id,)
            ).fetchall()

        events_by_app: dict[str, list[InterviewEvent]] = {}
        for r in ev_rows:
            events_by_app.setdefault(r["app_key"], []).append(
                InterviewEvent(
                    gmail_id=r["gmail_id"], thread_id=r["thread_id"],
                    rfc_message_id=r["rfc_message_id"],
                    occurred_at=_parse_dt(r["occurred_at"]),
                    event_type=EventType(r["event_type"]),
                    occurrence=Occurrence(r["occurrence"]),
                    sender=r["sender"] or "", sender_name=r["sender_name"] or "",
                    subject=r["subject"] or "", preview=r["preview"] or "",
                    evidence=r["evidence"] or "", confidence=r["confidence"] or 0.0,
                    event_datetime=_parse_dt(r["event_datetime"]),
                )
            )

        out: list[Application] = []
        for r in app_rows:
            events = events_by_app.get(r["app_key"], [])
            out.append(
                Application(
                    key=r["app_key"], user_id=user_id, employer=r["employer"],
                    employer_canonical=r["employer_canonical"] or "",
                    job_title=r["job_title"],
                    job_title_canonical=r["job_title_canonical"],
                    role_family=r["role_family"], requisition_id=r["requisition_id"],
                    events=events,
                    thread_ids=sorted({e.thread_id for e in events}),
                    recruiters=[
                        RecruiterContact(**d)
                        for d in json.loads(r["recruiters_json"] or "[]")
                    ],
                    current_stage=Stage(r["current_stage"]) if r["current_stage"] else Stage.OUTREACH,
                    status=ApplicationStatus(r["status"]) if r["status"] else ApplicationStatus.ACTIVE,
                    latest_activity_at=_parse_dt(r["latest_activity_at"]),
                    latest_activity_label=r["latest_activity_label"] or "",
                    last_supported_milestone=r["last_milestone"] or "",
                    match_confidence=r["match_confidence"] or 0.0,
                    needs_review=bool(r["needs_review"]),
                    history_possibly_incomplete=bool(r["history_incomplete"]),
                    notes=json.loads(r["notes_json"] or "[]"),
                )
            )
        out.sort(
            key=lambda a: a.latest_activity_at or datetime(1970, 1, 1, tzinfo=timezone.utc),
            reverse=True,
        )
        return out

    # --- decisions -----------------------------------------------------------

    def save_decisions(self, user_id: str, decisions: Sequence[Any]) -> None:
        now = _iso(datetime.now(timezone.utc))
        with self.tx() as c:
            c.execute("DELETE FROM match_decisions WHERE user_id = ?", (user_id,))
            c.executemany(
                """INSERT OR IGNORE INTO match_decisions (
                     user_id, gmail_id, app_key, kind, score, features_json,
                     reasons_json, blockers_json, source, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        user_id, d.gmail_id, d.application_key, d.kind.value, d.score,
                        d.features.model_dump_json(), json.dumps(d.reasons),
                        json.dumps(d.blockers), d.source.value, now,
                    )
                    for d in decisions
                ],
            )

    def load_decisions(self, user_id: str, gmail_id: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM match_decisions WHERE user_id = ?"
        params: list[Any] = [user_id]
        if gmail_id:
            q += " AND gmail_id = ?"
            params.append(gmail_id)
        q += " ORDER BY score DESC"
        with self._lock:
            return [dict(r) for r in self._conn.execute(q, params).fetchall()]

    # --- corrections (sticky) ------------------------------------------------

    def add_correction(self, correction: Correction) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO corrections (user_id, kind, gmail_id, app_key,
                                            target_key, payload_json, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    correction.user_id, correction.kind, correction.gmail_id,
                    correction.application_key, correction.target_key,
                    json.dumps(correction.payload), _iso(correction.created_at),
                ),
            )

    def load_corrections(self, user_id: str) -> list[Correction]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM corrections WHERE user_id = ? ORDER BY created_at ASC, id ASC",
                (user_id,),
            ).fetchall()
        return [
            Correction(
                user_id=r["user_id"], kind=r["kind"], gmail_id=r["gmail_id"],
                application_key=r["app_key"], target_key=r["target_key"],
                payload=json.loads(r["payload_json"] or "{}"),
                created_at=_parse_dt(r["created_at"]) or datetime.now(timezone.utc),
            )
            for r in rows
        ]

    def clear_corrections(self, user_id: str) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM corrections WHERE user_id = ?", (user_id,))

    # --- sync state ----------------------------------------------------------

    def set_sync_state(self, user_id: str, message_count: int,
                       window_start: datetime | None = None, notes: str = "") -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO sync_state (user_id, last_sync_at, message_count,
                                           window_start, notes)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(user_id) DO UPDATE SET
                     last_sync_at=excluded.last_sync_at,
                     message_count=excluded.message_count,
                     window_start=excluded.window_start,
                     notes=excluded.notes""",
                (
                    user_id, _iso(datetime.now(timezone.utc)), message_count,
                    _iso(window_start), notes,
                ),
            )

    def get_sync_state(self, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sync_state WHERE user_id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    # --- deletion ------------------------------------------------------------

    def delete_user_data(self, user_id: str) -> dict[str, int]:
        """Erase everything stored for a user. Required by section 10 of the spec."""
        counts: dict[str, int] = {}
        tables = ("interview_events", "applications", "match_decisions", "corrections",
                  "messages", "sync_state", "oauth_tokens")
        with self.tx() as c:
            for table in tables:
                cur = c.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
                counts[table] = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            cur = c.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            counts["users"] = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        return counts


_db: Database | None = None


def get_db(path: str | Path | None = None) -> Database:
    global _db
    if _db is None or path is not None:
        _db = Database(path)
    return _db
