from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "1"


class ReviewQueue:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_items (
                    record_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    route TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    selected_code TEXT,
                    reviewer_id_hash TEXT,
                    reason TEXT,
                    payload_json TEXT,
                    created_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                """
            )
            con.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('schema_version',?)", (SCHEMA_VERSION,))

    def enqueue(self, packets: Iterable[Mapping[str, Any]]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        count = 0
        with self._connect() as con:
            for packet in packets:
                record_id = str(packet.get("record_id", ""))
                if not record_id:
                    continue
                con.execute(
                    "INSERT OR REPLACE INTO review_items(record_id,payload_json,route,status,created_at_utc,updated_at_utc) VALUES(?,?,?,?,?,?)",
                    (record_id, json.dumps(dict(packet), ensure_ascii=False), str(packet.get("route", "")), "PENDING", now, now),
                )
                count += 1
        return count

    def pending(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT payload_json,status FROM review_items WHERE status='PENDING' ORDER BY created_at_utc LIMIT ?", (int(limit),)
            ).fetchall()
        out = []
        for payload_json, status in rows:
            payload = json.loads(payload_json)
            payload["review_status"] = status
            out.append(payload)
        return out

    def decide(
        self,
        record_id: str,
        *,
        action: str,
        selected_code: str = "",
        reviewer_id_hash: str = "",
        reason: str = "",
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        allowed = {"ACCEPT_TOP1", "SELECT_ALTERNATIVE", "RECODE_OUTSIDE_TOPK", "ESCALATE", "NO_CODE"}
        if action not in allowed:
            raise ValueError(f"Invalid review action: {action}")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as con:
            exists = con.execute("SELECT 1 FROM review_items WHERE record_id=?", (str(record_id),)).fetchone()
            if not exists:
                raise KeyError(record_id)
            con.execute(
                "INSERT INTO review_events(record_id,action,selected_code,reviewer_id_hash,reason,payload_json,created_at_utc) VALUES(?,?,?,?,?,?,?)",
                (str(record_id), action, str(selected_code), str(reviewer_id_hash), str(reason), json.dumps(dict(extra or {}), ensure_ascii=False), now),
            )
            status = "ESCALATED" if action == "ESCALATE" else "RESOLVED"
            con.execute("UPDATE review_items SET status=?, updated_at_utc=? WHERE record_id=?", (status, now, str(record_id)))

    def audit_trail(self, record_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT event_id,record_id,action,selected_code,reviewer_id_hash,reason,payload_json,created_at_utc FROM review_events"
        params: tuple[Any, ...] = ()
        if record_id is not None:
            query += " WHERE record_id=?"
            params = (str(record_id),)
        query += " ORDER BY event_id"
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
        keys = ["event_id","record_id","action","selected_code","reviewer_id_hash","reason","payload_json","created_at_utc"]
        return [dict(zip(keys, row)) for row in rows]

    def summary(self) -> dict[str, Any]:
        with self._connect() as con:
            status = dict(con.execute("SELECT status,COUNT(*) FROM review_items GROUP BY status").fetchall())
            actions = dict(con.execute("SELECT action,COUNT(*) FROM review_events GROUP BY action").fetchall())
        return {"schema_version": SCHEMA_VERSION, "status_counts": status, "action_counts": actions}
