"""SQLite storage for evaluation run results.

Stores run metadata and per-case results so we can compare across runs,
track trends over time, and generate reports without re-running evals.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.models import (
    CaseResult,
    CategoryAccuracy,
    EmailCategory,
    EvalRunResult,
    RunMetadata,
)


_DB_DIR = Path(__file__).resolve().parent.parent / "runs"
_DB_PATH = _DB_DIR / "eval_results.db"


def get_db_path() -> Path:
    import os
    override = os.environ.get("MRD_DB_PATH")
    if override:
        return Path(override)
    return _DB_PATH


def _get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _init_tables(conn)
    return conn


def _init_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id TEXT PRIMARY KEY,
            prompt_version TEXT NOT NULL,
            model TEXT NOT NULL,
            dataset_version TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            total_cases INTEGER DEFAULT 0,
            passed_cases INTEGER DEFAULT 0,
            failed_cases INTEGER DEFAULT 0,
            error_cases INTEGER DEFAULT 0,
            overall_accuracy REAL DEFAULT 0.0,
            avg_summary_score REAL DEFAULT 0.0,
            avg_latency_ms REAL DEFAULT 0.0,
            total_tokens_used INTEGER DEFAULT 0,
            total_cost_estimate_usd REAL DEFAULT 0.0,
            per_category_accuracy_json TEXT DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS case_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            case_id TEXT NOT NULL,
            input_email TEXT NOT NULL,
            expected_category TEXT NOT NULL,
            predicted_category TEXT,
            category_match INTEGER DEFAULT 0,
            expected_summary TEXT NOT NULL,
            predicted_summary TEXT DEFAULT '',
            summary_relevance_score REAL DEFAULT 0.0,
            latency_ms REAL DEFAULT 0.0,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            raw_response TEXT DEFAULT '',
            error TEXT,
            confidence REAL DEFAULT 0.0,
            FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_case_results_run_id
            ON case_results(run_id);
        CREATE INDEX IF NOT EXISTS idx_case_results_case_id
            ON case_results(case_id);
    """)
    conn.commit()


def save_eval_run(result: EvalRunResult, db_path: Path | None = None) -> None:
    """Persist a complete eval run to SQLite."""
    conn = _get_connection(db_path)
    try:
        meta = result.metadata
        conn.execute(
            """
            INSERT OR REPLACE INTO eval_runs
            (run_id, prompt_version, model, dataset_version, started_at,
             finished_at, total_cases, passed_cases, failed_cases, error_cases,
             overall_accuracy, avg_summary_score, avg_latency_ms,
             total_tokens_used, total_cost_estimate_usd, per_category_accuracy_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meta.run_id, meta.prompt_version, meta.model,
                meta.dataset_version, meta.started_at.isoformat(),
                meta.finished_at.isoformat() if meta.finished_at else None,
                meta.total_cases, meta.passed_cases, meta.failed_cases,
                meta.error_cases, result.overall_accuracy,
                result.avg_summary_score, result.avg_latency_ms,
                result.total_tokens_used, result.total_cost_estimate_usd,
                json.dumps([ca.model_dump() for ca in result.per_category_accuracy]),
            ),
        )

        for cr in result.case_results:
            conn.execute(
                """
                INSERT INTO case_results
                (run_id, case_id, input_email, expected_category,
                 predicted_category, category_match, expected_summary,
                 predicted_summary, summary_relevance_score, latency_ms,
                 prompt_tokens, completion_tokens, total_tokens,
                 raw_response, error, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meta.run_id, cr.case_id, cr.input_email,
                    cr.expected_category.value,
                    cr.predicted_category.value if cr.predicted_category else None,
                    1 if cr.category_match else 0, cr.expected_summary,
                    cr.predicted_summary, cr.summary_relevance_score,
                    cr.latency_ms, cr.prompt_tokens, cr.completion_tokens,
                    cr.total_tokens, cr.raw_response, cr.error, cr.confidence,
                ),
            )

        conn.commit()
    finally:
        conn.close()


def load_eval_run(run_id: str, db_path: Path | None = None) -> EvalRunResult | None:
    """Load a complete eval run from SQLite."""
    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM eval_runs WHERE run_id = ?", (run_id,)
        ).fetchone()

        if not row:
            return None

        meta = RunMetadata(
            run_id=row["run_id"],
            prompt_version=row["prompt_version"],
            model=row["model"],
            dataset_version=row["dataset_version"],
            started_at=datetime.fromisoformat(row["started_at"]),
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
            total_cases=row["total_cases"],
            passed_cases=row["passed_cases"],
            failed_cases=row["failed_cases"],
            error_cases=row["error_cases"],
        )

        case_rows = conn.execute(
            "SELECT * FROM case_results WHERE run_id = ? ORDER BY case_id",
            (run_id,),
        ).fetchall()

        case_results = []
        for cr in case_rows:
            case_results.append(
                CaseResult(
                    case_id=cr["case_id"],
                    input_email=cr["input_email"],
                    expected_category=cr["expected_category"],
                    predicted_category=cr["predicted_category"],
                    category_match=bool(cr["category_match"]),
                    expected_summary=cr["expected_summary"],
                    predicted_summary=cr["predicted_summary"],
                    summary_relevance_score=cr["summary_relevance_score"],
                    latency_ms=cr["latency_ms"],
                    prompt_tokens=cr["prompt_tokens"],
                    completion_tokens=cr["completion_tokens"],
                    total_tokens=cr["total_tokens"],
                    raw_response=cr["raw_response"],
                    error=cr["error"],
                    confidence=cr["confidence"],
                )
            )

        per_cat = json.loads(row["per_category_accuracy_json"])
        per_category_accuracy = [CategoryAccuracy(**ca) for ca in per_cat]

        return EvalRunResult(
            metadata=meta,
            case_results=case_results,
            overall_accuracy=row["overall_accuracy"],
            per_category_accuracy=per_category_accuracy,
            avg_summary_score=row["avg_summary_score"],
            avg_latency_ms=row["avg_latency_ms"],
            total_tokens_used=row["total_tokens_used"],
            total_cost_estimate_usd=row["total_cost_estimate_usd"],
        )
    finally:
        conn.close()


def get_latest_run_id(db_path: Path | None = None) -> str | None:
    """Get the run_id of the most recent eval run."""
    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT run_id FROM eval_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return row["run_id"] if row else None
    finally:
        conn.close()


def list_runs(limit: int = 20, db_path: Path | None = None) -> list[dict]:
    """List recent eval runs with summary info."""
    conn = _get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT run_id, prompt_version, model, dataset_version,
                   started_at, total_cases, passed_cases, overall_accuracy,
                   avg_summary_score, avg_latency_ms
            FROM eval_runs
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
