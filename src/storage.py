"""Storage layer for evaluation runs.

Provides dual storage:
1. A SQLite database (for fast indexing, history query, and baseline comparisons).
2. JSON files (for portable, git-friendly, detailed individual run records).
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models import (
    CaseResult,
    CategoryAccuracy,
    EmailCategory,
    EvalRunResult,
    RunMetadata,
)


def get_default_db_path() -> Path:
    """Get default database path in the data directory, allowing override via env."""
    override = os.environ.get("MRD_DB_PATH")
    if override:
        return Path(override)
    
    # Default to data/results.db
    data_dir = os.environ.get("MRD_DATA_DIR")
    if data_dir:
        base_dir = Path(data_dir)
    else:
        base_dir = Path(__file__).resolve().parent.parent / "data"
    
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "results.db"


def get_default_results_dir() -> Path:
    """Get default results directory for JSON records, allowing override via env."""
    override = os.environ.get("MRD_RESULTS_DIR")
    if override:
        return Path(override)
    
    data_dir = os.environ.get("MRD_DATA_DIR")
    if data_dir:
        base_dir = Path(data_dir)
    else:
        base_dir = Path(__file__).resolve().parent.parent / "data"
    
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


class EvaluationStorage:
    """Orchestrates SQLite database and JSON file persistence for eval runs."""

    def __init__(self, db_path: Path | None = None, results_dir: Path | None = None):
        self.db_path = db_path or get_default_db_path()
        self.results_dir = results_dir or get_default_results_dir()
        self._init_db()

    def _init_db(self) -> None:
        """Create the database tables if they do not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            
            # 1. Eval Runs Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    run_id TEXT PRIMARY KEY,
                    prompt_version TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    total_cases INTEGER NOT NULL,
                    passed_cases INTEGER NOT NULL,
                    failed_cases INTEGER NOT NULL,
                    error_cases INTEGER NOT NULL,
                    overall_accuracy REAL NOT NULL,
                    avg_summary_score REAL NOT NULL,
                    avg_latency_ms REAL NOT NULL,
                    total_tokens_used INTEGER NOT NULL,
                    total_cost_estimate_usd REAL NOT NULL
                )
                """
            )
            
            # 2. Case Results Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS case_results (
                    run_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    input_email TEXT NOT NULL,
                    expected_category TEXT NOT NULL,
                    predicted_category TEXT,
                    category_match INTEGER NOT NULL,
                    expected_summary TEXT NOT NULL,
                    predicted_summary TEXT NOT NULL,
                    summary_relevance_score REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    raw_response TEXT NOT NULL,
                    error TEXT,
                    confidence REAL NOT NULL,
                    PRIMARY KEY (run_id, case_id),
                    FOREIGN KEY (run_id) REFERENCES eval_runs (run_id) ON DELETE CASCADE
                )
                """
            )
            conn.commit()

    def save_run(self, result: EvalRunResult) -> None:
        """Save a complete evaluation run to both SQLite and a JSON file."""
        # 1. Save to SQLite
        m = result.metadata
        started_str = m.started_at.isoformat() if isinstance(m.started_at, datetime) else str(m.started_at)
        finished_str = (
            m.finished_at.isoformat() 
            if m.finished_at and isinstance(m.finished_at, datetime) 
            else (str(m.finished_at) if m.finished_at else None)
        )
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO eval_runs (
                    run_id, prompt_version, model, dataset_version, started_at, finished_at,
                    total_cases, passed_cases, failed_cases, error_cases, overall_accuracy,
                    avg_summary_score, avg_latency_ms, total_tokens_used, total_cost_estimate_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    m.run_id,
                    m.prompt_version,
                    m.model,
                    m.dataset_version,
                    started_str,
                    finished_str,
                    m.total_cases,
                    m.passed_cases,
                    m.failed_cases,
                    m.error_cases,
                    result.overall_accuracy,
                    result.avg_summary_score,
                    result.avg_latency_ms,
                    result.total_tokens_used,
                    result.total_cost_estimate_usd,
                ),
            )
            
            # Insert case results
            for cr in result.case_results:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO case_results (
                        run_id, case_id, input_email, expected_category, predicted_category,
                        category_match, expected_summary, predicted_summary,
                        summary_relevance_score, latency_ms, prompt_tokens,
                        completion_tokens, total_tokens, raw_response, error, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        m.run_id,
                        cr.case_id,
                        cr.input_email,
                        cr.expected_category.value if isinstance(cr.expected_category, EmailCategory) else str(cr.expected_category),
                        cr.predicted_category.value if cr.predicted_category and isinstance(cr.predicted_category, EmailCategory) else (str(cr.predicted_category) if cr.predicted_category else None),
                        1 if cr.category_match else 0,
                        cr.expected_summary,
                        cr.predicted_summary,
                        cr.summary_relevance_score,
                        cr.latency_ms,
                        cr.prompt_tokens,
                        cr.completion_tokens,
                        cr.total_tokens,
                        cr.raw_response,
                        cr.error,
                        cr.confidence,
                    ),
                )
            conn.commit()

        # 2. Save to JSON File
        self.results_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.results_dir / f"run_{m.run_id}.json"
        
        # We can dump using Pydantic's model_dump or model_dump_json
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

    def load_run(self, run_id: str) -> EvalRunResult | None:
        """Load an evaluation run by its ID. Prefers loading from JSON for full detail."""
        json_path = self.results_dir / f"run_{run_id}.json"
        
        # Try loading from JSON first to get a completely hydrated Pydantic model
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                return EvalRunResult(**raw)
            except Exception:
                # Fallback to SQLite if JSON is corrupt
                pass
        
        # SQLite Fallback
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Load metadata
            cursor.execute("SELECT * FROM eval_runs WHERE run_id = ?", (run_id,))
            run_row = cursor.fetchone()
            if not run_row:
                return None
            
            metadata = RunMetadata(
                run_id=run_row["run_id"],
                prompt_version=run_row["prompt_version"],
                model=run_row["model"],
                dataset_version=run_row["dataset_version"],
                started_at=datetime.fromisoformat(run_row["started_at"]),
                finished_at=datetime.fromisoformat(run_row["finished_at"]) if run_row["finished_at"] else None,
                total_cases=run_row["total_cases"],
                passed_cases=run_row["passed_cases"],
                failed_cases=run_row["failed_cases"],
                error_cases=run_row["error_cases"],
            )
            
            # Load case results
            cursor.execute("SELECT * FROM case_results WHERE run_id = ?", (run_id,))
            case_rows = cursor.fetchall()
            
            case_results = []
            for cr in case_rows:
                case_results.append(
                    CaseResult(
                        case_id=cr["case_id"],
                        input_email=cr["input_email"],
                        expected_category=EmailCategory(cr["expected_category"]),
                        predicted_category=EmailCategory(cr["predicted_category"]) if cr["predicted_category"] else None,
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
            
            # Reconstruct per-category accuracies
            categories = list(EmailCategory)
            per_cat_acc = []
            for cat in categories:
                cat_cases = [c for c in case_results if c.expected_category == cat]
                if cat_cases:
                    correct = sum(1 for c in cat_cases if c.passed)
                    per_cat_acc.append(
                        CategoryAccuracy(
                            category=cat,
                            total=len(cat_cases),
                            correct=correct,
                            accuracy=correct / len(cat_cases),
                        )
                    )
            
            return EvalRunResult(
                metadata=metadata,
                case_results=case_results,
                overall_accuracy=run_row["overall_accuracy"],
                per_category_accuracy=per_cat_acc,
                avg_summary_score=run_row["avg_summary_score"],
                avg_latency_ms=run_row["avg_latency_ms"],
                total_tokens_used=run_row["total_tokens_used"],
                total_cost_estimate_usd=run_row["total_cost_estimate_usd"],
            )

    def get_latest_run(
        self, prompt_version: str | None = None, dataset_version: str | None = None
    ) -> EvalRunResult | None:
        """Find the latest run, optionally filtering by prompt or dataset version."""
        query = "SELECT run_id FROM eval_runs"
        params = []
        conditions = []
        
        if prompt_version:
            conditions.append("prompt_version = ?")
            params.append(prompt_version)
        if dataset_version:
            conditions.append("dataset_version = ?")
            params.append(dataset_version)
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY started_at DESC LIMIT 1"
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            if not row:
                return None
            return self.load_run(row[0])

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """List historical runs, ordered from newest to oldest."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT run_id, prompt_version, model, dataset_version, started_at, finished_at,
                       total_cases, passed_cases, failed_cases, error_cases, overall_accuracy,
                       avg_summary_score, avg_latency_ms, total_tokens_used, total_cost_estimate_usd
                FROM eval_runs
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def delete_run(self, run_id: str) -> bool:
        """Delete a run from both SQLite and JSON file."""
        # SQLite
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM eval_runs WHERE run_id = ?", (run_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
            
        # JSON file
        json_path = self.results_dir / f"run_{run_id}.json"
        if json_path.exists():
            try:
                json_path.unlink()
                deleted = True
            except Exception:
                pass
                
        return deleted
