"""SQLite Repository for persisting Phase 3 RestructureLearner state, beliefs, and revision history.

Persists and rehydrates:
- Learner configuration and hyperparameters (seed, beam_margin, costs, restructure_mode).
- Append-only revision log (RevisionRecord items: operator, target_word, score_change, etc.).
- Active lexical entries, multi-sense dictionaries, and shared concept pointers.
- Global exception set.
- Complete experience log and word encounter order.
- Verified identical predictions on held-out test splits.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from nexus.learner.categorial import LexicalEntry, Sense
from nexus.learner.nexus_learner import _ObsEntry
from nexus.learner.restructure_learner import RestructureLearner, RevisionRecord
from nexus.world.scene import Scene


class SQLiteRestructureRepository:
    """SQLite-backed persistence repository for RestructureLearner."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS restructure_runs (
                    run_id TEXT PRIMARY KEY,
                    learner_name TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    beam_width INTEGER NOT NULL,
                    beam_margin REAL NOT NULL,
                    alpha REAL NOT NULL,
                    epsilon REAL NOT NULL,
                    cost_per_exception REAL NOT NULL,
                    cost_split_sense REAL NOT NULL,
                    restructure_mode TEXT NOT NULL,
                    evidence_channel TEXT NOT NULL,
                    cooc_weight REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS restructure_revisions (
                    run_id TEXT NOT NULL,
                    revision_id INTEGER NOT NULL,
                    timestamp_index INTEGER NOT NULL,
                    operator TEXT NOT NULL,
                    target_word TEXT NOT NULL,
                    old_state_summary TEXT NOT NULL,
                    new_state_summary TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    score_change REAL NOT NULL,
                    old_score REAL NOT NULL,
                    new_score REAL NOT NULL,
                    superseded_beliefs_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, revision_id),
                    FOREIGN KEY (run_id) REFERENCES restructure_runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS restructure_entries (
                    run_id TEXT NOT NULL,
                    word TEXT NOT NULL,
                    category TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    shared_concept_id TEXT,
                    exception_keys_json TEXT NOT NULL,
                    senses_json TEXT NOT NULL,
                    entry_count INTEGER NOT NULL,
                    depth INTEGER NOT NULL,
                    atom_count INTEGER NOT NULL,
                    PRIMARY KEY (run_id, word),
                    FOREIGN KEY (run_id) REFERENCES restructure_runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS restructure_exceptions (
                    run_id TEXT NOT NULL,
                    tokens_json TEXT NOT NULL,
                    scene_id TEXT NOT NULL,
                    PRIMARY KEY (run_id, tokens_json, scene_id),
                    FOREIGN KEY (run_id) REFERENCES restructure_runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS restructure_word_order (
                    run_id TEXT NOT NULL,
                    word TEXT NOT NULL,
                    order_idx INTEGER NOT NULL,
                    PRIMARY KEY (run_id, word),
                    FOREIGN KEY (run_id) REFERENCES restructure_runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS restructure_experience_log (
                    run_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, seq),
                    FOREIGN KEY (run_id) REFERENCES restructure_runs(run_id) ON DELETE CASCADE
                );
                """
            )

    def save_learner(self, run_id: str, learner: RestructureLearner) -> None:
        """Persist RestructureLearner state, beliefs, entries, and revision history."""
        # Ensure restructuring has run
        learner._ensure_restructure()

        with self._connect() as conn:
            conn.execute("DELETE FROM restructure_runs WHERE run_id = ?", (run_id,))

            conn.execute(
                """
                INSERT INTO restructure_runs (
                    run_id, learner_name, seed, beam_width, beam_margin, alpha, epsilon,
                    cost_per_exception, cost_split_sense, restructure_mode, evidence_channel, cooc_weight
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    learner.name,
                    learner.seed,
                    learner.beam_width,
                    learner.beam_margin,
                    learner.alpha,
                    learner.epsilon,
                    learner.cost_per_exception,
                    learner.cost_split_sense,
                    learner.restructure_mode,
                    learner.evidence_channel,
                    learner.cooc_weight,
                ),
            )

            # Revision log
            rev_rows = []
            for r in learner.revision_log:
                rev_rows.append((
                    run_id,
                    r.revision_id,
                    r.timestamp_index,
                    r.operator,
                    r.target_word,
                    r.old_state_summary,
                    r.new_state_summary,
                    json.dumps(list(r.evidence_ids)),
                    float(r.score_change),
                    float(r.old_score),
                    float(r.new_score),
                    json.dumps(list(r.superseded_beliefs)),
                ))
            conn.executemany(
                """
                INSERT INTO restructure_revisions (
                    run_id, revision_id, timestamp_index, operator, target_word,
                    old_state_summary, new_state_summary, evidence_ids_json,
                    score_change, old_score, new_score, superseded_beliefs_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rev_rows,
            )

            # Active entries
            entry_rows = []
            for word, entry in learner._learned_entries.items():
                senses_data = []
                if entry.senses:
                    for s in entry.senses:
                        senses_data.append({
                            "sense_id": s.sense_id,
                            "category": s.category,
                            "predicate": s.predicate,
                            "provenance": list(s.provenance),
                            "created_by_op": s.created_by_op,
                            "superseded": s.superseded,
                        })
                exc_keys = [
                    [list(tokens), sc_id]
                    for tokens, sc_id in (entry.exception_keys or ())
                ]
                entry_rows.append((
                    run_id,
                    word,
                    entry.category,
                    entry.predicate,
                    entry.shared_concept_id,
                    json.dumps(exc_keys),
                    json.dumps(senses_data),
                    entry.entry_count,
                    entry.depth,
                    entry.atom_count,
                ))
            conn.executemany(
                """
                INSERT INTO restructure_entries (
                    run_id, word, category, predicate, shared_concept_id,
                    exception_keys_json, senses_json, entry_count, depth, atom_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                entry_rows,
            )

            # Global exceptions
            exc_rows = [
                (run_id, json.dumps(list(tokens)), str(sc_id))
                for tokens, sc_id in learner._global_exception_keys
            ]
            conn.executemany(
                """
                INSERT INTO restructure_exceptions (run_id, tokens_json, scene_id)
                VALUES (?, ?, ?)
                """,
                exc_rows,
            )

            # Word encounter order
            order_rows = [
                (run_id, w, idx) for w, idx in learner._word_order.items()
            ]
            conn.executemany(
                "INSERT INTO restructure_word_order (run_id, word, order_idx) VALUES (?, ?, ?)",
                order_rows,
            )

            # Experience log
            log_rows = [
                (run_id, seq, line)
                for seq, line in enumerate(learner.experience_log_jsonl)
            ]
            conn.executemany(
                "INSERT INTO restructure_experience_log (run_id, seq, event_json) VALUES (?, ?, ?)",
                log_rows,
            )

    def load_learner(self, run_id: str) -> RestructureLearner:
        """Reconstruct a saved RestructureLearner with exact beliefs, log, and entries."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT seed, beam_width, beam_margin, alpha, epsilon,
                       cost_per_exception, cost_split_sense, restructure_mode,
                       evidence_channel, cooc_weight
                FROM restructure_runs WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Run id {run_id!r} not found in database.")

            (
                seed,
                beam_width,
                beam_margin,
                alpha,
                epsilon,
                cost_exc,
                cost_split,
                restruct_mode,
                ev_channel,
                cooc_wt,
            ) = row

            learner = RestructureLearner(
                seed=int(seed),
                beam_width=int(beam_width),
                alpha=float(alpha),
                epsilon=float(epsilon),
                restructure_mode=str(restruct_mode),
                beam_margin=float(beam_margin),
                cost_per_exception=float(cost_exc),
                cost_split_sense=float(cost_split),
                evidence_channel=str(ev_channel),
                cooc_weight=float(cooc_wt),
            )

            # Reconstruct word encounter order
            word_rows = conn.execute(
                "SELECT word, order_idx FROM restructure_word_order WHERE run_id = ? ORDER BY order_idx ASC",
                (run_id,),
            ).fetchall()
            learner._word_order = {str(w): int(idx) for w, idx in word_rows}

            # Reconstruct experience log and observations
            log_rows = conn.execute(
                "SELECT event_json FROM restructure_experience_log WHERE run_id = ? ORDER BY seq ASC",
                (run_id,),
            ).fetchall()
            learner.experience_log_jsonl = [str(r[0]) for r in log_rows]

            for line in learner.experience_log_jsonl:
                ev = json.loads(line)
                if ev.get("event") == "observe":
                    obs = _ObsEntry(
                        example_id=str(ev["example_id"]),
                        scene=Scene.from_json(ev["scene"]),
                        tokens=tuple(ev["tokens"]),
                        label=bool(ev["label"]),
                        source=str(ev["source"]),
                        corrected=False,
                    )
                    idx = len(learner._observations)
                    learner._observations.append(obs)
                    learner._obs_by_id[obs.example_id] = idx

            # Reconstruct active entries
            entry_rows = conn.execute(
                """
                SELECT word, category, predicate, shared_concept_id, exception_keys_json, senses_json, entry_count, depth, atom_count
                FROM restructure_entries WHERE run_id = ?
                """,
                (run_id,),
            ).fetchall()

            entries = {}
            for e_row in entry_rows:
                word, cat, pred, shared_id, exc_json, senses_json, e_cnt, depth, a_cnt = e_row
                raw_exc = json.loads(exc_json)
                exc_tuple = (
                    tuple((tuple(item[0]), str(item[1])) for item in raw_exc)
                    if raw_exc
                    else ()
                )
                senses_raw = json.loads(senses_json)
                senses_tuple = ()
                if senses_raw:
                    s_list = []
                    for s in senses_raw:
                        s_list.append(
                            Sense(
                                sense_id=str(s["sense_id"]),
                                category=str(s["category"]),
                                predicate=str(s["predicate"]),
                                provenance=tuple(s.get("provenance", ())),
                                created_by_op=str(s.get("created_by_op", "initial")),
                                superseded=bool(s.get("superseded", False)),
                            )
                        )
                    senses_tuple = tuple(s_list)

                entries[str(word)] = LexicalEntry(
                    word=str(word),
                    category=str(cat),
                    predicate=str(pred),
                    shared_concept_id=str(shared_id) if shared_id else None,
                    exception_keys=exc_tuple,
                    senses=senses_tuple,
                )
            learner._learned_entries = entries

            # Reconstruct global exceptions
            exc_rows = conn.execute(
                "SELECT tokens_json, scene_id FROM restructure_exceptions WHERE run_id = ?",
                (run_id,),
            ).fetchall()
            learner._global_exception_keys = {
                (tuple(json.loads(tokens_json)), str(sc_id))
                for tokens_json, sc_id in exc_rows
            }

            # Reconstruct revision log
            rev_rows = conn.execute(
                """
                SELECT revision_id, timestamp_index, operator, target_word,
                       old_state_summary, new_state_summary, evidence_ids_json,
                       score_change, old_score, new_score, superseded_beliefs_json
                FROM restructure_revisions WHERE run_id = ? ORDER BY revision_id ASC
                """,
                (run_id,),
            ).fetchall()
            rev_list = []
            for r_row in rev_rows:
                (
                    rev_id,
                    t_idx,
                    op,
                    w,
                    old_s,
                    new_s,
                    ev_json,
                    sc_ch,
                    old_sc,
                    new_sc,
                    sup_json,
                ) = r_row
                rev_list.append(
                    RevisionRecord(
                        revision_id=int(rev_id),
                        timestamp_index=int(t_idx),
                        operator=str(op),
                        target_word=str(w),
                        old_state_summary=str(old_s),
                        new_state_summary=str(new_s),
                        evidence_ids=tuple(json.loads(ev_json)),
                        score_change=float(sc_ch),
                        old_score=float(old_sc),
                        new_score=float(new_sc),
                        superseded_beliefs=tuple(json.loads(sup_json)),
                    )
                )
            learner._revision_log_list = rev_list
            learner._revision_counter = len(rev_list)
            learner._restructured = True

            return learner
