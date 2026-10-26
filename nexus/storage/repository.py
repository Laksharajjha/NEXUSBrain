"""Repository interface and SQLite implementation for persisting NEXUSLearner state.

Stores and restores:
- Hyperparameters and seed configuration
- Surface word encounter order
- Complete JSONL experience log (observations and corrections)
- Beam of joint hypotheses with posterior weights, prior/likelihood scores
- Lexical entries per hypothesis with supporting example IDs (provenance)
- Beam change records (``what_changed`` provenance)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Protocol, runtime_checkable

from nexus.learner.categorial import LexicalEntry
from nexus.learner.explain import ChangeRecord
from nexus.learner.nexus_learner import Hypothesis, NEXUSLearner, _ObsEntry
from nexus.world.scene import Scene

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


@runtime_checkable
class LearnerRepository(Protocol):
    """Backend-agnostic repository interface for saving and loading learners."""

    def save_learner(self, run_id: str, learner: NEXUSLearner) -> None:
        """Persist ``learner`` under ``run_id``."""

    def load_learner(self, run_id: str) -> NEXUSLearner:
        """Restore a :class:`NEXUSLearner` previously saved under ``run_id``."""

    def list_runs(self) -> list[str]:
        """List all stored ``run_id`` keys."""


class SQLiteLearnerRepository:
    """SQLite-backed implementation of :class:`LearnerRepository` using ``sqlite3``."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self) -> None:
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(schema_sql)

    def save_learner(self, run_id: str, learner: NEXUSLearner) -> None:
        """Save full learner state, beam hypotheses, lexical provenance, and JSONL log."""
        learner._ensure_beam()
        with self._connect() as conn:
            conn.execute("DELETE FROM learner_runs WHERE run_id = ?", (run_id,))
            conn.execute(
                """
                INSERT INTO learner_runs (
                    run_id, learner_name, seed, beam_width, alpha, epsilon,
                    max_slash_depth, max_candidates_per_word, given_lexicon_json, given_syntax_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    learner.name,
                    learner.seed,
                    learner.beam_width,
                    learner.alpha,
                    learner.epsilon,
                    learner.max_slash_depth,
                    learner.max_candidates_per_word,
                    None if learner.given_lexicon is None else json.dumps(learner.given_lexicon, sort_keys=True),
                    None if learner.given_syntax is None else json.dumps(learner.given_syntax, sort_keys=True),
                ),
            )

            conn.executemany(
                "INSERT INTO word_encounter_order (run_id, word, order_idx) VALUES (?, ?, ?)",
                [(run_id, w, idx) for w, idx in learner._word_order.items()],
            )

            conn.executemany(
                "INSERT INTO experience_log (run_id, seq, event_json) VALUES (?, ?, ?)",
                [(run_id, seq, line) for seq, line in enumerate(learner.experience_log_jsonl)],
            )

            for rank_idx, (hyp, weight) in enumerate(zip(learner._beam, learner._beam_weights)):
                conn.execute(
                    """
                    INSERT INTO hypotheses (
                        run_id, rank_idx, posterior_weight, prior_score,
                        likelihood_score, total_score, signature_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        rank_idx,
                        float(weight),
                        float(hyp.prior_score),
                        float(hyp.likelihood_score),
                        float(hyp.total_score),
                        json.dumps([list(item) for item in hyp.signature]),
                    ),
                )
                conn.executemany(
                    """
                    INSERT INTO lexical_entries (
                        run_id, rank_idx, word, category, predicate,
                        depth, atom_count, supporting_examples_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id,
                            rank_idx,
                            e.word,
                            e.category,
                            e.predicate,
                            e.depth,
                            e.atom_count,
                            json.dumps(list(e.supporting_examples)),
                        )
                        for e in hyp.entries.values()
                    ],
                )

            # Ensure change records exist for all corrections in experience log
            for line in learner.experience_log_jsonl:
                ev = json.loads(line)
                if ev.get("event") == "correct" and "target_example_id" in ev:
                    learner.what_changed(ev["target_example_id"])

            conn.executemany(
                """
                INSERT INTO change_records (
                    run_id, example_id, event_kind, promoted_json, demoted_json, created_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        rec.example_id,
                        rec.event_kind,
                        json.dumps(list(rec.promoted)),
                        json.dumps(list(rec.demoted)),
                        json.dumps(list(rec.created)),
                    )
                    for rec in learner._change_records.values()
                ],
            )

    def load_learner(self, run_id: str) -> NEXUSLearner:
        """Reconstruct a saved :class:`NEXUSLearner` with identical beam and provenance."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT seed, beam_width, alpha, epsilon, max_slash_depth,
                       max_candidates_per_word, given_lexicon_json, given_syntax_json
                FROM learner_runs WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown learner run_id: {run_id!r}")

            (
                seed,
                beam_width,
                alpha,
                epsilon,
                max_slash_depth,
                max_candidates_per_word,
                given_lex_json,
                given_syn_json,
            ) = row

            learner = NEXUSLearner(
                seed=int(seed),
                beam_width=int(beam_width),
                alpha=float(alpha),
                epsilon=float(epsilon),
                max_slash_depth=int(max_slash_depth),
                given_lexicon=None if given_lex_json is None else json.loads(given_lex_json),
                given_syntax=None if given_syn_json is None else json.loads(given_syn_json),
                max_candidates_per_word=int(max_candidates_per_word),
            )

            word_rows = conn.execute(
                "SELECT word, order_idx FROM word_encounter_order WHERE run_id = ? ORDER BY order_idx ASC",
                (run_id,),
            ).fetchall()
            learner._word_order = {str(w): int(idx) for w, idx in word_rows}

            log_rows = conn.execute(
                "SELECT event_json FROM experience_log WHERE run_id = ? ORDER BY seq ASC",
                (run_id,),
            ).fetchall()
            learner.experience_log_jsonl = [str(r[0]) for r in log_rows]

            # Reconstruct _observations and _obs_by_id from the persisted JSONL log
            for line in learner.experience_log_jsonl:
                ev = json.loads(line)
                if ev["event"] == "observe":
                    obs_entry = _ObsEntry(
                        example_id=str(ev["example_id"]),
                        scene=Scene.from_json(ev["scene"]),
                        tokens=tuple(ev["tokens"]),
                        label=bool(ev["label"]),
                        source=str(ev["source"]),
                        corrected=False,
                    )
                    idx = len(learner._observations)
                    learner._observations.append(obs_entry)
                    learner._obs_by_id[obs_entry.example_id] = idx
                elif ev["event"] == "correct":
                    t_id = str(ev["target_example_id"])
                    if t_id in learner._obs_by_id:
                        idx = learner._obs_by_id[t_id]
                        learner._observations[idx].label = bool(ev["corrected_label"])
                        learner._observations[idx].corrected = True

            hyp_rows = conn.execute(
                """
                SELECT rank_idx, posterior_weight, prior_score, likelihood_score, total_score, signature_json
                FROM hypotheses WHERE run_id = ? ORDER BY rank_idx ASC
                """,
                (run_id,),
            ).fetchall()

            beam: list[Hypothesis] = []
            weights: list[float] = []
            for rank_idx, weight, prior_s, ll_s, total_s, sig_json in hyp_rows:
                entry_rows = conn.execute(
                    """
                    SELECT word, category, predicate, supporting_examples_json
                    FROM lexical_entries WHERE run_id = ? AND rank_idx = ?
                    """,
                    (run_id, int(rank_idx)),
                ).fetchall()
                # Order entries by encounter order
                entry_rows_sorted = sorted(
                    entry_rows, key=lambda r: learner._word_order.get(str(r[0]), 0)
                )
                entries_map: dict[str, LexicalEntry] = {}
                for w, cat, pred, supp_json in entry_rows_sorted:
                    entries_map[str(w)] = LexicalEntry(
                        word=str(w),
                        category=str(cat),
                        predicate=str(pred),
                        supporting_examples=tuple(json.loads(supp_json)),
                    )
                sig_raw = json.loads(sig_json)
                sig = tuple((int(item[0]), str(item[1]), str(item[2])) for item in sig_raw)
                beam.append(
                    Hypothesis(
                        entries=entries_map,
                        prior_score=float(prior_s),
                        likelihood_score=float(ll_s),
                        total_score=float(total_s),
                        signature=sig,
                    )
                )
                weights.append(float(weight))

            learner._beam = beam
            learner._beam_weights = weights
            learner._beam_dirty = False

            crec_rows = conn.execute(
                """
                SELECT example_id, event_kind, promoted_json, demoted_json, created_json
                FROM change_records WHERE run_id = ?
                """,
                (run_id,),
            ).fetchall()
            for ex_id, ev_kind, prom_j, dem_j, cr_j in crec_rows:
                learner._change_records[str(ex_id)] = ChangeRecord(
                    example_id=str(ex_id),
                    event_kind=str(ev_kind),
                    promoted=tuple(json.loads(prom_j)),
                    demoted=tuple(json.loads(dem_j)),
                    created=tuple(json.loads(cr_j)),
                )

            return learner

    def list_runs(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT run_id FROM learner_runs ORDER BY run_id ASC").fetchall()
            return [str(r[0]) for r in rows]
