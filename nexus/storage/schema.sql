-- NEXUS-0 SQLite Persistence Schema (Phase 1 Learner State, Beliefs, and Provenance)

CREATE TABLE IF NOT EXISTS learner_runs (
    run_id TEXT PRIMARY KEY,
    learner_name TEXT NOT NULL,
    seed INTEGER NOT NULL,
    beam_width INTEGER NOT NULL,
    alpha REAL NOT NULL,
    epsilon REAL NOT NULL,
    max_slash_depth INTEGER NOT NULL,
    max_candidates_per_word INTEGER NOT NULL,
    given_lexicon_json TEXT,
    given_syntax_json TEXT
);

CREATE TABLE IF NOT EXISTS word_encounter_order (
    run_id TEXT NOT NULL,
    word TEXT NOT NULL,
    order_idx INTEGER NOT NULL,
    PRIMARY KEY (run_id, word),
    FOREIGN KEY (run_id) REFERENCES learner_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS experience_log (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_json TEXT NOT NULL,
    PRIMARY KEY (run_id, seq),
    FOREIGN KEY (run_id) REFERENCES learner_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS hypotheses (
    run_id TEXT NOT NULL,
    rank_idx INTEGER NOT NULL,
    posterior_weight REAL NOT NULL,
    prior_score REAL NOT NULL,
    likelihood_score REAL NOT NULL,
    total_score REAL NOT NULL,
    signature_json TEXT NOT NULL,
    PRIMARY KEY (run_id, rank_idx),
    FOREIGN KEY (run_id) REFERENCES learner_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS lexical_entries (
    run_id TEXT NOT NULL,
    rank_idx INTEGER NOT NULL,
    word TEXT NOT NULL,
    category TEXT NOT NULL,
    predicate TEXT NOT NULL,
    depth INTEGER NOT NULL,
    atom_count INTEGER NOT NULL,
    supporting_examples_json TEXT NOT NULL,
    PRIMARY KEY (run_id, rank_idx, word),
    FOREIGN KEY (run_id, rank_idx) REFERENCES hypotheses(run_id, rank_idx) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS change_records (
    run_id TEXT NOT NULL,
    example_id TEXT NOT NULL,
    event_kind TEXT NOT NULL,
    promoted_json TEXT NOT NULL,
    demoted_json TEXT NOT NULL,
    created_json TEXT NOT NULL,
    PRIMARY KEY (run_id, example_id),
    FOREIGN KEY (run_id) REFERENCES learner_runs(run_id) ON DELETE CASCADE
);
