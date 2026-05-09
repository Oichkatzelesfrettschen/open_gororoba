-- 0016_status_note_columns: add status_note column to insights and
-- experiments_cp so the gororoba-db Insight/Experiment mutators can do
-- the same column-level UPDATE the Claim mutator does.
--
-- Why: claims has a dedicated status_note TEXT column (migration 0002).
-- insights and experiments_cp do not -- their status_note string lives
-- inside the compat_toml_text blob (added in migration 0004). To unify
-- the mutator surface, both tables get the same first-class column.
--
-- The compat_toml_text blob continues to be the source of truth for
-- the eventual TOML re-export, but the new status_note column is the
-- canonical write target for direct edits via the CLI mutator.
--
-- ALTER TABLE in SQLite is an atomic, instant operation when adding a
-- nullable column; no data migration is needed.

ALTER TABLE insights ADD COLUMN status_note TEXT;

ALTER TABLE experiments_cp ADD COLUMN status_note TEXT;
