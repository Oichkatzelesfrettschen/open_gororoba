CREATE TABLE IF NOT EXISTS insight_evidence (
 insight_id TEXT PRIMARY KEY REFERENCES insights(id),
 spec_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS insight_admissions (
 id INTEGER PRIMARY KEY,
 admission_id TEXT NOT NULL UNIQUE,
 insight_id TEXT NOT NULL REFERENCES insights(id),
 spec_json TEXT NOT NULL,
 before_json TEXT NOT NULL,
 after_json TEXT NOT NULL,
 before_compat TEXT NOT NULL,
 after_compat TEXT NOT NULL,
 previous_evidence_json TEXT,
 actor TEXT NOT NULL,
 reason TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TRIGGER IF NOT EXISTS insight_admissions_no_update BEFORE UPDATE ON insight_admissions
BEGIN SELECT RAISE(ABORT,'insight admissions are append-only'); END;
CREATE TRIGGER IF NOT EXISTS insight_admissions_no_delete BEFORE DELETE ON insight_admissions
BEGIN SELECT RAISE(ABORT,'insight admissions are append-only'); END;

-- External-content FTS deletion requires the exact indexed old values.
DROP TRIGGER IF EXISTS insights_fts_au;
DROP TRIGGER IF EXISTS insights_fts_ad;
CREATE TRIGGER insights_fts_au AFTER UPDATE ON insights BEGIN
 INSERT INTO insights_fts(insights_fts,rowid,id,title,status)
 VALUES ('delete',old.rowid,old.id,old.title,old.status);
 INSERT INTO insights_fts(rowid,id,title,status)
 VALUES (new.rowid,new.id,new.title,new.status);
END;
CREATE TRIGGER insights_fts_ad AFTER DELETE ON insights BEGIN
 INSERT INTO insights_fts(insights_fts,rowid,id,title,status)
 VALUES ('delete',old.rowid,old.id,old.title,old.status);
END;
