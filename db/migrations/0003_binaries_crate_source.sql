ALTER TABLE binaries_cp ADD COLUMN crate_name TEXT NOT NULL DEFAULT '';
ALTER TABLE binaries_cp ADD COLUMN source TEXT NOT NULL DEFAULT 'registry';
