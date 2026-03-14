ALTER TABLE claims ADD COLUMN compat_toml_text TEXT NOT NULL DEFAULT '';
ALTER TABLE insights ADD COLUMN compat_toml_text TEXT NOT NULL DEFAULT '';
ALTER TABLE experiments_cp ADD COLUMN compat_toml_text TEXT NOT NULL DEFAULT '';
CREATE TABLE control_plane_meta (
    kind TEXT PRIMARY KEY,
    compat_toml_text TEXT NOT NULL
);
