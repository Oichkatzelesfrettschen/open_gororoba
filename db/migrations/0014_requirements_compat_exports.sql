CREATE TABLE requirements_registry_meta (
    kind TEXT PRIMARY KEY,
    authoritative INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'active',
    status_token TEXT NOT NULL DEFAULT 'ACTIVE',
    updated TEXT NOT NULL DEFAULT '',
    python_recommended TEXT NOT NULL DEFAULT '',
    python_allowed TEXT NOT NULL DEFAULT '',
    primary_markdown TEXT NOT NULL DEFAULT '',
    status_allowlist_json TEXT NOT NULL DEFAULT '[]',
    runtime_stack_allowlist_json TEXT NOT NULL DEFAULT '[]',
    required_module_fields_json TEXT NOT NULL DEFAULT '[]',
    required_gap_fields_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE requirements_modules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    markdown TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    status_token TEXT NOT NULL DEFAULT 'ACTIVE',
    runtime_stack TEXT NOT NULL DEFAULT 'mixed',
    requires_modules_json TEXT NOT NULL DEFAULT '[]',
    install_targets_json TEXT NOT NULL DEFAULT '[]',
    verify_targets_json TEXT NOT NULL DEFAULT '[]',
    acceptance_criteria_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE requirements_coverage_gaps (
    id TEXT PRIMARY KEY,
    area TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    status_token TEXT NOT NULL DEFAULT 'OPEN',
    description TEXT NOT NULL DEFAULT '',
    proposed_resolution TEXT NOT NULL DEFAULT '',
    related_module_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT INTO source_of_truth_manifest (
    table_name,
    category,
    authoritative,
    legacy_toml_path,
    description,
    migration_status
) VALUES
    (
        'requirements_modules',
        'planning',
        1,
        'registry/requirements.toml',
        'Requirements module registry',
        'migrated'
    ),
    (
        'requirements_coverage_gaps',
        'planning',
        1,
        'registry/requirements.toml',
        'Requirements coverage gaps',
        'migrated'
    );
