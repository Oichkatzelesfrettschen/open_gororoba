//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: db/schema.sql -->
//! <!-- Generated from: db/migrations/*.sql via cargo run -p xtask -- db-docs -->
//!
//! # Database Catalog
//!
//! Generated file. Do not edit.
//!
//! - Source of truth: `db/schema.sql`
//! - Canonical migrations: `db/migrations/*.sql`
//! - Regenerate with: `cargo run -p xtask -- db-docs`
//! - Objects: `30`
//!
//! ## `artifact_links` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `3`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `artifact_id` | `TEXT` | `true` | `` | `1` | `0` |
//! | 1 | `url` | `TEXT` | `true` | `` | `2` | `0` |
//! | 2 | `relation` | `TEXT` | `true` | `` | `3` | `0` |
//!
//! Foreign keys:
//!
//! | id | seq | table | from | to | on update | on delete | match |
//! | --- | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | 0 | `links` | `url` | `url` | `NO ACTION` | `CASCADE` | `NONE` |
//! | 1 | 0 | `artifacts` | `artifact_id` | `id` | `NO ACTION` | `CASCADE` | `NONE` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_artifact_links_1` | `true` | `pk` | `false` | `artifact_id, url, relation, <expr>` |
//!
//! ## `artifact_paths` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `3`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `artifact_id` | `TEXT` | `true` | `` | `1` | `0` |
//! | 1 | `path` | `TEXT` | `true` | `` | `2` | `0` |
//! | 2 | `relation` | `TEXT` | `true` | `` | `3` | `0` |
//!
//! Foreign keys:
//!
//! | id | seq | table | from | to | on update | on delete | match |
//! | --- | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | 0 | `artifacts` | `artifact_id` | `id` | `NO ACTION` | `CASCADE` | `NONE` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_artifact_paths_1` | `true` | `pk` | `false` | `artifact_id, path, relation, <expr>` |
//!
//! ## `artifacts` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `8`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `id` | `TEXT` | `false` | `` | `1` | `0` |
//! | 1 | `key` | `TEXT` | `true` | `` | `0` | `0` |
//! | 2 | `title` | `TEXT` | `true` | `` | `0` | `0` |
//! | 3 | `citation` | `TEXT` | `true` | `` | `0` | `0` |
//! | 4 | `status` | `TEXT` | `true` | `` | `0` | `0` |
//! | 5 | `minimum_requirement_met` | `INTEGER` | `true` | `` | `0` | `0` |
//! | 6 | `canonical_functional_url` | `TEXT` | `false` | `` | `0` | `0` |
//! | 7 | `canonical_download_path` | `TEXT` | `false` | `` | `0` | `0` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_artifacts_2` | `true` | `u` | `false` | `key, <expr>` |
//! | 1 | `sqlite_autoindex_artifacts_1` | `true` | `pk` | `false` | `id, <expr>` |
//!
//! ## `binaries_cp` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `5`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `name` | `TEXT` | `false` | `` | `1` | `0` |
//! | 1 | `description` | `TEXT` | `true` | `` | `0` | `0` |
//! | 2 | `experiment_id` | `TEXT` | `false` | `` | `0` | `0` |
//! | 3 | `crate_name` | `TEXT` | `true` | `''` | `0` | `0` |
//! | 4 | `source` | `TEXT` | `true` | `'registry'` | `0` | `0` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_binaries_cp_1` | `true` | `pk` | `false` | `name, <expr>` |
//!
//! ## `citations` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `5`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `id` | `INTEGER` | `false` | `` | `1` | `0` |
//! | 1 | `artifact_id` | `TEXT` | `false` | `` | `0` | `0` |
//! | 2 | `citation_text` | `TEXT` | `true` | `` | `0` | `0` |
//! | 3 | `doi` | `TEXT` | `false` | `` | `0` | `0` |
//! | 4 | `canonical_url` | `TEXT` | `false` | `` | `0` | `0` |
//!
//! Foreign keys:
//!
//! | id | seq | table | from | to | on update | on delete | match |
//! | --- | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | 0 | `artifacts` | `artifact_id` | `id` | `NO ACTION` | `CASCADE` | `NONE` |
//!
//! ## `claims` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `8`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `id` | `TEXT` | `false` | `` | `1` | `0` |
//! | 1 | `statement` | `TEXT` | `true` | `` | `0` | `0` |
//! | 2 | `status` | `TEXT` | `true` | `` | `0` | `0` |
//! | 3 | `where_stated` | `TEXT` | `true` | `` | `0` | `0` |
//! | 4 | `last_verified` | `TEXT` | `true` | `` | `0` | `0` |
//! | 5 | `formal_proof` | `TEXT` | `false` | `` | `0` | `0` |
//! | 6 | `status_note` | `TEXT` | `false` | `` | `0` | `0` |
//! | 7 | `compat_toml_text` | `TEXT` | `true` | `''` | `0` | `0` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_claims_1` | `true` | `pk` | `false` | `id, <expr>` |
//!
//! ## `control_plane_meta` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `2`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `kind` | `TEXT` | `false` | `` | `1` | `0` |
//! | 1 | `compat_toml_text` | `TEXT` | `true` | `` | `0` | `0` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_control_plane_meta_1` | `true` | `pk` | `false` | `kind, <expr>` |
//!
//! ## `control_plane_runs` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `4`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `id` | `INTEGER` | `false` | `` | `1` | `0` |
//! | 1 | `action` | `TEXT` | `true` | `` | `0` | `0` |
//! | 2 | `created_at` | `TEXT` | `true` | `` | `0` | `0` |
//! | 3 | `details_json` | `TEXT` | `true` | `` | `0` | `0` |
//!
//! ## `documents` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `11`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `id` | `TEXT` | `false` | `` | `1` | `0` |
//! | 1 | `path` | `TEXT` | `true` | `` | `0` | `0` |
//! | 2 | `title` | `TEXT` | `true` | `` | `0` | `0` |
//! | 3 | `kind` | `TEXT` | `true` | `` | `0` | `0` |
//! | 4 | `authoring_mode` | `TEXT` | `true` | `` | `0` | `0` |
//! | 5 | `generated` | `INTEGER` | `true` | `` | `0` | `0` |
//! | 6 | `status` | `TEXT` | `true` | `` | `0` | `0` |
//! | 7 | `toml_backing` | `TEXT` | `false` | `` | `0` | `0` |
//! | 8 | `sha256` | `TEXT` | `false` | `` | `0` | `0` |
//! | 9 | `size_bytes` | `INTEGER` | `false` | `` | `0` | `0` |
//! | 10 | `line_count` | `INTEGER` | `false` | `` | `0` | `0` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_documents_2` | `true` | `u` | `false` | `path, <expr>` |
//! | 1 | `sqlite_autoindex_documents_1` | `true` | `pk` | `false` | `id, <expr>` |
//!
//! ## `download_attempts` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `14`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `id` | `INTEGER` | `false` | `` | `1` | `0` |
//! | 1 | `job_id` | `INTEGER` | `true` | `` | `0` | `0` |
//! | 2 | `backend` | `TEXT` | `true` | `` | `0` | `0` |
//! | 3 | `http_code` | `INTEGER` | `false` | `` | `0` | `0` |
//! | 4 | `content_type` | `TEXT` | `false` | `` | `0` | `0` |
//! | 5 | `bytes` | `INTEGER` | `true` | `` | `0` | `0` |
//! | 6 | `sha256` | `TEXT` | `false` | `` | `0` | `0` |
//! | 7 | `is_pdf` | `INTEGER` | `true` | `` | `0` | `0` |
//! | 8 | `final_url` | `TEXT` | `false` | `` | `0` | `0` |
//! | 9 | `note` | `TEXT` | `true` | `` | `0` | `0` |
//! | 10 | `recorded_at` | `TEXT` | `true` | `` | `0` | `0` |
//! | 11 | `succeeded` | `INTEGER` | `true` | `1` | `0` | `0` |
//! | 12 | `error_message` | `TEXT` | `false` | `` | `0` | `0` |
//! | 13 | `failure_class` | `TEXT` | `false` | `` | `0` | `0` |
//!
//! Foreign keys:
//!
//! | id | seq | table | from | to | on update | on delete | match |
//! | --- | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | 0 | `download_jobs` | `job_id` | `id` | `NO ACTION` | `CASCADE` | `NONE` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `idx_download_attempts_job_id` | `false` | `c` | `false` | `job_id, <expr>` |
//!
//! ## `download_campaign_jobs` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `2`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `campaign_id` | `INTEGER` | `true` | `` | `1` | `0` |
//! | 1 | `job_id` | `INTEGER` | `true` | `` | `2` | `0` |
//!
//! Foreign keys:
//!
//! | id | seq | table | from | to | on update | on delete | match |
//! | --- | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | 0 | `download_jobs` | `job_id` | `id` | `NO ACTION` | `CASCADE` | `NONE` |
//! | 1 | 0 | `download_campaigns` | `campaign_id` | `id` | `NO ACTION` | `CASCADE` | `NONE` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `idx_download_campaign_jobs_job_id` | `false` | `c` | `false` | `job_id, <expr>` |
//! | 1 | `sqlite_autoindex_download_campaign_jobs_1` | `true` | `pk` | `false` | `campaign_id, job_id, <expr>` |
//!
//! ## `download_campaigns` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `8`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `id` | `INTEGER` | `false` | `` | `1` | `0` |
//! | 1 | `name` | `TEXT` | `true` | `` | `0` | `0` |
//! | 2 | `command_kind` | `TEXT` | `true` | `` | `0` | `0` |
//! | 3 | `input_path` | `TEXT` | `true` | `` | `0` | `0` |
//! | 4 | `out_ledger_path` | `TEXT` | `false` | `` | `0` | `0` |
//! | 5 | `dest_dir` | `TEXT` | `false` | `` | `0` | `0` |
//! | 6 | `note` | `TEXT` | `false` | `` | `0` | `0` |
//! | 7 | `created_at` | `TEXT` | `true` | `` | `0` | `0` |
//!
//! ## `download_jobs` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `12`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `id` | `INTEGER` | `false` | `` | `1` | `0` |
//! | 1 | `requested_url` | `TEXT` | `true` | `` | `0` | `0` |
//! | 2 | `transfer_kind` | `TEXT` | `true` | `` | `0` | `0` |
//! | 3 | `requested_backend` | `TEXT` | `true` | `` | `0` | `0` |
//! | 4 | `route_scheme` | `TEXT` | `true` | `` | `0` | `0` |
//! | 5 | `route_host` | `TEXT` | `false` | `` | `0` | `0` |
//! | 6 | `route_backends_json` | `TEXT` | `true` | `` | `0` | `0` |
//! | 7 | `note` | `TEXT` | `false` | `` | `0` | `0` |
//! | 8 | `status` | `TEXT` | `true` | `` | `0` | `0` |
//! | 9 | `final_url` | `TEXT` | `false` | `` | `0` | `0` |
//! | 10 | `output_path` | `TEXT` | `false` | `` | `0` | `0` |
//! | 11 | `created_at` | `TEXT` | `true` | `` | `0` | `0` |
//!
//! ## `experiments_cp` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `6`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `id` | `TEXT` | `false` | `` | `1` | `0` |
//! | 1 | `title` | `TEXT` | `true` | `` | `0` | `0` |
//! | 2 | `status` | `TEXT` | `true` | `` | `0` | `0` |
//! | 3 | `binary_name` | `TEXT` | `false` | `` | `0` | `0` |
//! | 4 | `claim_refs_json` | `TEXT` | `true` | `` | `0` | `0` |
//! | 5 | `compat_toml_text` | `TEXT` | `true` | `''` | `0` | `0` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_experiments_cp_1` | `true` | `pk` | `false` | `id, <expr>` |
//!
//! ## `export_runs` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `6`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `id` | `INTEGER` | `false` | `` | `1` | `0` |
//! | 1 | `action` | `TEXT` | `true` | `` | `0` | `0` |
//! | 2 | `created_at` | `TEXT` | `true` | `` | `0` | `0` |
//! | 3 | `artifact_count` | `INTEGER` | `true` | `` | `0` | `0` |
//! | 4 | `document_count` | `INTEGER` | `true` | `` | `0` | `0` |
//! | 5 | `details_json` | `TEXT` | `true` | `` | `0` | `0` |
//!
//! ## `external_source_contract_values` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `4`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `contract_id` | `TEXT` | `true` | `` | `1` | `0` |
//! | 1 | `relation` | `TEXT` | `true` | `` | `2` | `0` |
//! | 2 | `ord` | `INTEGER` | `true` | `` | `3` | `0` |
//! | 3 | `value` | `TEXT` | `true` | `` | `0` | `0` |
//!
//! Foreign keys:
//!
//! | id | seq | table | from | to | on update | on delete | match |
//! | --- | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | 0 | `external_source_contracts` | `contract_id` | `id` | `NO ACTION` | `CASCADE` | `NONE` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_external_source_contract_values_1` | `true` | `pk` | `false` | `contract_id, relation, ord, <expr>` |
//!
//! ## `external_source_contracts` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `9`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `id` | `TEXT` | `false` | `` | `1` | `0` |
//! | 1 | `path_glob` | `TEXT` | `true` | `` | `0` | `0` |
//! | 2 | `canonical_url` | `TEXT` | `true` | `` | `0` | `0` |
//! | 3 | `access_class` | `TEXT` | `true` | `` | `0` | `0` |
//! | 4 | `status` | `TEXT` | `true` | `` | `0` | `0` |
//! | 5 | `retrieval_method` | `TEXT` | `true` | `` | `0` | `0` |
//! | 6 | `attempt_deadline_utc` | `TEXT` | `true` | `` | `0` | `0` |
//! | 7 | `resolution_deadline_utc` | `TEXT` | `true` | `` | `0` | `0` |
//! | 8 | `blocker_note` | `TEXT` | `true` | `` | `0` | `0` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_external_source_contracts_1` | `true` | `pk` | `false` | `id, <expr>` |
//!
//! ## `external_source_contracts_meta` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `4`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `kind` | `TEXT` | `false` | `` | `1` | `0` |
//! | 1 | `updated` | `TEXT` | `true` | `` | `0` | `0` |
//! | 2 | `authoritative` | `INTEGER` | `true` | `` | `0` | `0` |
//! | 3 | `policy_version` | `TEXT` | `true` | `` | `0` | `0` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_external_source_contracts_meta_1` | `true` | `pk` | `false` | `kind, <expr>` |
//!
//! ## `external_source_dossier_values` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `4`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `dossier_id` | `TEXT` | `true` | `` | `1` | `0` |
//! | 1 | `relation` | `TEXT` | `true` | `` | `2` | `0` |
//! | 2 | `ord` | `INTEGER` | `true` | `` | `3` | `0` |
//! | 3 | `value` | `TEXT` | `true` | `` | `0` | `0` |
//!
//! Foreign keys:
//!
//! | id | seq | table | from | to | on update | on delete | match |
//! | --- | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | 0 | `external_source_dossiers` | `dossier_id` | `id` | `NO ACTION` | `CASCADE` | `NONE` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_external_source_dossier_values_1` | `true` | `pk` | `false` | `dossier_id, relation, ord, <expr>` |
//!
//! ## `external_source_dossiers` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `14`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `id` | `TEXT` | `false` | `` | `1` | `0` |
//! | 1 | `source_markdown` | `TEXT` | `true` | `` | `0` | `0` |
//! | 2 | `slug` | `TEXT` | `true` | `` | `0` | `0` |
//! | 3 | `title` | `TEXT` | `true` | `` | `0` | `0` |
//! | 4 | `status_token` | `TEXT` | `true` | `` | `0` | `0` |
//! | 5 | `content_kind` | `TEXT` | `true` | `` | `0` | `0` |
//! | 6 | `authority_level` | `TEXT` | `true` | `` | `0` | `0` |
//! | 7 | `verification_level` | `TEXT` | `true` | `` | `0` | `0` |
//! | 8 | `operational_role` | `TEXT` | `true` | `` | `0` | `0` |
//! | 9 | `source_lineage_summary` | `TEXT` | `true` | `` | `0` | `0` |
//! | 10 | `has_full_transcript` | `INTEGER` | `true` | `` | `0` | `0` |
//! | 11 | `line_count` | `INTEGER` | `true` | `` | `0` | `0` |
//! | 12 | `notes` | `TEXT` | `true` | `` | `0` | `0` |
//! | 13 | `body_markdown` | `TEXT` | `true` | `` | `0` | `0` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_external_source_dossiers_1` | `true` | `pk` | `false` | `id, <expr>` |
//!
//! ## `external_source_dossiers_meta` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `5`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `kind` | `TEXT` | `false` | `` | `1` | `0` |
//! | 1 | `updated` | `TEXT` | `true` | `` | `0` | `0` |
//! | 2 | `authoritative` | `INTEGER` | `true` | `` | `0` | `0` |
//! | 3 | `source_markdown_glob` | `TEXT` | `true` | `` | `0` | `0` |
//! | 4 | `document_count` | `INTEGER` | `true` | `` | `0` | `0` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_external_source_dossiers_meta_1` | `true` | `pk` | `false` | `kind, <expr>` |
//!
//! ## `ingest_fingerprints` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `4`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `path` | `TEXT` | `false` | `` | `1` | `0` |
//! | 1 | `blake3_hex` | `TEXT` | `true` | `` | `0` | `0` |
//! | 2 | `size_bytes` | `INTEGER` | `true` | `` | `0` | `0` |
//! | 3 | `indexed_at` | `TEXT` | `true` | `` | `0` | `0` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_ingest_fingerprints_1` | `true` | `pk` | `false` | `path, <expr>` |
//!
//! ## `insights` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `5`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `id` | `TEXT` | `false` | `` | `1` | `0` |
//! | 1 | `title` | `TEXT` | `true` | `` | `0` | `0` |
//! | 2 | `status` | `TEXT` | `true` | `` | `0` | `0` |
//! | 3 | `claim_refs_json` | `TEXT` | `true` | `` | `0` | `0` |
//! | 4 | `compat_toml_text` | `TEXT` | `true` | `''` | `0` | `0` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_insights_1` | `true` | `pk` | `false` | `id, <expr>` |
//!
//! ## `lane_assignments` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `2`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `artifact_id` | `TEXT` | `true` | `` | `1` | `0` |
//! | 1 | `lane_name` | `TEXT` | `true` | `` | `2` | `0` |
//!
//! Foreign keys:
//!
//! | id | seq | table | from | to | on update | on delete | match |
//! | --- | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | 0 | `artifacts` | `artifact_id` | `id` | `NO ACTION` | `CASCADE` | `NONE` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_lane_assignments_1` | `true` | `pk` | `false` | `artifact_id, lane_name, <expr>` |
//!
//! ## `links` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `2`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `url` | `TEXT` | `false` | `` | `1` | `0` |
//! | 1 | `host` | `TEXT` | `false` | `` | `0` | `0` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_links_1` | `true` | `pk` | `false` | `url, <expr>` |
//!
//! ## `mirror_observations` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `3`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `artifact_id` | `TEXT` | `true` | `` | `1` | `0` |
//! | 1 | `url` | `TEXT` | `true` | `` | `2` | `0` |
//! | 2 | `mirror_kind` | `TEXT` | `true` | `` | `3` | `0` |
//!
//! Foreign keys:
//!
//! | id | seq | table | from | to | on update | on delete | match |
//! | --- | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | 0 | `links` | `url` | `url` | `NO ACTION` | `CASCADE` | `NONE` |
//! | 1 | 0 | `artifacts` | `artifact_id` | `id` | `NO ACTION` | `CASCADE` | `NONE` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_mirror_observations_1` | `true` | `pk` | `false` | `artifact_id, url, mirror_kind, <expr>` |
//!
//! ## `record_sources` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `3`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `entity_kind` | `TEXT` | `true` | `` | `1` | `0` |
//! | 1 | `entity_id` | `TEXT` | `true` | `` | `2` | `0` |
//! | 2 | `source_ref` | `TEXT` | `true` | `` | `3` | `0` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_record_sources_1` | `true` | `pk` | `false` | `entity_kind, entity_id, source_ref, <expr>` |
//!
//! ## `registry_snapshots` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `5`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `registry_kind` | `TEXT` | `false` | `` | `1` | `0` |
//! | 1 | `source_path` | `TEXT` | `true` | `` | `0` | `0` |
//! | 2 | `content_text` | `TEXT` | `true` | `` | `0` | `0` |
//! | 3 | `blake3_hex` | `TEXT` | `true` | `` | `0` | `0` |
//! | 4 | `indexed_at` | `TEXT` | `true` | `` | `0` | `0` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_registry_snapshots_1` | `true` | `pk` | `false` | `registry_kind, <expr>` |
//!
//! ## `theorems` (table)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `6`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `id` | `TEXT` | `false` | `` | `1` | `0` |
//! | 1 | `title` | `TEXT` | `true` | `` | `0` | `0` |
//! | 2 | `proof_path` | `TEXT` | `true` | `` | `0` | `0` |
//! | 3 | `status` | `TEXT` | `true` | `` | `0` | `0` |
//! | 4 | `linked_claim_ids_json` | `TEXT` | `true` | `` | `0` | `0` |
//! | 5 | `source` | `TEXT` | `true` | `` | `0` | `0` |
//!
//! Indexes:
//!
//! | seq | name | unique | origin | partial | columns |
//! | --- | --- | --- | --- | --- | --- |
//! | 0 | `sqlite_autoindex_theorems_2` | `true` | `u` | `false` | `proof_path, <expr>` |
//! | 1 | `sqlite_autoindex_theorems_1` | `true` | `pk` | `false` | `id, <expr>` |
//!
//! ## `document_search` (virtual)
//!
//! - Strict: `false`
//! - Without rowid: `false`
//! - Declared columns: `6`
//!
//! | cid | name | type | not null | default | pk | hidden |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 0 | `document_id` | `` | `false` | `` | `0` | `0` |
//! | 1 | `path` | `` | `false` | `` | `0` | `0` |
//! | 2 | `title` | `` | `false` | `` | `0` | `0` |
//! | 3 | `kind` | `` | `false` | `` | `0` | `0` |
//! | 4 | `document_search` | `` | `false` | `` | `0` | `1` |
//! | 5 | `rank` | `` | `false` | `` | `0` | `1` |
//!
