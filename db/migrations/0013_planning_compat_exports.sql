ALTER TABLE roadmap_items
    ADD COLUMN claims_json TEXT NOT NULL DEFAULT '[]';

ALTER TABLE roadmap_items
    ADD COLUMN insight TEXT NOT NULL DEFAULT '';

UPDATE source_of_truth_manifest
SET migration_status = 'migrated'
WHERE table_name IN ('roadmap_items', 'todo_items', 'next_action_items');
