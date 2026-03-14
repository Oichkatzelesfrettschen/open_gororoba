ALTER TABLE download_attempts ADD COLUMN succeeded INTEGER NOT NULL DEFAULT 1;
ALTER TABLE download_attempts ADD COLUMN error_message TEXT;
