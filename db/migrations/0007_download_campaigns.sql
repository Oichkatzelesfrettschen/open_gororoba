CREATE TABLE download_campaigns (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    command_kind TEXT NOT NULL,
    input_path TEXT NOT NULL,
    out_ledger_path TEXT,
    dest_dir TEXT,
    note TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE download_campaign_jobs (
    campaign_id INTEGER NOT NULL,
    job_id INTEGER NOT NULL,
    PRIMARY KEY(campaign_id, job_id),
    FOREIGN KEY(campaign_id) REFERENCES download_campaigns(id) ON DELETE CASCADE,
    FOREIGN KEY(job_id) REFERENCES download_jobs(id) ON DELETE CASCADE
);
CREATE INDEX idx_download_campaign_jobs_job_id ON download_campaign_jobs(job_id);
