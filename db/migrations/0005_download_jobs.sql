CREATE TABLE download_jobs (
    id INTEGER PRIMARY KEY,
    requested_url TEXT NOT NULL,
    transfer_kind TEXT NOT NULL,
    requested_backend TEXT NOT NULL,
    route_scheme TEXT NOT NULL,
    route_host TEXT,
    route_backends_json TEXT NOT NULL,
    note TEXT,
    status TEXT NOT NULL,
    final_url TEXT,
    output_path TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE download_attempts (
    id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL,
    backend TEXT NOT NULL,
    http_code INTEGER,
    content_type TEXT,
    bytes INTEGER NOT NULL,
    sha256 TEXT,
    is_pdf INTEGER NOT NULL,
    final_url TEXT,
    note TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES download_jobs(id) ON DELETE CASCADE
);
CREATE INDEX idx_download_attempts_job_id ON download_attempts(job_id);
