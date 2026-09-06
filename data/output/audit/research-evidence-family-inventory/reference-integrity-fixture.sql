-- Load into SQLite :memory: before inventory-queries.sql; never into a canonical store.
CREATE TABLE claims(id TEXT, status TEXT);
CREATE TABLE insights(id TEXT, title TEXT, status TEXT, claim_refs_json TEXT);
CREATE TABLE experiments_cp(status TEXT);
CREATE TABLE lacunae(status TEXT);
CREATE TABLE claim_evidence(spec_json TEXT);
CREATE TABLE claim_insight_refs(claim_id TEXT, insight_id TEXT);
INSERT INTO claims VALUES ('C-001','Verified');
INSERT INTO insights VALUES ('I-001','declared orphan','verified','["C-001","C-missing"]');
INSERT INTO insights VALUES ('I-002','undeclared and missing normalized','verified','["C-unlinked"]');
INSERT INTO claim_insight_refs VALUES ('C-001','I-001');
INSERT INTO claim_insight_refs VALUES ('C-missing','I-001');
INSERT INTO claim_insight_refs VALUES ('C-001','I-missing');
INSERT INTO claim_insight_refs VALUES ('C-001','I-002');
