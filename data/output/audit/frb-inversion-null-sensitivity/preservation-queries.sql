ATTACH '.cache/frb-admission/before.sqlite3' AS baseline;
SELECT 'artifact_links' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."artifact_links" EXCEPT SELECT * FROM main."artifact_links")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."artifact_links" EXCEPT SELECT * FROM baseline."artifact_links")) AS added_rows
UNION ALL
SELECT 'artifact_paths' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."artifact_paths" EXCEPT SELECT * FROM main."artifact_paths")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."artifact_paths" EXCEPT SELECT * FROM baseline."artifact_paths")) AS added_rows
UNION ALL
SELECT 'artifact_retrieval_observations' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."artifact_retrieval_observations" EXCEPT SELECT * FROM main."artifact_retrieval_observations")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."artifact_retrieval_observations" EXCEPT SELECT * FROM baseline."artifact_retrieval_observations")) AS added_rows
UNION ALL
SELECT 'artifacts' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."artifacts" EXCEPT SELECT * FROM main."artifacts")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."artifacts" EXCEPT SELECT * FROM baseline."artifacts")) AS added_rows
UNION ALL
SELECT 'bibliography' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."bibliography" EXCEPT SELECT * FROM main."bibliography")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."bibliography" EXCEPT SELECT * FROM baseline."bibliography")) AS added_rows
UNION ALL
SELECT 'bibliography_fts' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."bibliography_fts" EXCEPT SELECT * FROM main."bibliography_fts")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."bibliography_fts" EXCEPT SELECT * FROM baseline."bibliography_fts")) AS added_rows
UNION ALL
SELECT 'bibliography_fts_config' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."bibliography_fts_config" EXCEPT SELECT * FROM main."bibliography_fts_config")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."bibliography_fts_config" EXCEPT SELECT * FROM baseline."bibliography_fts_config")) AS added_rows
UNION ALL
SELECT 'bibliography_fts_data' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."bibliography_fts_data" EXCEPT SELECT * FROM main."bibliography_fts_data")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."bibliography_fts_data" EXCEPT SELECT * FROM baseline."bibliography_fts_data")) AS added_rows
UNION ALL
SELECT 'bibliography_fts_docsize' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."bibliography_fts_docsize" EXCEPT SELECT * FROM main."bibliography_fts_docsize")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."bibliography_fts_docsize" EXCEPT SELECT * FROM baseline."bibliography_fts_docsize")) AS added_rows
UNION ALL
SELECT 'bibliography_fts_idx' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."bibliography_fts_idx" EXCEPT SELECT * FROM main."bibliography_fts_idx")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."bibliography_fts_idx" EXCEPT SELECT * FROM baseline."bibliography_fts_idx")) AS added_rows
UNION ALL
SELECT 'binaries_cp' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."binaries_cp" EXCEPT SELECT * FROM main."binaries_cp")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."binaries_cp" EXCEPT SELECT * FROM baseline."binaries_cp")) AS added_rows
UNION ALL
SELECT 'build_metadata' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."build_metadata" EXCEPT SELECT * FROM main."build_metadata")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."build_metadata" EXCEPT SELECT * FROM baseline."build_metadata")) AS added_rows
UNION ALL
SELECT 'citations' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."citations" EXCEPT SELECT * FROM main."citations")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."citations" EXCEPT SELECT * FROM baseline."citations")) AS added_rows
UNION ALL
SELECT 'claim_evidence' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claim_evidence" EXCEPT SELECT * FROM main."claim_evidence")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claim_evidence" EXCEPT SELECT * FROM baseline."claim_evidence")) AS added_rows
UNION ALL
SELECT 'claim_evidence_revision_experiments' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claim_evidence_revision_experiments" EXCEPT SELECT * FROM main."claim_evidence_revision_experiments")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claim_evidence_revision_experiments" EXCEPT SELECT * FROM baseline."claim_evidence_revision_experiments")) AS added_rows
UNION ALL
SELECT 'claim_evidence_revisions' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claim_evidence_revisions" EXCEPT SELECT * FROM main."claim_evidence_revisions")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claim_evidence_revisions" EXCEPT SELECT * FROM baseline."claim_evidence_revisions")) AS added_rows
UNION ALL
SELECT 'claim_experiment_refs' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claim_experiment_refs" EXCEPT SELECT * FROM main."claim_experiment_refs")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claim_experiment_refs" EXCEPT SELECT * FROM baseline."claim_experiment_refs")) AS added_rows
UNION ALL
SELECT 'claim_relations' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claim_relations" EXCEPT SELECT * FROM main."claim_relations")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claim_relations" EXCEPT SELECT * FROM baseline."claim_relations")) AS added_rows
UNION ALL
SELECT 'claim_revisions' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claim_revisions" EXCEPT SELECT * FROM main."claim_revisions")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claim_revisions" EXCEPT SELECT * FROM baseline."claim_revisions")) AS added_rows
UNION ALL
SELECT 'claim_status_write_context' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claim_status_write_context" EXCEPT SELECT * FROM main."claim_status_write_context")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claim_status_write_context" EXCEPT SELECT * FROM baseline."claim_status_write_context")) AS added_rows
UNION ALL
SELECT 'claim_transition_assumptions' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claim_transition_assumptions" EXCEPT SELECT * FROM main."claim_transition_assumptions")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claim_transition_assumptions" EXCEPT SELECT * FROM baseline."claim_transition_assumptions")) AS added_rows
UNION ALL
SELECT 'claim_transition_events' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claim_transition_events" EXCEPT SELECT * FROM main."claim_transition_events")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claim_transition_events" EXCEPT SELECT * FROM baseline."claim_transition_events")) AS added_rows
UNION ALL
SELECT 'claim_transition_evidence' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claim_transition_evidence" EXCEPT SELECT * FROM main."claim_transition_evidence")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claim_transition_evidence" EXCEPT SELECT * FROM baseline."claim_transition_evidence")) AS added_rows
UNION ALL
SELECT 'claim_transition_experiments' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claim_transition_experiments" EXCEPT SELECT * FROM main."claim_transition_experiments")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claim_transition_experiments" EXCEPT SELECT * FROM baseline."claim_transition_experiments")) AS added_rows
UNION ALL
SELECT 'claim_transition_successor_evidence' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claim_transition_successor_evidence" EXCEPT SELECT * FROM main."claim_transition_successor_evidence")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claim_transition_successor_evidence" EXCEPT SELECT * FROM baseline."claim_transition_successor_evidence")) AS added_rows
UNION ALL
SELECT 'claim_transition_successor_where_stated' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claim_transition_successor_where_stated" EXCEPT SELECT * FROM main."claim_transition_successor_where_stated")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claim_transition_successor_where_stated" EXCEPT SELECT * FROM baseline."claim_transition_successor_where_stated")) AS added_rows
UNION ALL
SELECT 'claim_transition_successors' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claim_transition_successors" EXCEPT SELECT * FROM main."claim_transition_successors")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claim_transition_successors" EXCEPT SELECT * FROM baseline."claim_transition_successors")) AS added_rows
UNION ALL
SELECT 'claims' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claims" EXCEPT SELECT * FROM main."claims")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claims" EXCEPT SELECT * FROM baseline."claims")) AS added_rows
UNION ALL
SELECT 'claims_fts' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claims_fts" EXCEPT SELECT * FROM main."claims_fts")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claims_fts" EXCEPT SELECT * FROM baseline."claims_fts")) AS added_rows
UNION ALL
SELECT 'claims_fts_config' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claims_fts_config" EXCEPT SELECT * FROM main."claims_fts_config")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claims_fts_config" EXCEPT SELECT * FROM baseline."claims_fts_config")) AS added_rows
UNION ALL
SELECT 'claims_fts_data' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claims_fts_data" EXCEPT SELECT * FROM main."claims_fts_data")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claims_fts_data" EXCEPT SELECT * FROM baseline."claims_fts_data")) AS added_rows
UNION ALL
SELECT 'claims_fts_docsize' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claims_fts_docsize" EXCEPT SELECT * FROM main."claims_fts_docsize")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claims_fts_docsize" EXCEPT SELECT * FROM baseline."claims_fts_docsize")) AS added_rows
UNION ALL
SELECT 'claims_fts_idx' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."claims_fts_idx" EXCEPT SELECT * FROM main."claims_fts_idx")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."claims_fts_idx" EXCEPT SELECT * FROM baseline."claims_fts_idx")) AS added_rows
UNION ALL
SELECT 'control_plane_meta' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."control_plane_meta" EXCEPT SELECT * FROM main."control_plane_meta")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."control_plane_meta" EXCEPT SELECT * FROM baseline."control_plane_meta")) AS added_rows
UNION ALL
SELECT 'control_plane_runs' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."control_plane_runs" EXCEPT SELECT * FROM main."control_plane_runs")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."control_plane_runs" EXCEPT SELECT * FROM baseline."control_plane_runs")) AS added_rows
UNION ALL
SELECT 'derivation_steps' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."derivation_steps" EXCEPT SELECT * FROM main."derivation_steps")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."derivation_steps" EXCEPT SELECT * FROM baseline."derivation_steps")) AS added_rows
UNION ALL
SELECT 'document_search' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."document_search" EXCEPT SELECT * FROM main."document_search")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."document_search" EXCEPT SELECT * FROM baseline."document_search")) AS added_rows
UNION ALL
SELECT 'document_search_config' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."document_search_config" EXCEPT SELECT * FROM main."document_search_config")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."document_search_config" EXCEPT SELECT * FROM baseline."document_search_config")) AS added_rows
UNION ALL
SELECT 'document_search_data' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."document_search_data" EXCEPT SELECT * FROM main."document_search_data")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."document_search_data" EXCEPT SELECT * FROM baseline."document_search_data")) AS added_rows
UNION ALL
SELECT 'document_search_docsize' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."document_search_docsize" EXCEPT SELECT * FROM main."document_search_docsize")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."document_search_docsize" EXCEPT SELECT * FROM baseline."document_search_docsize")) AS added_rows
UNION ALL
SELECT 'document_search_idx' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."document_search_idx" EXCEPT SELECT * FROM main."document_search_idx")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."document_search_idx" EXCEPT SELECT * FROM baseline."document_search_idx")) AS added_rows
UNION ALL
SELECT 'documents' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."documents" EXCEPT SELECT * FROM main."documents")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."documents" EXCEPT SELECT * FROM baseline."documents")) AS added_rows
UNION ALL
SELECT 'download_attempts' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."download_attempts" EXCEPT SELECT * FROM main."download_attempts")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."download_attempts" EXCEPT SELECT * FROM baseline."download_attempts")) AS added_rows
UNION ALL
SELECT 'download_campaign_jobs' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."download_campaign_jobs" EXCEPT SELECT * FROM main."download_campaign_jobs")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."download_campaign_jobs" EXCEPT SELECT * FROM baseline."download_campaign_jobs")) AS added_rows
UNION ALL
SELECT 'download_campaigns' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."download_campaigns" EXCEPT SELECT * FROM main."download_campaigns")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."download_campaigns" EXCEPT SELECT * FROM baseline."download_campaigns")) AS added_rows
UNION ALL
SELECT 'download_jobs' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."download_jobs" EXCEPT SELECT * FROM main."download_jobs")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."download_jobs" EXCEPT SELECT * FROM baseline."download_jobs")) AS added_rows
UNION ALL
SELECT 'equation_atoms' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."equation_atoms" EXCEPT SELECT * FROM main."equation_atoms")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."equation_atoms" EXCEPT SELECT * FROM baseline."equation_atoms")) AS added_rows
UNION ALL
SELECT 'evidence_edges' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."evidence_edges" EXCEPT SELECT * FROM main."evidence_edges")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."evidence_edges" EXCEPT SELECT * FROM baseline."evidence_edges")) AS added_rows
UNION ALL
SELECT 'experiment_revisions' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."experiment_revisions" EXCEPT SELECT * FROM main."experiment_revisions")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."experiment_revisions" EXCEPT SELECT * FROM baseline."experiment_revisions")) AS added_rows
UNION ALL
SELECT 'experiments_cp' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."experiments_cp" EXCEPT SELECT * FROM main."experiments_cp")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."experiments_cp" EXCEPT SELECT * FROM baseline."experiments_cp")) AS added_rows
UNION ALL
SELECT 'external_source_contract_values' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."external_source_contract_values" EXCEPT SELECT * FROM main."external_source_contract_values")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."external_source_contract_values" EXCEPT SELECT * FROM baseline."external_source_contract_values")) AS added_rows
UNION ALL
SELECT 'external_source_contracts' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."external_source_contracts" EXCEPT SELECT * FROM main."external_source_contracts")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."external_source_contracts" EXCEPT SELECT * FROM baseline."external_source_contracts")) AS added_rows
UNION ALL
SELECT 'external_source_contracts_meta' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."external_source_contracts_meta" EXCEPT SELECT * FROM main."external_source_contracts_meta")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."external_source_contracts_meta" EXCEPT SELECT * FROM baseline."external_source_contracts_meta")) AS added_rows
UNION ALL
SELECT 'external_source_dossier_values' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."external_source_dossier_values" EXCEPT SELECT * FROM main."external_source_dossier_values")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."external_source_dossier_values" EXCEPT SELECT * FROM baseline."external_source_dossier_values")) AS added_rows
UNION ALL
SELECT 'external_source_dossiers' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."external_source_dossiers" EXCEPT SELECT * FROM main."external_source_dossiers")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."external_source_dossiers" EXCEPT SELECT * FROM baseline."external_source_dossiers")) AS added_rows
UNION ALL
SELECT 'external_source_dossiers_meta' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."external_source_dossiers_meta" EXCEPT SELECT * FROM main."external_source_dossiers_meta")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."external_source_dossiers_meta" EXCEPT SELECT * FROM baseline."external_source_dossiers_meta")) AS added_rows
UNION ALL
SELECT 'ingest_fingerprints' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."ingest_fingerprints" EXCEPT SELECT * FROM main."ingest_fingerprints")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."ingest_fingerprints" EXCEPT SELECT * FROM baseline."ingest_fingerprints")) AS added_rows
UNION ALL
SELECT 'lacunae' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."lacunae" EXCEPT SELECT * FROM main."lacunae")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."lacunae" EXCEPT SELECT * FROM baseline."lacunae")) AS added_rows
UNION ALL
SELECT 'lane_assignments' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."lane_assignments" EXCEPT SELECT * FROM main."lane_assignments")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."lane_assignments" EXCEPT SELECT * FROM baseline."lane_assignments")) AS added_rows
UNION ALL
SELECT 'links' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."links" EXCEPT SELECT * FROM main."links")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."links" EXCEPT SELECT * FROM baseline."links")) AS added_rows
UNION ALL
SELECT 'literature_novelty_similar_papers' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."literature_novelty_similar_papers" EXCEPT SELECT * FROM main."literature_novelty_similar_papers")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."literature_novelty_similar_papers" EXCEPT SELECT * FROM baseline."literature_novelty_similar_papers")) AS added_rows
UNION ALL
SELECT 'literature_verification_results' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."literature_verification_results" EXCEPT SELECT * FROM main."literature_verification_results")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."literature_verification_results" EXCEPT SELECT * FROM baseline."literature_verification_results")) AS added_rows
UNION ALL
SELECT 'literature_verification_runs' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."literature_verification_runs" EXCEPT SELECT * FROM main."literature_verification_runs")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."literature_verification_runs" EXCEPT SELECT * FROM baseline."literature_verification_runs")) AS added_rows
UNION ALL
SELECT 'mirror_observations' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."mirror_observations" EXCEPT SELECT * FROM main."mirror_observations")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."mirror_observations" EXCEPT SELECT * FROM baseline."mirror_observations")) AS added_rows
UNION ALL
SELECT 'next_action_items' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."next_action_items" EXCEPT SELECT * FROM main."next_action_items")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."next_action_items" EXCEPT SELECT * FROM baseline."next_action_items")) AS added_rows
UNION ALL
SELECT 'notebook_sessions' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."notebook_sessions" EXCEPT SELECT * FROM main."notebook_sessions")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."notebook_sessions" EXCEPT SELECT * FROM baseline."notebook_sessions")) AS added_rows
UNION ALL
SELECT 'proof_atoms' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."proof_atoms" EXCEPT SELECT * FROM main."proof_atoms")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."proof_atoms" EXCEPT SELECT * FROM baseline."proof_atoms")) AS added_rows
UNION ALL
SELECT 'proof_skeletons' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."proof_skeletons" EXCEPT SELECT * FROM main."proof_skeletons")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."proof_skeletons" EXCEPT SELECT * FROM baseline."proof_skeletons")) AS added_rows
UNION ALL
SELECT 'record_sources' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."record_sources" EXCEPT SELECT * FROM main."record_sources")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."record_sources" EXCEPT SELECT * FROM baseline."record_sources")) AS added_rows
UNION ALL
SELECT 'registry_snapshots' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."registry_snapshots" EXCEPT SELECT * FROM main."registry_snapshots")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."registry_snapshots" EXCEPT SELECT * FROM baseline."registry_snapshots")) AS added_rows
UNION ALL
SELECT 'requirements_coverage_gaps' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."requirements_coverage_gaps" EXCEPT SELECT * FROM main."requirements_coverage_gaps")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."requirements_coverage_gaps" EXCEPT SELECT * FROM baseline."requirements_coverage_gaps")) AS added_rows
UNION ALL
SELECT 'requirements_modules' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."requirements_modules" EXCEPT SELECT * FROM main."requirements_modules")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."requirements_modules" EXCEPT SELECT * FROM baseline."requirements_modules")) AS added_rows
UNION ALL
SELECT 'requirements_registry_meta' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."requirements_registry_meta" EXCEPT SELECT * FROM main."requirements_registry_meta")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."requirements_registry_meta" EXCEPT SELECT * FROM baseline."requirements_registry_meta")) AS added_rows
UNION ALL
SELECT 'research_narrative_search' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."research_narrative_search" EXCEPT SELECT * FROM main."research_narrative_search")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."research_narrative_search" EXCEPT SELECT * FROM baseline."research_narrative_search")) AS added_rows
UNION ALL
SELECT 'research_narrative_search_config' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."research_narrative_search_config" EXCEPT SELECT * FROM main."research_narrative_search_config")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."research_narrative_search_config" EXCEPT SELECT * FROM baseline."research_narrative_search_config")) AS added_rows
UNION ALL
SELECT 'research_narrative_search_data' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."research_narrative_search_data" EXCEPT SELECT * FROM main."research_narrative_search_data")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."research_narrative_search_data" EXCEPT SELECT * FROM baseline."research_narrative_search_data")) AS added_rows
UNION ALL
SELECT 'research_narrative_search_docsize' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."research_narrative_search_docsize" EXCEPT SELECT * FROM main."research_narrative_search_docsize")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."research_narrative_search_docsize" EXCEPT SELECT * FROM baseline."research_narrative_search_docsize")) AS added_rows
UNION ALL
SELECT 'research_narrative_search_idx' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."research_narrative_search_idx" EXCEPT SELECT * FROM main."research_narrative_search_idx")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."research_narrative_search_idx" EXCEPT SELECT * FROM baseline."research_narrative_search_idx")) AS added_rows
UNION ALL
SELECT 'research_narratives' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."research_narratives" EXCEPT SELECT * FROM main."research_narratives")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."research_narratives" EXCEPT SELECT * FROM baseline."research_narratives")) AS added_rows
UNION ALL
SELECT 'roadmap_items' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."roadmap_items" EXCEPT SELECT * FROM main."roadmap_items")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."roadmap_items" EXCEPT SELECT * FROM baseline."roadmap_items")) AS added_rows
UNION ALL
SELECT 'source_of_truth_manifest' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."source_of_truth_manifest" EXCEPT SELECT * FROM main."source_of_truth_manifest")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."source_of_truth_manifest" EXCEPT SELECT * FROM baseline."source_of_truth_manifest")) AS added_rows
UNION ALL
SELECT 'theorem_claim_links' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."theorem_claim_links" EXCEPT SELECT * FROM main."theorem_claim_links")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."theorem_claim_links" EXCEPT SELECT * FROM baseline."theorem_claim_links")) AS added_rows
UNION ALL
SELECT 'theorem_identities' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."theorem_identities" EXCEPT SELECT * FROM main."theorem_identities")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."theorem_identities" EXCEPT SELECT * FROM baseline."theorem_identities")) AS added_rows
UNION ALL
SELECT 'theorem_identity_events' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."theorem_identity_events" EXCEPT SELECT * FROM main."theorem_identity_events")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."theorem_identity_events" EXCEPT SELECT * FROM baseline."theorem_identity_events")) AS added_rows
UNION ALL
SELECT 'theorem_identity_evidence' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."theorem_identity_evidence" EXCEPT SELECT * FROM main."theorem_identity_evidence")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."theorem_identity_evidence" EXCEPT SELECT * FROM baseline."theorem_identity_evidence")) AS added_rows
UNION ALL
SELECT 'theorems' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."theorems" EXCEPT SELECT * FROM main."theorems")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."theorems" EXCEPT SELECT * FROM baseline."theorems")) AS added_rows
UNION ALL
SELECT 'todo_items' AS table_name, (SELECT count(*) FROM (SELECT * FROM baseline."todo_items" EXCEPT SELECT * FROM main."todo_items")) AS lost_rows, (SELECT count(*) FROM (SELECT * FROM main."todo_items" EXCEPT SELECT * FROM baseline."todo_items")) AS added_rows;
