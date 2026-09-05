use super::*;

impl ProvenanceStore {
    pub fn export_control_plane_compat_paths(
        &mut self,
        repo_root: &Path,
        paths: CompatExportPaths<'_>,
    ) -> Result<()> {
        self.backfill_control_plane_compat_from_snapshots()?;
        let outputs = self.render_control_plane_compat_outputs()?;
        write_text(paths.claims, &outputs.claims)?;
        write_text(paths.insights, &outputs.insights)?;
        write_text(paths.experiments, &outputs.experiments)?;
        write_text(paths.binaries, &outputs.binaries)?;
        write_text(paths.theorems, &outputs.theorems)?;
        write_text(paths.theorems_mirror, &outputs.theorems_mirror)?;
        let transition_events = repo_root.join("registry/claim_transitions.toml");
        let transition_relations = repo_root.join("registry/claim_relations.toml");
        self.export_claim_transition_compat_paths(
            repo_root,
            ClaimTransitionCompatPaths {
                events: &transition_events,
                relations: &transition_relations,
            },
        )?;

        self.record_control_plane_run(
            "export_control_plane",
            &serde_json::json!({
                "claims": to_repo_rel(repo_root, paths.claims),
                "insights": to_repo_rel(repo_root, paths.insights),
                "experiments": to_repo_rel(repo_root, paths.experiments),
                "binaries": to_repo_rel(repo_root, paths.binaries),
                "theorems": to_repo_rel(repo_root, paths.theorems),
                "theorems_mirror": to_repo_rel(repo_root, paths.theorems_mirror),
            })
            .to_string(),
        )?;
        Ok(())
    }

    // Separate path arguments mirror the CLI/export surface; CompatExportPaths
    // keeps the implementation typed after the public wrapper boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn export_control_plane_compat(
        &mut self,
        repo_root: &Path,
        claims_path: &Path,
        insights_path: &Path,
        experiments_path: &Path,
        binaries_path: &Path,
        theorems_path: &Path,
        theorems_mirror_path: &Path,
    ) -> Result<()> {
        self.export_control_plane_compat_paths(
            repo_root,
            CompatExportPaths {
                claims: claims_path,
                insights: insights_path,
                experiments: experiments_path,
                binaries: binaries_path,
                theorems: theorems_path,
                theorems_mirror: theorems_mirror_path,
            },
        )
    }

    pub fn control_plane_compat_text(&mut self, kind: ControlPlaneCompatKind) -> Result<String> {
        self.backfill_control_plane_compat_from_snapshots()?;
        let outputs = self.render_control_plane_compat_outputs()?;
        let text = match kind {
            ControlPlaneCompatKind::Claims => outputs.claims,
            ControlPlaneCompatKind::Insights => outputs.insights,
            ControlPlaneCompatKind::Experiments => outputs.experiments,
            ControlPlaneCompatKind::Binaries => outputs.binaries,
            ControlPlaneCompatKind::Theorems => outputs.theorems,
            ControlPlaneCompatKind::TheoremsMirror => outputs.theorems_mirror,
        };
        Ok(text)
    }

    pub fn verify_control_plane_compat_exports_paths(
        &mut self,
        repo_root: &Path,
        paths: CompatExportPaths<'_>,
    ) -> Result<()> {
        self.backfill_control_plane_compat_from_snapshots()?;
        let outputs = self.render_control_plane_compat_outputs()?;
        let checks = [
            (paths.claims, outputs.claims.as_str()),
            (paths.insights, outputs.insights.as_str()),
            (paths.experiments, outputs.experiments.as_str()),
            (paths.binaries, outputs.binaries.as_str()),
            (paths.theorems, outputs.theorems.as_str()),
            (paths.theorems_mirror, outputs.theorems_mirror.as_str()),
        ];
        let transition_events_path = repo_root.join("registry/claim_transitions.toml");
        let transition_relations_path = repo_root.join("registry/claim_relations.toml");
        let transition_checks =
            if scalar_count(&self.conn, "SELECT COUNT(*) FROM claim_transition_events")? != 0
                || transition_events_path.exists()
                || transition_relations_path.exists()
            {
                let (transition_events, transition_relations) =
                    self.claim_transition_compat_texts()?;
                Some((transition_events, transition_relations))
            } else {
                None
            };
        let mut failures = Vec::new();
        for (path, expected) in checks {
            if !path.exists() {
                failures.push(format!("missing compatibility export {}", path.display()));
                continue;
            }
            let actual = load_text(path)?;
            if actual != compat_render::normalized_export_text(expected) {
                failures.push(format!(
                    "stale compatibility export {} relative to {}",
                    path.display(),
                    repo_root.display()
                ));
            }
        }
        if let Some((transition_events, transition_relations)) = transition_checks {
            for (path, expected) in [
                (transition_events_path.as_path(), transition_events.as_str()),
                (
                    transition_relations_path.as_path(),
                    transition_relations.as_str(),
                ),
            ] {
                if !path.exists() {
                    failures.push(format!("missing compatibility export {}", path.display()));
                    continue;
                }
                let actual = load_text(path)?;
                if actual != compat_render::normalized_export_text(expected) {
                    failures.push(format!(
                        "stale compatibility export {} relative to {}",
                        path.display(),
                        repo_root.display()
                    ));
                }
            }
        }
        if !failures.is_empty() {
            bail!(
                "control-plane compatibility exports failed:\n- {}",
                failures.join("\n- ")
            );
        }
        Ok(())
    }

    // Separate path arguments mirror the CLI/verify surface; CompatExportPaths
    // keeps the implementation typed after the public wrapper boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_control_plane_compat_exports(
        &mut self,
        repo_root: &Path,
        claims_path: &Path,
        insights_path: &Path,
        experiments_path: &Path,
        binaries_path: &Path,
        theorems_path: &Path,
        theorems_mirror_path: &Path,
    ) -> Result<()> {
        self.verify_control_plane_compat_exports_paths(
            repo_root,
            CompatExportPaths {
                claims: claims_path,
                insights: insights_path,
                experiments: experiments_path,
                binaries: binaries_path,
                theorems: theorems_path,
                theorems_mirror: theorems_mirror_path,
            },
        )
    }

    pub fn export_external_sources_compat(
        &mut self,
        repo_root: &Path,
        source_contracts_path: &Path,
        dossiers_registry_path: &Path,
    ) -> Result<()> {
        let outputs = self.render_external_sources_compat_outputs()?;
        write_text(source_contracts_path, &outputs.source_contracts)?;
        write_text(dossiers_registry_path, &outputs.dossiers_registry)?;
        for (path, body) in &outputs.docs {
            write_text(&repo_root.join(path.as_str()), body)?;
        }
        self.record_control_plane_run(
            "export_external_sources",
            &serde_json::json!({
                "source_contracts": to_repo_rel(repo_root, source_contracts_path),
                "dossiers_registry": to_repo_rel(repo_root, dossiers_registry_path),
                "doc_count": outputs.docs.len(),
            })
            .to_string(),
        )?;
        Ok(())
    }

    pub fn verify_external_sources_compat_exports(
        &mut self,
        repo_root: &Path,
        source_contracts_path: &Path,
        dossiers_registry_path: &Path,
    ) -> Result<()> {
        let outputs = self.render_external_sources_compat_outputs()?;
        let mut failures = Vec::new();
        for (path, expected) in [
            (source_contracts_path, outputs.source_contracts.as_str()),
            (dossiers_registry_path, outputs.dossiers_registry.as_str()),
        ] {
            if !path.exists() {
                failures.push(format!("missing compatibility export {}", path.display()));
                continue;
            }
            let actual = load_text(path)?;
            if actual != compat_render::normalized_export_text(expected) {
                failures.push(format!("stale compatibility export {}", path.display()));
            }
        }
        for (path, expected) in outputs.docs {
            let full = repo_root.join(path.as_str());
            if !full.exists() {
                failures.push(format!("missing generated dossier {}", full.display()));
                continue;
            }
            let actual = load_text(&full)?;
            if actual != compat_render::normalized_export_text(&expected) {
                failures.push(format!("stale dossier export {}", full.display()));
            }
        }
        if !failures.is_empty() {
            bail!(
                "external-source compatibility exports failed:\n- {}",
                failures.join("\n- ")
            );
        }
        Ok(())
    }

    pub(super) fn render_control_plane_compat_outputs(&self) -> Result<ControlPlaneCompatOutputs> {
        let theorem_rows = self.list_theorems()?;
        let experiments_meta = self
            .control_plane_meta_toml("experiments")?
            .unwrap_or_default();
        Ok(ControlPlaneCompatOutputs {
            claims: self.overlay_claim_evidence(render_claims_registry(&self.list_claims()?))?,
            insights: render_insights_registry(&self.list_insights_for_compat()?),
            experiments: render_experiments_registry(
                &experiments_meta,
                &self.list_experiments_for_compat()?,
            ),
            binaries: render_binaries_registry(&self.list_binaries()?),
            theorems: render_theorem_markdown(
                "SQLite canonical database (compatibility export)",
                &theorem_rows,
            ),
            theorems_mirror: render_theorem_markdown(
                "registry/canonical/control_plane.sqlite3",
                &theorem_rows,
            ),
        })
    }

    pub(super) fn render_external_sources_compat_outputs(
        &self,
    ) -> Result<ExternalSourcesCompatOutputs> {
        let contracts_meta = self.external_source_contracts_meta()?;
        let contracts = self.list_external_source_contracts()?;
        let dossiers_meta = self.external_source_dossiers_meta()?;
        let dossiers = self.list_external_source_dossiers()?;
        let docs = dossiers
            .iter()
            .map(|dossier| {
                (
                    Utf8PathBuf::from(dossier.source_markdown.clone()),
                    render_external_source_dossier_markdown(dossier),
                )
            })
            .collect();
        Ok(ExternalSourcesCompatOutputs {
            source_contracts: render_external_source_contracts_registry(
                &contracts_meta,
                &contracts,
            ),
            dossiers_registry: render_external_source_dossiers_registry(&dossiers_meta, &dossiers),
            docs,
        })
    }
}
