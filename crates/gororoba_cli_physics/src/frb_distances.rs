//! Explicit CHIME dispersion-column selection and bounded distance conversion.

use anyhow::{Context, Result, ensure};
use clap::ValueEnum;
use cosmology_core::distances::{
    comoving_distance, dm_excess_to_redshift, planck2018, radec_to_cartesian,
};
use data_core::catalogs::chime::{FrbEvent, parse_chime_csv_with_required_columns};
use std::path::Path;

/// Whole-point or distance-label permutations preserve the tested geometry.
/// Retain their numerical summaries as diagnostics rather than physical verdicts.
pub const RELABELING_DIAGNOSTIC: &str = "UncalibratedGeometryRelabeling";

/// Galactic-model excess columns accepted by the cosmic-DM conversion.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum DmExcessColumn {
    #[value(name = "dm_exc_ne2001")]
    Ne2001,
    #[value(name = "dm_exc_ymw16")]
    Ymw16,
}

impl DmExcessColumn {
    pub fn header(self) -> &'static str {
        match self {
            Self::Ne2001 => "dm_exc_ne2001",
            Self::Ymw16 => "dm_exc_ymw16",
        }
    }

    fn excess(self, event: &FrbEvent) -> f64 {
        match self {
            Self::Ne2001 => event.dm_exc_ne2001,
            Self::Ymw16 => event.dm_exc_ymw16,
        }
    }
}

#[derive(Debug)]
pub struct FrbDistance {
    pub name: String,
    pub redshift: f64,
    pub comoving_mpc: f64,
    pub cartesian_mpc: (f64, f64, f64),
}

/// Counts use the parser's positive, finite bonsai-DM population as denominator.
#[derive(Debug, Default)]
pub struct FrbDistanceSelection {
    pub parsed_rows: usize,
    pub invalid_excess: usize,
    pub invalid_sky_position: usize,
    pub nonpositive_cosmic_dm: usize,
    pub distances: Vec<FrbDistance>,
}

/// Convert a selected Galactic-model excess with a fixed observer-frame host.
/// Catalog halo completeness and physical host provenance require separate admission.
/// Missing/duplicate selected columns and inversion failures abort the conversion.
pub fn load_frb_distances(
    path: &Path,
    column: DmExcessColumn,
    host_observer: f64,
) -> Result<FrbDistanceSelection> {
    ensure!(
        host_observer.is_finite() && host_observer >= 0.0,
        "observer-frame host DM must be finite and nonnegative"
    );
    let events =
        parse_chime_csv_with_required_columns(path, &["bonsai_dm", "ra", "dec", column.header()])
            .with_context(|| format!("read CHIME column {}", column.header()))?;
    let mut selected = FrbDistanceSelection {
        parsed_rows: events.len(),
        ..Default::default()
    };
    for event in &events {
        let excess = column.excess(event);
        if !excess.is_finite() || excess <= 0.0 {
            selected.invalid_excess += 1;
            continue;
        }
        if !event.ra.is_finite()
            || !(0.0..360.0).contains(&event.ra)
            || !event.dec.is_finite()
            || !(-90.0..=90.0).contains(&event.dec)
        {
            selected.invalid_sky_position += 1;
            continue;
        }
        let cosmic_dm = excess - host_observer;
        if cosmic_dm <= 0.0 {
            selected.nonpositive_cosmic_dm += 1;
            continue;
        }
        let redshift = dm_excess_to_redshift(
            cosmic_dm,
            planck2018::OMEGA_M,
            planck2018::OMEGA_B,
            planck2018::H0,
        )
        .with_context(|| {
            format!(
                "FRB {}: invert {} cosmic DM {cosmic_dm}",
                event.tns_name,
                column.header()
            )
        })?;
        let comoving_mpc = comoving_distance(redshift, planck2018::OMEGA_M, planck2018::H0);
        ensure!(
            comoving_mpc.is_finite() && comoving_mpc > 0.0,
            "FRB {}: invalid comoving distance",
            event.tns_name
        );
        selected.distances.push(FrbDistance {
            name: event.tns_name.clone(),
            redshift,
            comoving_mpc,
            cartesian_mpc: radec_to_cartesian(event.ra, event.dec, comoving_mpc),
        });
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn catalog(body: &str) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(body.as_bytes()).unwrap();
        file
    }

    #[test]
    fn requested_column_controls_the_actual_distance() {
        let input = catalog(
            "tns_name,bonsai_dm,ra,dec,dm_exc_ne2001,dm_exc_ymw16\nFRB,1600,10,20,500,1500\n",
        );
        let ne = load_frb_distances(input.path(), DmExcessColumn::Ne2001, 50.0).unwrap();
        let ymw = load_frb_distances(input.path(), DmExcessColumn::Ymw16, 50.0).unwrap();
        assert_eq!(ne.distances.len(), 1);
        assert_eq!(ymw.distances.len(), 1);
        assert!(ymw.distances[0].redshift > 2.0 * ne.distances[0].redshift);
        assert!(ymw.distances[0].comoving_mpc > ne.distances[0].comoving_mpc);
        assert!(<DmExcessColumn as ValueEnum>::from_str("bonsai_dm", false).is_err());
    }

    #[test]
    fn selected_column_and_foreground_errors_fail_closed() {
        let input = catalog("bonsai_dm,ra,dec,dm_exc_ne2001\n100,10,20,80\n");
        assert!(load_frb_distances(input.path(), DmExcessColumn::Ymw16, 0.0).is_err());
        for host in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(load_frb_distances(input.path(), DmExcessColumn::Ne2001, host).is_err());
        }
        let huge = catalog("bonsai_dm,ra,dec,dm_exc_ne2001\n1e100,10,20,1e100\n");
        assert!(load_frb_distances(huge.path(), DmExcessColumn::Ne2001, 0.0).is_err());
    }

    #[test]
    fn exclusions_partition_the_parsed_population() {
        let input = catalog(
            "bonsai_dm,ra,dec,dm_exc_ne2001\n100,10,20,80\n100,10,20,inf\n100,inf,20,80\n100,10,91,80\n100,10,20,30\n",
        );
        let sample = load_frb_distances(input.path(), DmExcessColumn::Ne2001, 50.0).unwrap();
        assert_eq!(sample.parsed_rows, 5);
        assert_eq!(sample.distances.len(), 1);
        assert_eq!(sample.invalid_excess, 1);
        assert_eq!(sample.invalid_sky_position, 2);
        assert_eq!(sample.nonpositive_cosmic_dm, 1);
    }
}
