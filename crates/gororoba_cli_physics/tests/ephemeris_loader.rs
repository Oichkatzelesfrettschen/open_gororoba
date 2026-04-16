use gororoba_cli_physics::ephemeris_loader::{EphemerisLoader, flyby_epochs};
use std::path::PathBuf;

fn try_load() -> Option<EphemerisLoader> {
    let bsp = PathBuf::from("data/external/de440.bsp");
    EphemerisLoader::load(&bsp).ok()
}

fn norm3(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[test]
fn test_load_requires_bsp() {
    let bogus = PathBuf::from("/nonexistent/de440.bsp");
    assert!(EphemerisLoader::load(&bogus).is_err());
}

#[test]
fn test_moon_distance_if_bsp_available() {
    let Some(loader) = try_load() else { return };
    let pos = loader.moon_geocentric_j2000(flyby_epochs::NEAR);
    assert!(pos[0].is_finite());
    let dist = norm3(&pos);
    assert!(
        dist > 350_000.0 && dist < 410_000.0,
        "Moon distance {} km out of range",
        dist
    );
}

#[test]
fn test_sun_distance_if_bsp_available() {
    let Some(loader) = try_load() else { return };
    let pos = loader.sun_geocentric_j2000(flyby_epochs::NEAR);
    assert!(pos[0].is_finite());
    let dist = norm3(&pos);
    assert!(
        dist > 1.45e8 && dist < 1.55e8,
        "Sun distance {} km out of range (expected ~1 AU)",
        dist
    );
}

#[test]
fn test_emb_offset_if_bsp_available() {
    let Some(loader) = try_load() else { return };
    let emb = loader.earth_moon_barycenter(flyby_epochs::NEAR);
    let dist = norm3(&emb);
    assert!(
        dist > 3000.0 && dist < 6000.0,
        "EMB offset {} km out of range (expected ~4670 km)",
        dist
    );
}

#[test]
fn test_three_body_all_flybys_if_bsp_available() {
    let Some(loader) = try_load() else { return };
    let epochs = [
        flyby_epochs::GALILEO,
        flyby_epochs::NEAR,
        flyby_epochs::CASSINI,
        flyby_epochs::ROSETTA_I,
        flyby_epochs::MESSENGER,
        flyby_epochs::JUNO,
    ];
    for &jed in &epochs {
        let state = loader.three_body_state(jed);
        assert!(
            norm3(&state.moon_pos_km) > 300_000.0,
            "Moon too close at JED {}",
            jed
        );
        assert!(
            norm3(&state.sun_pos_km) > 1.4e8,
            "Sun too close at JED {}",
            jed
        );
        assert!(
            norm3(&state.emb_offset_km) < 10_000.0,
            "EMB offset too large at JED {}",
            jed
        );
    }
}

const _: () = {
    assert!(flyby_epochs::GALILEO < flyby_epochs::NEAR);
    assert!(flyby_epochs::NEAR < flyby_epochs::CASSINI);
    assert!(flyby_epochs::CASSINI < flyby_epochs::ROSETTA_I);
    assert!(flyby_epochs::ROSETTA_I < flyby_epochs::MESSENGER);
    assert!(flyby_epochs::MESSENGER < flyby_epochs::JUNO);
};
