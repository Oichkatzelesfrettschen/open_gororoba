//! paper_fetcher: Pure Rust port of `papers/pdf/fetch_dm_papers.sh`.
//! Fetches dark matter bibliography PDFs.

use std::fs;
use std::path::Path;
use std::thread::sleep;
use std::time::Duration;
use ureq;

const UA: &str = "gororoba-fetch/0.1 (research; https://github.com/eirikr/open_gororoba)";

fn fetch(url: &str, slug: &str) -> bool {
    let filename = format!("{}.pdf", slug);
    if Path::new(&filename).exists() {
        println!("SKIP {} (exists)", filename);
        return true;
    }

    println!("FETCHING {}...", url);
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .user_agent(UA)
        .timeout_global(Some(Duration::from_secs(60)))
        .build()
        .into();

    let response = agent.get(url).call();

    match response {
        Ok(res) => {
            let mut file = match fs::File::create(&filename) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("FAIL {} (create file: {})", filename, e);
                    return false;
                }
            };
            if let Err(e) = std::io::copy(&mut res.into_body().into_reader(), &mut file) {
                eprintln!("FAIL {} (copy body: {})", filename, e);
                let _ = fs::remove_file(&filename);
                return false;
            }
            println!("OK   {}", filename);
            true
        }
        Err(e) => {
            eprintln!("FAIL {} (request: {})", filename, e);
            false
        }
    }
}

fn fetch_arxiv(id: &str, slug: &str) -> bool {
    let url = format!("https://arxiv.org/pdf/{}", id);
    let success = fetch(&url, slug);
    sleep(Duration::from_secs(3));
    success
}

fn fetch_direct(url: &str, slug: &str) -> bool {
    let success = fetch(url, slug);
    sleep(Duration::from_secs(3));
    success
}

fn main() {
    let mut ok = 0;
    let mut fail = 0;

    println!("=== Downloading dark matter bibliography PDFs ===");

    let papers = vec![
        ("arxiv", "2603.03446", "brax_2026_dark_universe_review"),
        ("arxiv", "2505.05663", "abdullahi_2025_rich_dark_sectors"),
        ("arxiv", "2510.17473", "crnogorcev_2025_dark_matters_icrc"),
        ("arxiv", "2602.23708", "bramante_2026_heavy_composite_dm"),
        ("arxiv", "2401.12286", "bodas_2024_matter_dm_coincidence"),
        ("arxiv", "2505.01634", "atlas_2025_semi_visible_jets"),
        ("arxiv", "2601.13033", "wilson_edwards_2026_atlas_dark_sectors"),
        ("arxiv", "2402.14491", "cms_2024_llp_dimuon"),
        ("direct", "https://cds.cern.ch/record/2951492/files/ilten.pdf", "ilten_2025_lhcb_dark_sectors"),
        ("arxiv", "2601.06284", "lomte_2026_higgs_portal_dm"),
        ("arxiv", "2602.16822", "huang_2026_overabundant_dm_fopt"),
        ("arxiv", "2602.14866", "feng_2026_dark_u1_gw"),
        ("arxiv", "2602.18564", "carvalho_correa_2026_z4_freeze_in"),
        ("arxiv", "2602.20242", "feiteira_2026_warm_dm_freeze_in"),
        ("arxiv", "2601.13147", "das_2026_singlet_fermion_dm"),
        ("arxiv", "2504.16525", "yamashita_2025_gravitational_positivity"),
        ("arxiv", "2602.20760", "chaudhuri_2026_dark_temperature_gw"),
        ("arxiv", "2602.03235", "elgammal_2026_dark_higgs_fcc"),
        ("arxiv", "2512.21457", "abed_2025_dark_higgs_hidden_sector"),
        ("arxiv", "2406.09971", "atlas_2024_hh_combination"),
        ("arxiv", "2205.08582", "cms_2022_llp_muon_pairs"),
        ("arxiv", "2603.00247", "angel_2026_light_dark_sectors_ephoton"),
        ("arxiv", "2602.11405", "gninenko_2026_dark_axion_portal"),
        ("arxiv", "2603.03433", "arza_2026_cosmic_wispers"),
        ("arxiv", "2603.05006", "ivanov_2026_dielectric_haloscope"),
        ("arxiv", "2603.00554", "de_romeri_2026_vector_mediators_cevns"),
        ("arxiv", "2602.22308", "barir_2026_nasduck_dark_photon"),
        ("arxiv", "2511.07508", "berger_2025_dark_photon_magnetic_fields"),
        ("arxiv", "2503.14738", "desi_2025_dr2_bao"),
        ("arxiv", "2602.12238", "pantos_2026_s8_tension_review"),
        ("arxiv", "2601.14559", "des_2026_y6_3x2pt"),
        ("arxiv", "2602.11310", "escobal_2026_interacting_dark_sector"),
        ("arxiv", "2602.21774", "yildiz_2026_dark_sector_fr_gravity"),
        ("arxiv", "2601.04048", "pardo_2026_fr_weak_lensing"),
        ("arxiv", "2602.22990", "london_2026_voids_dm_probe"),
        ("arxiv", "2512.00701", "barranco_2026_compact_dark_objects"),
        ("arxiv", "2602.15974", "green_2026_pbh_microlensing"),
        ("arxiv", "2507.13794", "mroz_2025_ogle_smc_microlensing"),
        ("arxiv", "2509.05910", "turyshev_2025_solar_system_dark_energy"),
        ("arxiv", "2507.19577", "thoss_2025_dark_objects_gw"),
        ("arxiv", "2603.04841", "gurrola_2026_dark_energy_moon"),
        ("arxiv", "2308.12336", "maity_2023_solar_neutrinos_dm"),
        ("arxiv", "2512.08065", "lz_2025_light_dm_cevns"),
        ("arxiv", "2601.11296", "xenon_2026_light_dm_ionization"),
        ("arxiv", "2505.22710", "asadi_2025_uv_freeze_in_cogenesis"),
        ("arxiv", "2602.20237", "mckenna_2026_leptogenesis_adm"),
        ("arxiv", "2601.01849", "takahashi_2026_adm_leptogenesis"),
        ("arxiv", "2602.03384", "abhishek_2026_adm_s4_modular"),
        ("arxiv", "2507.10655", "mojahed_2025_asymgenesis"),
        ("arxiv", "2504.08304", "kanemura_2025_asteroid_mass_soliton"),
        ("arxiv", "2602.15132", "di_mauro_2026_antinuclei_sueps"),
    ];

    for (kind, id_or_url, slug) in papers {
        let success = match kind {
            "arxiv" => fetch_arxiv(id_or_url, slug),
            "direct" => fetch_direct(id_or_url, slug),
            _ => false,
        };
        if success {
            ok += 1;
        } else {
            fail += 1;
        }
    }

    println!("\n=== Done: {} ok, {} failed ===", ok, fail);
}
