#!/bin/sh
# Fetch dark matter bibliography PDFs.
# Uses wget with curl fallback. 3-second delay between arXiv requests.
set -eu

UA='gororoba-fetch/0.1 (research; https://github.com/eirikr/open_gororoba)'
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

ok=0
fail=0

fetch_arxiv() {
  id="$1"; slug="$2"
  if [ -f "${slug}.pdf" ]; then
    echo "SKIP ${slug}.pdf (exists)"
    ok=$((ok + 1))
    return
  fi
  url="https://arxiv.org/pdf/${id}"
  if wget -q -O "${slug}.pdf" --header "User-Agent: ${UA}" --timeout=60 --tries=2 "$url" 2>/dev/null; then
    echo "OK   ${slug}.pdf"
    ok=$((ok + 1))
  elif curl -sS -f -o "${slug}.pdf" -H "User-Agent: ${UA}" --connect-timeout 60 --retry 2 "$url" 2>/dev/null; then
    echo "OK   ${slug}.pdf (curl)"
    ok=$((ok + 1))
  else
    echo "FAIL ${slug}.pdf"
    rm -f "${slug}.pdf"
    fail=$((fail + 1))
  fi
  sleep 3
}

fetch_direct() {
  url="$1"; slug="$2"
  if [ -f "${slug}.pdf" ]; then
    echo "SKIP ${slug}.pdf (exists)"
    ok=$((ok + 1))
    return
  fi
  if wget -q -O "${slug}.pdf" --header "User-Agent: ${UA}" --timeout=60 --tries=2 "$url" 2>/dev/null; then
    echo "OK   ${slug}.pdf"
    ok=$((ok + 1))
  elif curl -sS -f -o "${slug}.pdf" -H "User-Agent: ${UA}" --connect-timeout 60 --retry 2 "$url" 2>/dev/null; then
    echo "OK   ${slug}.pdf (curl)"
    ok=$((ok + 1))
  else
    echo "FAIL ${slug}.pdf"
    rm -f "${slug}.pdf"
    fail=$((fail + 1))
  fi
  sleep 3
}

echo "=== Downloading dark matter bibliography PDFs ==="

# --- Cross-cutting reviews (5 arXiv) ---
fetch_arxiv "2603.03446" "brax_2026_dark_universe_review"
fetch_arxiv "2505.05663" "abdullahi_2025_rich_dark_sectors"
fetch_arxiv "2510.17473" "crnogorcev_2025_dark_matters_icrc"
fetch_arxiv "2602.23708" "bramante_2026_heavy_composite_dm"
fetch_arxiv "2401.12286" "bodas_2024_matter_dm_coincidence"

# --- Collider dark sectors (3 arXiv + 1 CDS) ---
fetch_arxiv "2505.01634" "atlas_2025_semi_visible_jets"
fetch_arxiv "2601.13033" "wilson_edwards_2026_atlas_dark_sectors"
fetch_arxiv "2402.14491" "cms_2024_llp_dimuon"
fetch_direct "https://cds.cern.ch/record/2951492/files/ilten.pdf" "ilten_2025_lhcb_dark_sectors"

# --- Higgs/dark Higgs/scalar portals (12 arXiv) ---
fetch_arxiv "2601.06284" "lomte_2026_higgs_portal_dm"
fetch_arxiv "2602.16822" "huang_2026_overabundant_dm_fopt"
fetch_arxiv "2602.14866" "feng_2026_dark_u1_gw"
fetch_arxiv "2602.18564" "carvalho_correa_2026_z4_freeze_in"
fetch_arxiv "2602.20242" "feiteira_2026_warm_dm_freeze_in"
fetch_arxiv "2601.13147" "das_2026_singlet_fermion_dm"
fetch_arxiv "2504.16525" "yamashita_2025_gravitational_positivity"
fetch_arxiv "2602.20760" "chaudhuri_2026_dark_temperature_gw"
fetch_arxiv "2602.03235" "elgammal_2026_dark_higgs_fcc"
fetch_arxiv "2512.21457" "abed_2025_dark_higgs_hidden_sector"
fetch_arxiv "2406.09971" "atlas_2024_hh_combination"
fetch_arxiv "2205.08582" "cms_2022_llp_muon_pairs"

# --- Vector portals, dark photons, WISPs (7 arXiv) ---
fetch_arxiv "2603.00247" "angel_2026_light_dark_sectors_ephoton"
fetch_arxiv "2602.11405" "gninenko_2026_dark_axion_portal"
fetch_arxiv "2603.03433" "arza_2026_cosmic_wispers"
fetch_arxiv "2603.05006" "ivanov_2026_dielectric_haloscope"
fetch_arxiv "2603.00554" "de_romeri_2026_vector_mediators_cevns"
fetch_arxiv "2602.22308" "barir_2026_nasduck_dark_photon"
fetch_arxiv "2511.07508" "berger_2025_dark_photon_magnetic_fields"

# --- Cosmology, dark energy, surveys (7 arXiv) ---
fetch_arxiv "2503.14738" "desi_2025_dr2_bao"
fetch_arxiv "2602.12238" "pantos_2026_s8_tension_review"
fetch_arxiv "2601.14559" "des_2026_y6_3x2pt"
fetch_arxiv "2602.11310" "escobal_2026_interacting_dark_sector"
fetch_arxiv "2602.21774" "yildiz_2026_dark_sector_fr_gravity"
fetch_arxiv "2601.04048" "pardo_2026_fr_weak_lensing"
fetch_arxiv "2602.22990" "london_2026_voids_dm_probe"

# --- Local/Solar-System probes (7 arXiv) ---
fetch_arxiv "2512.00701" "barranco_2026_compact_dark_objects"
fetch_arxiv "2602.15974" "green_2026_pbh_microlensing"
fetch_arxiv "2507.13794" "mroz_2025_ogle_smc_microlensing"
fetch_arxiv "2509.05910" "turyshev_2025_solar_system_dark_energy"
fetch_arxiv "2507.19577" "thoss_2025_dark_objects_gw"
fetch_arxiv "2603.04841" "gurrola_2026_dark_energy_moon"
fetch_arxiv "2308.12336" "maity_2023_solar_neutrinos_dm"

# --- Direct detection (2 arXiv) ---
fetch_arxiv "2512.08065" "lz_2025_light_dm_cevns"
fetch_arxiv "2601.11296" "xenon_2026_light_dm_ionization"

# --- Asymmetric DM, co-genesis (7 arXiv) ---
fetch_arxiv "2505.22710" "asadi_2025_uv_freeze_in_cogenesis"
fetch_arxiv "2602.20237" "mckenna_2026_leptogenesis_adm"
fetch_arxiv "2601.01849" "takahashi_2026_adm_leptogenesis"
fetch_arxiv "2602.03384" "abhishek_2026_adm_s4_modular"
fetch_arxiv "2507.10655" "mojahed_2025_asymgenesis"
fetch_arxiv "2504.08304" "kanemura_2025_asteroid_mass_soliton"
fetch_arxiv "2602.15132" "di_mauro_2026_antinuclei_sueps"

echo ""
echo "=== Done: ${ok} ok, ${fail} failed ==="
