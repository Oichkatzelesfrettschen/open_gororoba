//! <!-- AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT. -->
//! <!-- Source of truth: registry/external_sources.toml -->
//! <!-- Canonical write path: registry/canonical/control_plane.sqlite3 -->
//! <!-- Source label: XS-022 -->
//! <!-- Regenerate with: cargo run -p gororoba_cli_data --bin provenance -- export-external-sources -->
//!
//! # Wow! Signal Source Dossier
//!
//! Status: **ACTIVE**
//! Content kind: Claim-dataset provenance chain
//! Authority level: Primary dataset index
//! Verification level: Source capture
//!
//! ## Overview
//!
//! This dossier documents the provenance chain for the 1977 Wow! signal archival
//! data and the Breakthrough Listen (BL) 6EQUJ5 follow-up campaign. It supports
//! claims C-769 through C-773 in the open_gororoba registry.
//!
//! ## Primary Sources
//!
//! ### 1. Ohio History Connection -- Archival Printout Scan
//!
//! - **Collection**: Ohio State University Radio Observatory Records
//! - **IIIF Image**: `https://cdm16007.contentdm.oclc.org/iiif/2/p267401coll32:12429/full/full/0/default.jpg`
//! - **IIIF Manifest**: `https://cdm16007.contentdm.oclc.org/iiif/2/p267401coll32:12429/manifest.json`
//! - **Format**: JPEG, full-resolution scan of the 1977-08-15 computer printout
//! - **Content**: 50-channel receiver output in base-36 encoding (space=0, 0-9, A-Z=10-35)
//! - **Signal**: Channel 2, row sequence 6EQUJ5 = intensities [6, 14, 26, 30, 19, 5]
//! - **Access**: Public domain (government records), CC0 equivalent
//! - **Verification**: JPEG magic bytes FF D8 FF (passes validate_not_html check)
//!
//! ### 2. Big Ear Memorial Website
//!
//! - **URL**: `https://www.bigear.org/6equj5.htm`
//! - **Content**: Jerry Ehman's annotated explanation of the 6EQUJ5 sequence
//! - **Key fact**: The peak intensity U=30 corresponds to ~30 sigma above noise
//! - **Note**: Fan-maintained site, not an institutional source. Cross-reference with
//! Ehman (1997) for authoritative details.
//!
//! ### 3. Breakthrough Listen 6EQUJ5 Campaign Hub
//!
//! - **URL**: `https://seti.berkeley.edu/wow/`
//! - **Content**: Campaign overview, observation strategy, data release announcements
//! - **Institution**: Berkeley SETI Research Center
//! - **Telescope**: Green Bank Telescope (GBT), 100m dish
//! - **Strategy**: ABACAD cadence (ON-OFF-ON-OFF-ON-OFF) for RFI discrimination
//!
//! ### 4. BL 6EQUJ5 Data Directory
//!
//! - **Root**: `https://bldata.berkeley.edu/6EQUJ5/`
//! - **Structure**: `di_YYYYMMDD/spliced_blcNN_guppi_MJDTTTTT_TARGET_OBSNUM/`
//! - **Products**: Filterbank files (.fil), high-resolution spectrograms
//! - **CAUTION**: Directory listings return HTML -- never use as provider URL.
//! Only explicit file paths (ending in .fil or .h5) are valid download targets.
//!
//! ## Key Bibliography
//!
//! | ID | Citation | DOI/URL |
//! |----|----------|---------|
//! | BIB-0212 | Ehman, J. R. (1997). "The Big Ear Wow! Signal: What We Know and Don't Know About It After 20 Years." | bigear.org/wow20th.htm |
//! | BIB-0213 | Gray, R. H. & Ellingsen, S. P. (2002). "A Search for Periodic Emissions at the Wow Locale." ApJ 578, 967-971. | 10.1086/342646 |
//! | BIB-0214 | Perez, K. et al. (2022). "Breakthrough Listen Observations of the Wow! Signal Region." RNAAS 6, 197. | 10.3847/2515-5172/ac9631 |
//!
//! ## Signal Parameters
//!
//! - **Frequency**: 1420.405751786 MHz (21 cm hydrogen line)
//! - **Duration**: 72 seconds (full beam transit time for a point source)
//! - **RA (J2000)**: ~19h 25m 31s (reconstructed from beam positions)
//! - **Dec (J2000)**: ~-27d 03m (beam 2 of 2-beam feed horn array)
//! - **Epoch**: 1977-08-15 22:16:00 UT (Eastern Daylight Time 18:16)
//! - **Peak SNR**: ~30 sigma (U = 30 in base-36)
//! - **Bandwidth**: < 10 kHz (narrowband, consistent with artificial origin hypothesis)
//!
//! ## Provenance Chain
//!
//! 1. Ohio History Connection IIIF endpoint -> `data/external/wow_1977_printout.jpg`
//! 2. Manual transcription verified against Ehman (1997) -> `data/csv/wow1977_transcription.csv`
//! 3. BL 6EQUJ5 GBT manifest (curated from bldata.berkeley.edu) -> `data/csv/bl_6equj5_gbt_manifest.csv`
//! 4. Claims C-769..C-773 reference these artifacts with SHA-256 checksums
//!
//! ## Falsifiable Theses
//!
//! - **WT-001**: The 6EQUJ5 transcription matches the archival printout character-for-character.
//! - **WT-002**: Earth-motion drift rate at the Wow! locale is bounded by |drift| < 0.37 Hz/s.
//! - **WT-003**: BL ABACAD cadence discriminates the Wow! locale from terrestrial RFI.
//! - **WT-004**: ON and OFF cadence pointings are topologically indistinguishable (null result).
//! - **WT-005**: Candidate feature vectors show non-trivial ultrametric structure vs null.
//!
