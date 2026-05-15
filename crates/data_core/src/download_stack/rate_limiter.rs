//! Per-host rate limiter for outbound network requests.
//!
//! The limiter sits in front of every backend dispatch through
//! `DownloadStack` and the `fetch_text` high-level path. If no limit
//! applies to a host the `gate()` call is a microsecond no-op.
//!
//! The default registry is seeded from the per-host minimum delays
//! published in `registry/data_servers.toml` as of 2026-04-17. Call
//! `RateLimiter::with_limit(...)` or rebuild via the registry loader
//! to extend the table at construction time.

use std::{
    collections::HashMap,
    sync::Mutex,
    time::{Duration, Instant},
};

use super::host_matches_suffix;

/// Per-host minimum-delay rate limiter.
///
/// Gates outbound requests so a single thread does not hammer a host
/// faster than its policy allows. The limiter holds a registry of
/// host-suffix -> minimum delay rules and a per-host last-request
/// timestamp map.
///
/// Hosts are matched suffix-style ("api.foo.com" matches the rule
/// "foo.com"), letting one rule cover an entire vendor with subdomains.
/// The gate is invoked from every backend dispatch (execute_with_trace)
/// and the high-level `fetch_text` path. If no limit applies to a host,
/// gating is a no-op (~ns).
///
/// Internally uses a `Mutex<HashMap<host, Instant>>` tracking the
/// last-request timestamp per host. On re-entry the gate computes
/// `delay - elapsed` and sleeps if positive, releasing the mutex
/// before sleeping to avoid serializing unrelated hosts.
#[derive(Debug)]
pub struct RateLimiter {
    /// Host-suffix -> minimum inter-request delay.
    limits: Vec<(String, Duration)>,
    last_request: Mutex<HashMap<String, Instant>>,
}

impl RateLimiter {
    /// Empty limiter: no host limited.
    pub fn empty() -> Self {
        Self {
            limits: Vec::new(),
            last_request: Mutex::new(HashMap::new()),
        }
    }

    /// Default limiter seeded from `registry/data_servers.toml` values
    /// as of 2026-04-17. Keep in sync with that registry or call
    /// `load_from_data_servers_toml` to pull fresh at init.
    pub fn with_registry_defaults() -> Self {
        Self {
            limits: vec![
                // LoTSS / ASTRON VO
                ("vo.astron.nl".to_string(), Duration::from_millis(250)),
                ("astrowise.org".to_string(), Duration::from_millis(250)),
                // HEASARC
                (
                    "heasarc.gsfc.nasa.gov".to_string(),
                    Duration::from_millis(250),
                ),
                // MAST
                ("mast.stsci.edu".to_string(), Duration::from_millis(250)),
                // CDS VizieR
                (
                    "cdsarc.cds.unistra.fr".to_string(),
                    Duration::from_millis(250),
                ),
                (
                    "cdsarc.u-strasbg.fr".to_string(),
                    Duration::from_millis(250),
                ),
                // SDSS
                ("data.sdss.org".to_string(), Duration::from_millis(250)),
                // Zenodo
                ("zenodo.org".to_string(), Duration::from_millis(250)),
                // HEPData
                ("hepdata.net".to_string(), Duration::from_millis(250)),
                // McGill, SORCE, magnetar
                ("lasp.colorado.edu".to_string(), Duration::from_millis(250)),
                // Bartol legacy FTP (slower)
                (
                    "ftp.bartol.udel.edu".to_string(),
                    Duration::from_millis(500),
                ),
                // GWOSC
                ("gwosc.org".to_string(), Duration::from_millis(500)),
                // Materials / AFLOW / JARVIS (slower per ToS)
                ("aflow.org".to_string(), Duration::from_millis(500)),
                ("jarvis.nist.gov".to_string(), Duration::from_millis(500)),
                (
                    "breakthroughinitiatives.org".to_string(),
                    Duration::from_millis(500),
                ),
                // BepiColombo / ESA Euclid Q1 (per ESA ToS)
                (
                    "easdr1.esac.esa.int".to_string(),
                    Duration::from_millis(500),
                ),
                // Gaia TAP (ESA recommendation)
                ("gea.esac.esa.int".to_string(), Duration::from_millis(1000)),
                // AMDA (IRAP)
                ("amda.irap.omp.eu".to_string(), Duration::from_millis(500)),
            ],
            last_request: Mutex::new(HashMap::new()),
        }
    }

    /// Override / extend limits at construction.
    pub fn with_limit(mut self, host_suffix: impl Into<String>, delay: Duration) -> Self {
        self.limits.push((host_suffix.into(), delay));
        self
    }

    /// Returns the min-delay that would apply to this host, if any.
    pub fn delay_for(&self, host: &str) -> Option<Duration> {
        self.limits
            .iter()
            .find(|(suffix, _)| host_matches_suffix(host, suffix))
            .map(|(_, d)| *d)
    }

    /// Gate a request to `host`. If the previous gated request to this
    /// host landed less than the configured delay ago, block the current
    /// thread for the remainder. No-op if no limit applies.
    pub fn gate(&self, host: &str) {
        let Some(delay) = self.delay_for(host) else {
            return;
        };
        // Compute wait inside the lock, sleep outside it.
        let wait: Option<Duration> = {
            let mut map = self
                .last_request
                .lock()
                .expect("rate-limiter mutex poisoned");
            let now = Instant::now();
            let w = map.get(host).and_then(|last| {
                let elapsed = now.duration_since(*last);
                if elapsed < delay {
                    Some(delay - elapsed)
                } else {
                    None
                }
            });
            // Reserve the slot now so a concurrent gate() on the same
            // host computes its wait against our anticipated completion,
            // not a stale value.
            map.insert(host.to_string(), now + w.unwrap_or_default());
            w
        };
        if let Some(w) = wait {
            std::thread::sleep(w);
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::with_registry_defaults()
    }
}
