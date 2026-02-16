use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

fn spawn_gpu_telemetry_sampler_nvidia_smi(
    out_path: PathBuf,
    interval: Duration,
) -> Option<(Arc<AtomicBool>, thread::JoinHandle<()>)> {
    if Command::new("nvidia-smi").arg("--help").output().is_err() {
        eprintln!(
            "[WARN] nvidia-smi unavailable; skipping GPU telemetry sampler at {}",
            out_path.display()
        );
        return None;
    }
    let stop = Arc::new(AtomicBool::new(false));
    let stop_worker = Arc::clone(&stop);
    let handle = thread::spawn(move || {
        if let Some(parent) = out_path.parent() {
            if let Err(err) = std::fs::create_dir_all(parent) {
                eprintln!(
                    "[WARN] failed to create telemetry dir {}: {}",
                    parent.display(),
                    err
                );
                return;
            }
        }
        let mut file = match std::fs::File::create(&out_path) {
            Ok(file) => file,
            Err(err) => {
                eprintln!(
                    "[WARN] failed to create telemetry file {}: {}",
                    out_path.display(),
                    err
                );
                return;
            }
        };
        let _ = writeln!(
            file,
            "elapsed_s,timestamp,temp_c,sm_clock_mhz,mem_clock_mhz,power_w,gpu_util_pct,mem_util_pct,status"
        );

        let start = Instant::now();
        while !stop_worker.load(Ordering::Relaxed) {
            let elapsed_s = start.elapsed().as_secs_f64();
            let sample = Command::new("nvidia-smi")
                .args([
                    "--query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,utilization.memory",
                    "--format=csv,noheader,nounits",
                ])
                .output();

            match sample {
                Ok(output) if output.status.success() => {
                    let raw = String::from_utf8_lossy(&output.stdout);
                    let line = raw.lines().next().unwrap_or("");
                    let columns: Vec<&str> = line.split(',').map(|v| v.trim()).collect();
                    if columns.len() >= 7 {
                        let _ = writeln!(
                            file,
                            "{:.3},{},{},{},{},{},{},{},ok",
                            elapsed_s,
                            columns[0],
                            columns[1],
                            columns[2],
                            columns[3],
                            columns[4],
                            columns[5],
                            columns[6]
                        );
                    } else {
                        let _ = writeln!(
                            file,
                            "{:.3},,,,,,,,parse_error:{}",
                            elapsed_s,
                            line.replace(',', ";")
                        );
                    }
                }
                Ok(output) => {
                    let _ = writeln!(
                        file,
                        "{:.3},,,,,,,,cmd_status_{}",
                        elapsed_s,
                        output.status.code().unwrap_or(-1)
                    );
                }
                Err(err) => {
                    let _ = writeln!(file, "{:.3},,,,,,,,cmd_error:{}", elapsed_s, err);
                }
            }
            let _ = file.flush();
            thread::sleep(interval);
        }
    });
    Some((stop, handle))
}

#[cfg(feature = "gpu")]
fn spawn_gpu_telemetry_sampler_nvml(
    out_path: PathBuf,
    interval: Duration,
) -> Option<(Arc<AtomicBool>, thread::JoinHandle<()>)> {
    use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
    use nvml_wrapper::Nvml;

    // Validate NVML availability up front so callers can fall back cleanly.
    let nvml = Nvml::init().ok()?;
    let _ = nvml.device_by_index(0).ok()?;

    let stop = Arc::new(AtomicBool::new(false));
    let stop_worker = Arc::clone(&stop);
    let handle = thread::spawn(move || {
        let nvml = match Nvml::init() {
            Ok(nvml) => nvml,
            Err(err) => {
                eprintln!("[WARN] NVML init failed in telemetry thread: {err}");
                return;
            }
        };
        let device = match nvml.device_by_index(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("[WARN] NVML device(0) failed in telemetry thread: {err}");
                return;
            }
        };

        if let Some(parent) = out_path.parent() {
            if let Err(err) = std::fs::create_dir_all(parent) {
                eprintln!(
                    "[WARN] failed to create telemetry dir {}: {}",
                    parent.display(),
                    err
                );
                return;
            }
        }
        let mut file = match std::fs::File::create(&out_path) {
            Ok(file) => file,
            Err(err) => {
                eprintln!(
                    "[WARN] failed to create telemetry file {}: {}",
                    out_path.display(),
                    err
                );
                return;
            }
        };
        let _ = writeln!(
            file,
            "elapsed_s,timestamp,temp_c,sm_clock_mhz,mem_clock_mhz,power_w,gpu_util_pct,mem_util_pct,status"
        );

        let start = Instant::now();
        while !stop_worker.load(Ordering::Relaxed) {
            let elapsed_s = start.elapsed().as_secs_f64();
            let timestamp = chrono::Utc::now().to_rfc3339();

            let temp_c = device.temperature(TemperatureSensor::Gpu).ok();
            let sm_clock_mhz = device.clock_info(Clock::SM).ok();
            let mem_clock_mhz = device.clock_info(Clock::Memory).ok();
            let power_w = device.power_usage().ok().map(|mw| mw as f64 / 1000.0);
            let util = device.utilization_rates().ok();

            let status = if temp_c.is_some()
                && sm_clock_mhz.is_some()
                && mem_clock_mhz.is_some()
                && power_w.is_some()
                && util.is_some()
            {
                "ok".to_string()
            } else {
                "partial".to_string()
            };

            let gpu_util_pct = util.as_ref().map(|u| u.gpu);
            let mem_util_pct = util.as_ref().map(|u| u.memory);

            let _ = writeln!(
                file,
                "{:.3},{},{},{},{},{:.3},{},{},{}",
                elapsed_s,
                timestamp,
                temp_c.map(|v| v.to_string()).unwrap_or_default(),
                sm_clock_mhz.map(|v| v.to_string()).unwrap_or_default(),
                mem_clock_mhz.map(|v| v.to_string()).unwrap_or_default(),
                power_w.unwrap_or(0.0),
                gpu_util_pct.map(|v| v.to_string()).unwrap_or_default(),
                mem_util_pct.map(|v| v.to_string()).unwrap_or_default(),
                status
            );

            let _ = file.flush();
            thread::sleep(interval);
        }
    });
    Some((stop, handle))
}

/// Spawn a background GPU telemetry sampler.
///
/// Preference order:
/// 1. NVML (no shelling out, works when `nvidia-smi` is blocked)
/// 2. `nvidia-smi` fallback
pub fn spawn_gpu_telemetry_sampler(
    out_path: PathBuf,
    interval: Duration,
) -> Option<(Arc<AtomicBool>, thread::JoinHandle<()>)> {
    #[cfg(feature = "gpu")]
    if let Some(handle) = spawn_gpu_telemetry_sampler_nvml(out_path.clone(), interval) {
        return Some(handle);
    }
    spawn_gpu_telemetry_sampler_nvidia_smi(out_path, interval)
}
