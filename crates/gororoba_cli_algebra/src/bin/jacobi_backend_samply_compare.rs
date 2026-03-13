//! Compare sample-weighted line-level Jacobi hotspots from presymbolicated samply profiles.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use csv::Writer;
use flate2::read::GzDecoder;
use serde::Deserialize;

#[derive(Parser, Debug)]
#[command(name = "jacobi-backend-samply-compare")]
#[command(
    about = "Summarize line-level hotspots across reference_f64, x87, and double_double samply captures"
)]
struct Args {
    /// Reference backend samply profile (.json.gz).
    #[arg(
        long,
        default_value = "reports/benchmarks/jacobi_backend_samply_quantized_shell_72_reference_dev.json.gz"
    )]
    reference: PathBuf,

    /// x87 backend samply profile (.json.gz).
    #[arg(
        long,
        default_value = "reports/benchmarks/jacobi_backend_samply_quantized_shell_72_x87_dev.json.gz"
    )]
    x87: PathBuf,

    /// Double-double backend samply profile (.json.gz).
    #[arg(
        long,
        default_value = "reports/benchmarks/jacobi_backend_samply_quantized_shell_72_dd_dev.json.gz"
    )]
    double_double: PathBuf,

    /// Output CSV path.
    #[arg(
        long,
        default_value = "reports/benchmarks/jacobi_backend_samply_compare.csv"
    )]
    output: PathBuf,

    /// Optional Markdown summary path.
    #[arg(long)]
    summary: Option<PathBuf>,

    /// Number of hotspots to retain per backend in the Markdown summary.
    #[arg(long, default_value = "10")]
    top: usize,
}

#[derive(Debug, Clone, Copy)]
enum BackendKind {
    ReferenceF64,
    X87,
    DoubleDouble,
}

impl BackendKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::ReferenceF64 => "reference_f64",
            Self::X87 => "x87",
            Self::DoubleDouble => "double_double",
        }
    }
}

#[derive(Debug)]
struct BackendInput<'a> {
    kind: BackendKind,
    profile: &'a Path,
}

#[derive(Debug, Clone)]
struct HotspotRow {
    backend: BackendKind,
    lib: String,
    function: String,
    file: String,
    line: Option<u64>,
    symbol: String,
    category: HotspotCategory,
    inclusive_samples: u64,
    percent: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HotspotCategory {
    BackendCore,
    RepoShared,
    RepoSupport,
    Dependency,
    RuntimeWrapper,
    Libm,
    Libc,
    Other,
}

impl HotspotCategory {
    fn as_str(self) -> &'static str {
        match self {
            Self::BackendCore => "backend_core",
            Self::RepoShared => "repo_shared",
            Self::RepoSupport => "repo_support",
            Self::Dependency => "dependency",
            Self::RuntimeWrapper => "runtime_wrapper",
            Self::Libm => "libm",
            Self::Libc => "libc",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct HotspotKey {
    lib: String,
    function: String,
    file: String,
    line: Option<u64>,
    symbol: String,
}

#[derive(Debug, Clone)]
struct ResolvedSymbol {
    lib: String,
    function: String,
    file: String,
    line: Option<u64>,
    symbol: String,
}

#[derive(Debug, Deserialize)]
struct SamplyProfile {
    threads: Vec<Thread>,
}

#[derive(Debug, Deserialize)]
struct Thread {
    name: String,
    #[serde(rename = "stringArray")]
    string_array: Vec<String>,
    samples: Samples,
    #[serde(rename = "stackTable")]
    stack_table: StackTable,
    #[serde(rename = "frameTable")]
    frame_table: FrameTable,
    #[serde(rename = "funcTable")]
    func_table: FuncTable,
    #[serde(rename = "resourceTable")]
    resource_table: ResourceTable,
}

#[derive(Debug, Deserialize)]
struct Samples {
    stack: Vec<usize>,
    weight: Vec<u64>,
}

#[derive(Debug, Deserialize)]
struct StackTable {
    prefix: Vec<Option<usize>>,
    frame: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct FrameTable {
    address: Vec<i64>,
    func: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct FuncTable {
    resource: Vec<i64>,
}

#[derive(Debug, Deserialize)]
struct ResourceTable {
    name: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct SamplySidecar {
    #[serde(rename = "string_table")]
    string_table: Vec<String>,
    data: Vec<SidecarLib>,
}

#[derive(Debug, Deserialize)]
struct SidecarLib {
    debug_name: String,
    symbol_table: Vec<SidecarSymbol>,
    known_addresses: Vec<(u64, usize)>,
}

#[derive(Debug, Deserialize)]
struct SidecarSymbol {
    rva: u64,
    size: u64,
    symbol: usize,
    frames: Option<Vec<SidecarFrame>>,
}

#[derive(Debug, Deserialize)]
struct SidecarFrame {
    function: usize,
    file: usize,
    line: Option<u64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let inputs = [
        BackendInput {
            kind: BackendKind::ReferenceF64,
            profile: &args.reference,
        },
        BackendInput {
            kind: BackendKind::X87,
            profile: &args.x87,
        },
        BackendInput {
            kind: BackendKind::DoubleDouble,
            profile: &args.double_double,
        },
    ];

    let mut rows = Vec::new();
    for input in inputs {
        rows.extend(load_backend_rows(input)?);
    }

    rows.sort_by(|lhs, rhs| {
        lhs.backend
            .as_str()
            .cmp(rhs.backend.as_str())
            .then_with(|| rhs.inclusive_samples.cmp(&lhs.inclusive_samples))
            .then_with(|| lhs.lib.cmp(&rhs.lib))
            .then_with(|| lhs.function.cmp(&rhs.function))
            .then_with(|| lhs.line.cmp(&rhs.line))
    });

    write_csv(&args.output, &rows)?;
    if let Some(summary_path) = &args.summary {
        write_summary(summary_path, &rows, args.top)?;
    }

    println!(
        "Wrote {} hotspot rows to {}",
        rows.len(),
        args.output.display()
    );
    if let Some(summary_path) = &args.summary {
        println!(
            "Wrote samply comparison summary to {}",
            summary_path.display()
        );
    }

    Ok(())
}

fn load_backend_rows(input: BackendInput<'_>) -> Result<Vec<HotspotRow>> {
    let profile = read_profile(input.profile)?;
    let sidecar_path = sidecar_path_for(input.profile);
    let sidecar = read_sidecar(&sidecar_path)?;
    let thread = profile
        .threads
        .iter()
        .find(|thread| thread.name == "jacobi-backend-sweep")
        .context("find jacobi-backend-sweep thread")?;

    let total_samples: u64 = thread.samples.weight.iter().copied().sum();
    if total_samples == 0 {
        bail!("profile {} contains zero samples", input.profile.display());
    }

    let counts = inclusive_frame_counts(thread);
    let mut aggregate: HashMap<HotspotKey, u64> = HashMap::new();

    for (frame_idx, sample_count) in counts {
        let resolved = resolve_frame(thread, &sidecar, frame_idx);
        let Some(resolved) = resolved else {
            continue;
        };
        let key = HotspotKey {
            lib: resolved.lib,
            function: resolved.function,
            file: resolved.file,
            line: resolved.line,
            symbol: resolved.symbol,
        };
        *aggregate.entry(key).or_default() += sample_count;
    }

    let mut rows = aggregate
        .into_iter()
        .map(|(key, inclusive_samples)| {
            let category = classify_hotspot(input.kind, &key.lib, &key.function, &key.file);
            HotspotRow {
                backend: input.kind,
                lib: key.lib,
                function: key.function,
                file: key.file,
                line: key.line,
                symbol: key.symbol,
                category,
                inclusive_samples,
                percent: (inclusive_samples as f64 / total_samples as f64) * 100.0,
            }
        })
        .collect::<Vec<_>>();

    rows.sort_by(|lhs, rhs| {
        rhs.inclusive_samples
            .cmp(&lhs.inclusive_samples)
            .then_with(|| lhs.lib.cmp(&rhs.lib))
            .then_with(|| lhs.function.cmp(&rhs.function))
            .then_with(|| lhs.line.cmp(&rhs.line))
    });

    Ok(rows)
}

fn inclusive_frame_counts(thread: &Thread) -> HashMap<usize, u64> {
    let mut counts = HashMap::new();
    for (&stack_index, &weight) in thread.samples.stack.iter().zip(&thread.samples.weight) {
        let mut cursor = Some(stack_index);
        while let Some(stack_row) = cursor {
            let frame_index = thread.stack_table.frame[stack_row];
            *counts.entry(frame_index).or_default() += weight;
            cursor = thread.stack_table.prefix[stack_row];
        }
    }
    counts
}

fn resolve_frame(
    thread: &Thread,
    sidecar: &SamplySidecar,
    frame_idx: usize,
) -> Option<ResolvedSymbol> {
    let address = *thread.frame_table.address.get(frame_idx)?;
    if address < 0 {
        return None;
    }

    let func_idx = *thread.frame_table.func.get(frame_idx)?;
    let resource_idx = *thread.func_table.resource.get(func_idx)?;
    if resource_idx < 0 {
        return None;
    }
    let resource_idx = resource_idx as usize;
    let lib_name = thread
        .resource_name(resource_idx)
        .unwrap_or_else(|| "unknown".to_string());
    let sidecar_lib = sidecar
        .data
        .iter()
        .find(|entry| entry.debug_name == lib_name)?;
    let symbol_index = exact_symbol_index(sidecar_lib, address as u64)
        .or_else(|| range_symbol_index(sidecar_lib, address as u64))?;
    let symbol = sidecar_lib.symbol_table.get(symbol_index)?;
    let symbol_name = sidecar
        .string_table
        .get(symbol.symbol)
        .cloned()
        .unwrap_or_else(|| format!("0x{:x}", address));

    let (function, file, line) = best_frame(symbol.frames.as_deref(), sidecar)
        .map(|frame| {
            (
                sidecar
                    .string_table
                    .get(frame.function)
                    .cloned()
                    .unwrap_or_else(|| symbol_name.clone()),
                sidecar
                    .string_table
                    .get(frame.file)
                    .cloned()
                    .unwrap_or_else(String::new),
                frame.line,
            )
        })
        .unwrap_or_else(|| (symbol_name.clone(), String::new(), None));

    Some(ResolvedSymbol {
        lib: lib_name,
        function,
        file,
        line,
        symbol: symbol_name,
    })
}

fn best_frame<'a>(
    frames: Option<&'a [SidecarFrame]>,
    sidecar: &'a SamplySidecar,
) -> Option<&'a SidecarFrame> {
    let frames = frames?;
    frames
        .iter()
        .max_by_key(|frame| frame_priority(frame, sidecar))
        .or_else(|| frames.first())
}

fn frame_priority(frame: &SidecarFrame, sidecar: &SamplySidecar) -> u8 {
    let file = sidecar
        .string_table
        .get(frame.file)
        .map(String::as_str)
        .unwrap_or("");
    let function = sidecar
        .string_table
        .get(frame.function)
        .map(String::as_str)
        .unwrap_or("");

    if file.contains("/home/eirikr/Github/open_gororoba/") {
        5
    } else if file.contains("/.cache/cargo-home/registry/src/") {
        4
    } else if file.contains("/rust/library/") {
        2
    } else if function.contains("RangeIteratorImpl") || function.contains("Iterator>::next") {
        1
    } else {
        3
    }
}

fn exact_symbol_index(sidecar_lib: &SidecarLib, address: u64) -> Option<usize> {
    sidecar_lib
        .known_addresses
        .iter()
        .find_map(|(known_address, symbol_index)| {
            (*known_address == address).then_some(*symbol_index)
        })
}

fn range_symbol_index(sidecar_lib: &SidecarLib, address: u64) -> Option<usize> {
    sidecar_lib
        .symbol_table
        .iter()
        .enumerate()
        .find_map(|(index, symbol)| {
            let end = symbol.rva.saturating_add(symbol.size.max(1));
            (symbol.rva <= address && address < end).then_some(index)
        })
}

fn read_profile(path: &Path) -> Result<SamplyProfile> {
    let file = fs::File::open(path).with_context(|| format!("open profile {}", path.display()))?;
    let reader = GzDecoder::new(file);
    serde_json::from_reader(reader).with_context(|| format!("parse profile {}", path.display()))
}

fn read_sidecar(path: &Path) -> Result<SamplySidecar> {
    let file = fs::File::open(path).with_context(|| format!("open sidecar {}", path.display()))?;
    serde_json::from_reader(file).with_context(|| format!("parse sidecar {}", path.display()))
}

fn sidecar_path_for(profile_path: &Path) -> PathBuf {
    let profile_name = profile_path
        .file_name()
        .and_then(|name| name.to_str())
        .expect("profile path should have utf-8 file name");
    let sidecar_name = profile_name
        .strip_suffix(".json.gz")
        .map(|stem| format!("{stem}.json.syms.json"))
        .unwrap_or_else(|| format!("{profile_name}.syms.json"));
    profile_path.with_file_name(sidecar_name)
}

fn write_csv(path: &Path, rows: &[HotspotRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }

    let mut writer =
        Writer::from_path(path).with_context(|| format!("create {}", path.display()))?;
    writer.write_record([
        "backend",
        "lib",
        "function",
        "file",
        "line",
        "symbol",
        "category",
        "inclusive_samples",
        "percent",
    ])?;

    for row in rows {
        writer.write_record([
            row.backend.as_str(),
            row.lib.as_str(),
            row.function.as_str(),
            row.file.as_str(),
            &row.line.map(|line| line.to_string()).unwrap_or_default(),
            row.symbol.as_str(),
            row.category.as_str(),
            &row.inclusive_samples.to_string(),
            &format!("{:.4}", row.percent),
        ])?;
    }

    writer.flush().context("flush hotspot CSV")
}

fn write_summary(path: &Path, rows: &[HotspotRow], top: usize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }

    let mut markdown = String::new();
    markdown.push_str("# Jacobi Samply Comparison\n\n");
    markdown.push_str(
        "Inclusive sample-weighted hotspots joined from the samply profile JSON and the presymbolicated sidecar.\n\n",
    );

    for backend in [
        BackendKind::ReferenceF64,
        BackendKind::X87,
        BackendKind::DoubleDouble,
    ] {
        markdown.push_str(&format!("## {}\n\n", backend.as_str()));
        markdown.push_str("### Backend-Core Hotspots\n\n");
        markdown
            .push_str("| lib | function | file:line | category | inclusive_samples | percent |\n");
        markdown.push_str("| --- | --- | --- | --- | ---: | ---: |\n");
        for row in rows
            .iter()
            .filter(|row| row.backend.as_str() == backend.as_str())
            .filter(|row| row.category == HotspotCategory::BackendCore)
            .take(top)
        {
            let file_line = if row.file.is_empty() {
                String::new()
            } else if let Some(line) = row.line {
                format!("{}:{}", row.file, line)
            } else {
                row.file.clone()
            };
            markdown.push_str(&format!(
                "| {} | {} | {} | {} | {} | {:.2}% |\n",
                row.lib,
                row.function,
                file_line,
                row.category.as_str(),
                row.inclusive_samples,
                row.percent
            ));
        }
        markdown.push('\n');

        markdown.push_str("### Shared/Support Hotspots\n\n");
        markdown
            .push_str("| lib | function | file:line | category | inclusive_samples | percent |\n");
        markdown.push_str("| --- | --- | --- | --- | ---: | ---: |\n");
        for row in rows
            .iter()
            .filter(|row| row.backend.as_str() == backend.as_str())
            .filter(|row| {
                row.category == HotspotCategory::RepoShared
                    || row.category == HotspotCategory::RepoSupport
            })
            .take(top)
        {
            let file_line = if row.file.is_empty() {
                String::new()
            } else if let Some(line) = row.line {
                format!("{}:{}", row.file, line)
            } else {
                row.file.clone()
            };
            markdown.push_str(&format!(
                "| {} | {} | {} | {} | {} | {:.2}% |\n",
                row.lib,
                row.function,
                file_line,
                row.category.as_str(),
                row.inclusive_samples,
                row.percent
            ));
        }
        markdown.push('\n');

        markdown.push_str("### External Math/Runtime Hotspots\n\n");
        markdown
            .push_str("| lib | function | file:line | category | inclusive_samples | percent |\n");
        markdown.push_str("| --- | --- | --- | --- | ---: | ---: |\n");
        for row in rows
            .iter()
            .filter(|row| row.backend.as_str() == backend.as_str())
            .filter(|row| row.category != HotspotCategory::BackendCore)
            .filter(|row| row.category != HotspotCategory::RepoShared)
            .filter(|row| row.category != HotspotCategory::RepoSupport)
            .filter(|row| row.category != HotspotCategory::RuntimeWrapper)
            .take(top)
        {
            let file_line = if row.file.is_empty() {
                String::new()
            } else if let Some(line) = row.line {
                format!("{}:{}", row.file, line)
            } else {
                row.file.clone()
            };
            markdown.push_str(&format!(
                "| {} | {} | {} | {} | {} | {:.2}% |\n",
                row.lib,
                row.function,
                file_line,
                row.category.as_str(),
                row.inclusive_samples,
                row.percent
            ));
        }
        markdown.push('\n');
    }

    markdown.push_str("## Synthesis\n\n");
    markdown.push_str("- The comparer now prefers repo-local inline frames over generic `core` iterator frames, which fixes the earlier DD misattribution.\n");
    markdown.push_str("- `reference_f64` should show backend-core lines in `reference_jacobi.rs`, shared/support rows in the sweep/scaffold, dependency rows for `nalgebra`, and external libm trig work.\n");
    markdown.push_str("- `x87` should surface the x87 Jacobi path plus shared `cd_kernel` Givens/ext80 helpers, with unresolved inline-asm pseudo-symbols remaining in the external bucket.\n");
    markdown.push_str("- `double_double` should now show `dd_jacobi.rs` as backend-core even when smaller DD helpers remain inlined.\n");

    fs::write(path, markdown).with_context(|| format!("write {}", path.display()))
}

impl Thread {
    fn resource_name(&self, resource_idx: usize) -> Option<String> {
        let name_idx = *self.resource_table.name.get(resource_idx)?;
        self.string_array.get(name_idx).cloned()
    }
}

fn classify_hotspot(
    backend: BackendKind,
    lib: &str,
    function: &str,
    file: &str,
) -> HotspotCategory {
    if function == "main"
        || function == "start"
        || function.contains("lang_start")
        || function.contains("__rust_begin_short_backtrace")
        || function.contains("call_once")
        || function.contains("_libc_start_")
    {
        return HotspotCategory::RuntimeWrapper;
    }

    if lib == "libm.so.6" {
        return HotspotCategory::Libm;
    }

    if lib == "libc.so.6" {
        return HotspotCategory::Libc;
    }

    if file.contains("/home/eirikr/Github/open_gororoba/crates/gororoba_cli_algebra/src/bin/jacobi_backend_sweep.rs")
    {
        return HotspotCategory::RepoSupport;
    }

    if file.contains("/home/eirikr/Github/open_gororoba/.cache/cargo-home/registry/src/") {
        return HotspotCategory::Dependency;
    }

    if is_backend_core(backend, function, file) {
        return HotspotCategory::BackendCore;
    }

    if file.contains("/home/eirikr/Github/open_gororoba/") {
        return HotspotCategory::RepoShared;
    }

    HotspotCategory::Other
}

fn is_backend_core(backend: BackendKind, function: &str, file: &str) -> bool {
    match backend {
        BackendKind::ReferenceF64 => {
            file.contains("/crates/algebra_analysis/src/reference_jacobi.rs")
                || function.contains("algebra_analysis::reference_jacobi::")
        }
        BackendKind::X87 => {
            file.contains("/crates/algebra_analysis/src/x87_jacobi.rs")
                || file.contains("/crates/cd_kernel/src/x87_jacobi_kernels.rs")
                || file.contains("/crates/cd_kernel/src/x87_transcendentals.rs")
                || file.contains("/crates/cd_kernel/src/x87_ext80.rs")
                || function.contains("algebra_analysis::x87_jacobi::")
                || function.contains("cd_kernel::x87_jacobi_kernels::")
                || function.contains("cd_kernel::x87_transcendentals::")
                || function.contains("cd_kernel::x87_ext80::")
                || function.contains("<cd_kernel::x87_ext80::Ext80>::")
        }
        BackendKind::DoubleDouble => {
            file.contains("/crates/algebra_analysis/src/dd_jacobi.rs")
                || file.contains("/crates/algebra_analysis/src/double_double.rs")
                || function.contains("algebra_analysis::dd_jacobi::")
                || function.contains("algebra_analysis::double_double::")
                || function.contains("<algebra_analysis::double_double::DD>::")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sidecar(strings: &[&str]) -> SamplySidecar {
        SamplySidecar {
            string_table: strings.iter().map(|s| s.to_string()).collect(),
            data: Vec::new(),
        }
    }

    #[test]
    fn best_frame_prefers_repo_local_frame_over_core_inline_frame() {
        let sidecar = test_sidecar(&[
            "core_fn",
            "/rust/library/core/src/iter/range.rs",
            "repo_fn",
            "/home/eirikr/Github/open_gororoba/crates/algebra_analysis/src/dd_jacobi.rs",
        ]);
        let frames = vec![
            SidecarFrame {
                function: 0,
                file: 1,
                line: Some(772),
            },
            SidecarFrame {
                function: 2,
                file: 3,
                line: Some(96),
            },
        ];

        let best = best_frame(Some(&frames), &sidecar).expect("best frame");
        assert_eq!(best.function, 2);
        assert_eq!(best.file, 3);
        assert_eq!(best.line, Some(96));
    }

    #[test]
    fn classify_hotspot_marks_dd_function_names_as_backend_core_even_without_repo_file() {
        let category = classify_hotspot(
            BackendKind::DoubleDouble,
            "jacobi-backend-sweep",
            "algebra_analysis::dd_jacobi::dd_apply_plane_rotation",
            "/home/eirikr/.rustup/toolchains/nightly/lib/rustlib/src/rust/library/core/src/iter/range.rs",
        );
        assert_eq!(category, HotspotCategory::BackendCore);
    }

    #[test]
    fn classify_hotspot_separates_dependencies_from_repo_shared() {
        let category = classify_hotspot(
            BackendKind::ReferenceF64,
            "jacobi-backend-sweep",
            "<nalgebra::linalg::symmetric_eigen::SymmetricEigen<f64, _>>::do_decompose",
            "/home/eirikr/Github/open_gororoba/.cache/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/nalgebra-0.33.2/src/linalg/symmetric_eigen.rs",
        );
        assert_eq!(category, HotspotCategory::Dependency);
    }
}
