use anyhow::Result;
use clap::Parser;
use quick_xml::{events::Event, reader::Reader};
use std::io::{self, Read};

#[derive(Parser, Debug)]
#[command(
    name = "cpd-report",
    about = "PMD CPD XML report formatter (Rust replacement for cpd_report.py)"
)]
struct Cli {
    #[arg(long)]
    strict: bool,
    #[arg(long, default_value_t = 20)]
    top: usize,
}

#[derive(Debug, Default)]
struct Duplication {
    lines: usize,
    tokens: usize,
    files: Vec<CpdFile>,
    codefragment: String,
}

#[derive(Debug, Default)]
struct CpdFile {
    path: String,
    line: usize,
}

fn short_path(path: &str) -> &str {
    for marker in &["crates/", "src/"] {
        if let Some(idx) = path.find(marker) {
            return &path[idx..];
        }
    }
    path
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;

    if input.trim().is_empty() {
        eprintln!("cpd-report: no XML received (pmd produced no output).");
        std::process::exit(2);
    }

    let mut reader = Reader::from_str(&input);
    let mut dups = Vec::new();
    let mut current_dup: Option<Duplication> = None;
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => match e.name().as_ref() {
                b"duplication" => {
                    let mut dup = Duplication::default();
                    for attr in e.attributes().flatten() {
                        match attr.key.as_ref() {
                            b"lines" => dup.lines = std::str::from_utf8(&attr.value)?.parse()?,
                            b"tokens" => dup.tokens = std::str::from_utf8(&attr.value)?.parse()?,
                            _ => {}
                        }
                    }
                    current_dup = Some(dup);
                }
                b"file" => {
                    if let Some(ref mut dup) = current_dup {
                        let mut file = CpdFile::default();
                        for attr in e.attributes().flatten() {
                            match attr.key.as_ref() {
                                b"path" => {
                                    file.path = std::str::from_utf8(&attr.value)?.to_string()
                                }
                                b"line" => file.line = std::str::from_utf8(&attr.value)?.parse()?,
                                _ => {}
                            }
                        }
                        dup.files.push(file);
                    }
                }
                b"codefragment" => {
                    if let Some(ref mut dup) = current_dup {
                        dup.codefragment = reader.read_text(e.name())?.to_string();
                    }
                }
                _ => {}
            },
            Ok(Event::End(e)) if e.name().as_ref() == b"duplication" => {
                if let Some(dup) = current_dup.take() {
                    dups.push(dup);
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                eprintln!("cpd-report: XML parse error: {}", e);
                std::process::exit(2);
            }
            _ => {}
        }
        buf.clear();
    }

    let total_lines: usize = dups.iter().map(|d| d.lines).sum();

    println!("CPD audit  --minimum-tokens 42  Rust");
    println!("Clusters : {}", dups.len());
    println!("Dup lines: {}", total_lines);

    if dups.is_empty() {
        println!("No duplication found.");
        return Ok(());
    }

    dups.sort_by_key(|d| std::cmp::Reverse(d.lines));
    let show = &dups[..std::cmp::min(dups.len(), cli.top)];

    println!();
    println!("{: <4}  {: >6}  {: >7}  Locations", "#", "Lines", "Tokens");
    println!("{}", "-".repeat(72));

    for (i, dup) in show.iter().enumerate() {
        let locs: Vec<String> = dup
            .files
            .iter()
            .map(|f| format!("{}:{}", short_path(&f.path), f.line))
            .collect();
        let loc_str = locs.join("  /  ");
        println!(
            "{: <4}  {: >6}  {: >7}  {}",
            i + 1,
            dup.lines,
            dup.tokens,
            loc_str
        );
        let preview = dup.codefragment.replace('\n', " ");
        let preview = if preview.len() > 90 {
            format!("{}...", &preview[..87])
        } else {
            preview
        };
        println!("       {:?}", preview);
    }

    if dups.len() > cli.top {
        println!("  ... {} more clusters not shown.", dups.len() - cli.top);
    }

    if cli.strict && !dups.is_empty() {
        eprintln!(
            "\ncpd-audit-strict: FAIL -- {} cluster(s) found.",
            dups.len()
        );
        std::process::exit(1);
    }

    Ok(())
}
