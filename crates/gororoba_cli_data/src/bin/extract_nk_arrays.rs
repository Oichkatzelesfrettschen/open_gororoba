use anyhow::Result;
use regex::Regex;
use std::{collections::HashMap, fs, path::Path};

const SRC: &str = "crates/materials_core/src/tabulated_nk.rs";
const OUT: &str = "crates/materials_data/data/nk";

fn main() -> Result<()> {
    let src_path = Path::new(SRC);
    let out_dir = Path::new(OUT);

    if !src_path.exists() {
        eprintln!("Source file {} not found", SRC);
        return Ok(());
    }

    let src = fs::read_to_string(src_path)?;
    let mut arrays: HashMap<String, Vec<f64>> = HashMap::new();

    // Regex: capture const name, size, and the array body (across multiple lines)
    let const_re =
        Regex::new(r"(?s)const\s+([A-Z0-9_]+)\s*:\s*\[f64\s*;\s*(\d+)\]\s*=\s*\[(.*?)\];")?;
    let float_re = Regex::new(r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?")?;
    let comment_re = Regex::new(r"//[^\n]*")?;

    for cap in const_re.captures_iter(&src) {
        let name = cap[1].to_string();
        let size: usize = cap[2].parse()?;
        let body = &cap[3];

        let body_no_comments = comment_re.replace_all(body, "");
        let values: Vec<f64> = float_re
            .find_iter(&body_no_comments)
            .map(|m| m.as_str().parse())
            .collect::<Result<Vec<f64>, _>>()?;

        if values.len() != size {
            eprintln!(
                "WARNING: {} expected {} values, got {}",
                name,
                size,
                values.len()
            );
        }
        arrays.insert(name, values);
    }

    println!(
        "Found {} const arrays: {:?}",
        arrays.len(),
        arrays.keys().collect::<Vec<_>>()
    );

    let groups = [
        ("JC_AU", "au_jc1972"),
        ("JC_AG", "ag_jc1972"),
        ("JC_CU", "cu_jc1972"),
        ("SPLICED_AU", "au_ordal_jc"),
        ("SPLICED_CU", "cu_ordal_jc"),
        ("SPLICED3_AU", "au_ordal_jc_henke"),
        ("SPLICED3_AG", "ag_ordal_jc_henke"),
        ("SPLICED3_CU", "cu_ordal_jc_henke"),
    ];

    fs::create_dir_all(out_dir)?;
    let mut written = 0;

    for (prefix, fname) in groups {
        let ev_key = format!("{}_EV", prefix);
        let n_key = format!("{}_N", prefix);
        let k_key = format!("{}_K", prefix);

        if !arrays.contains_key(&ev_key)
            || !arrays.contains_key(&n_key)
            || !arrays.contains_key(&k_key)
        {
            eprintln!("SKIP {}: missing one or more channels", prefix);
            continue;
        }

        let ev = &arrays[&ev_key];
        let n = &arrays[&n_key];
        let k = &arrays[&k_key];

        if ev.len() != n.len() || n.len() != k.len() {
            eprintln!(
                "ERROR {}: mismatched lengths EV={} N={} K={}",
                prefix,
                ev.len(),
                n.len(),
                k.len()
            );
            continue;
        }

        let out_path = out_dir.join(format!("{}.csv", fname));
        let mut content = String::from("energy_ev,n,k\n");
        for i in 0..ev.len() {
            content.push_str(&format!("{:?},{:?},{:?}\n", ev[i], n[i], k[i]));
        }
        fs::write(&out_path, content)?;

        println!("  Wrote {} ({} rows)", out_path.display(), ev.len());
        written += 1;
    }

    println!("Done: {} CSV files written to {}/", written, OUT);
    Ok(())
}
