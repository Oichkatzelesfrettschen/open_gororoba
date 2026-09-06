//! Independent raw-ordinal check of causal support after timestamp quarantine.

use std::{env, fs, path::Path};

fn integer(record: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(record[offset..offset + 8].try_into().unwrap())
}

fn verify_record(record: &[u8], file_id: usize, markers: &[bool]) -> Result<(), String> {
    if record.len() != 59 || usize::from(u16::from_le_bytes([record[0], record[1]])) != file_id {
        return Err("record width or file identity".into());
    }
    let decision = usize::try_from(integer(record, 2)).map_err(|error| error.to_string())?;
    let first = usize::try_from(integer(record, 10)).map_err(|error| error.to_string())?;
    let feature_first = integer(record, 18);
    let last = usize::try_from(integer(record, 26)).map_err(|error| error.to_string())?;
    let count = integer(record, 34);
    if first > last
        || last >= decision
        || decision >= markers.len()
        || count != 1031
        || last - first + 1 != 1031
        || feature_first.checked_add(5) != Some(last as u64)
        || record[58] > 1
    {
        return Err("support index, warmup, or label invariant".into());
    }
    if markers[first..=decision].iter().any(|marker| *marker) {
        return Err("quarantined row reaches history or decision".into());
    }
    let decision_time = i64::from_le_bytes(record[42..50].try_into().unwrap());
    let latest_time = i64::from_le_bytes(record[50..58].try_into().unwrap());
    if latest_time >= decision_time {
        return Err("timestamp causality".into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = env::args().collect();
    if arguments.len() != 3 {
        return Err("usage: support-audit DERIVED_INPUT_ROOT RESULT_ROOT".into());
    }
    let root = Path::new(&arguments[1]);
    let results = Path::new(&arguments[2]);
    let map = fs::read_to_string(root.join("file-map.csv"))?;
    let mut lines = map.lines();
    if lines.next() != Some("file_id,path") {
        return Err("file map header".into());
    }
    let mut file_count = 0;
    println!("file_id\tdate\traw_rows\tquarantined_rows\tadmitted_decisions\tpositive_decisions");
    for (file_id, line) in lines.enumerate() {
        let (identifier, relative) = line.split_once(',').ok_or("file map row")?;
        if identifier.parse::<usize>()? != file_id {
            return Err("file map order".into());
        }
        let raw = fs::read(root.join(relative))?;
        let markers: Vec<_> = raw
            .split_inclusive(|byte| *byte == b'\n')
            .map(|row| row.starts_with(b"AMENDMENT_QUARANTINED,"))
            .collect();
        let support = fs::read(results.join(format!("supports/file-{file_id:04}.bin")))?;
        if support.len() % 59 != 0 {
            return Err("partial support record".into());
        }
        let mut positives = 0;
        let mut preceding_decision = None;
        for record in support.chunks_exact(59) {
            verify_record(record, file_id, &markers)?;
            let decision = integer(record, 2);
            if preceding_decision.is_some_and(|previous| previous >= decision) {
                return Err("decision ordinal order".into());
            }
            preceding_decision = Some(decision);
            positives += usize::from(record[58]);
        }
        println!(
            "{file_id}\t{}\t{}\t{}\t{}\t{positives}",
            relative.split('/').next().ok_or("date path")?,
            markers.len(),
            markers.iter().filter(|marker| **marker).count(),
            support.len() / 59
        );
        file_count += 1;
    }
    if file_count != 166 || fs::read_dir(results.join("supports"))?.count() != 166 {
        return Err("support file denominator".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_contract_rejects_contamination_and_short_warmup() {
        let mut record = [0_u8; 59];
        for (offset, value) in [
            (2, 1031_u64),
            (10, 0),
            (18, 1025),
            (26, 1030),
            (34, 1031),
            (42, 2000),
            (50, 1999),
        ] {
            record[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        }
        let mut markers = vec![false; 1032];
        assert!(verify_record(&record, 0, &markers).is_ok());
        for ordinal in [0, 1025, 1030, 1031] {
            markers[ordinal] = true;
            assert!(verify_record(&record, 0, &markers).is_err());
            markers[ordinal] = false;
        }
        record[34..42].copy_from_slice(&1030_u64.to_le_bytes());
        assert!(verify_record(&record, 0, &markers).is_err());
    }
}
