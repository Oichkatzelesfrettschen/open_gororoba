//! Detect admission tokens outside Rocq comments and strings.
//! Explicit Axiom declarations remain separate proof assumptions.
//! String doubling and nested comments follow Rocq 9.1 cLexer.ml.

use std::{env, fs, path::Path, process::ExitCode};

fn identifier_character(character: char) -> bool {
    character.is_alphanumeric()
        || matches!(character, '_' | '\'')
        || matches!(character as u32, 0x0300..=0x036f | 0x1ab0..=0x1aff | 0x1dc0..=0x1dff | 0x20d0..=0x20ff | 0xfe20..=0xfe2f)
}

fn scan(source: &str) -> Result<Vec<(usize, String)>, String> {
    let characters: Vec<char> = source.chars().collect();
    let mut position = 0;
    let mut line = 1;
    let mut comments = Vec::new();
    let mut admissions = Vec::new();
    while position < characters.len() {
        let character = characters[position];
        let next = characters.get(position + 1).copied();
        if character == '"' {
            let opening_line = line;
            position += 1;
            loop {
                let Some(&character) = characters.get(position) else {
                    return Err(format!("line {opening_line}: unterminated string"));
                };
                position += 1;
                if character == '"' {
                    if characters.get(position) == Some(&'"') {
                        position += 1;
                    } else {
                        break;
                    }
                } else if character == '\n' {
                    line += 1;
                }
            }
        } else if character == '(' && next == Some('*') {
            comments.push(line);
            position += 2;
        } else if character == '*' && next == Some(')') && !comments.is_empty() {
            comments.pop();
            position += 2;
        } else if comments.is_empty() && identifier_character(character) {
            let start = position;
            position += 1;
            while characters
                .get(position)
                .is_some_and(|&character| identifier_character(character))
            {
                position += 1;
            }
            let token: String = characters[start..position].iter().collect();
            if token == "Admitted" || token == "admit" {
                admissions.push((line, token));
            }
        } else {
            if character == '\n' {
                line += 1;
            }
            position += 1;
        }
    }
    if let Some(opening_line) = comments.first() {
        return Err(format!("line {opening_line}: unterminated comment"));
    }
    Ok(admissions)
}

fn check_path(path: &Path, files: &mut usize, failures: &mut usize) -> Result<(), String> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if metadata.is_dir() {
        let mut children = fs::read_dir(path)
            .map_err(|error| format!("{}: {error}", path.display()))?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("{}: {error}", path.display()))?;
        children.sort();
        for child in children {
            check_path(&child, files, failures)?;
        }
    } else if path.extension().is_some_and(|extension| extension == "v") {
        if !metadata.is_file() {
            return Err(format!(
                "{}: expected a regular Rocq source file",
                path.display()
            ));
        }
        *files += 1;
        let source =
            fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
        match scan(&source) {
            Ok(admissions) => {
                for (line, token) in admissions {
                    eprintln!(
                        "{}:{line}: forbidden admission token {token}",
                        path.display()
                    );
                    *failures += 1;
                }
            }
            Err(error) => {
                eprintln!("{}:{error}", path.display());
                *failures += 1;
            }
        }
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let mut files = 0;
    let mut failures = 0;
    for argument in env::args_os().skip(1) {
        check_path(Path::new(&argument), &mut files, &mut failures)?;
    }
    if files == 0 {
        return Err("expected at least one .v source file".into());
    }
    if failures > 0 {
        return Err(format!(
            "{failures} admission or lexical failure(s) in {files} Rocq sources"
        ));
    }
    println!(
        "PASS: {files} Rocq sources contain zero admission tokens; explicit axioms remain separate assumptions."
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::scan;

    #[test]
    fn comments_strings_and_identifiers_are_not_admissions() {
        let source = r#"(* Zero Admitted. (* admit; *) *)
            Check "Admitted. ""admit;"" (* text *)".
            exact Hadmit. Check fp24_value_admit. Check admit'. Check admité.
            Axiom explicit_assumption : True."#;
        assert!(scan(source).unwrap().is_empty());
    }

    #[test]
    fn strings_inside_comments_protect_comment_delimiters() {
        assert!(scan(r#"(* "*)" "(*" """admit;""" *)"#).unwrap().is_empty());
    }

    #[test]
    fn commands_and_semicolon_tactics_retain_line_numbers() {
        assert_eq!(
            scan("Lemma example : True.\nAdmitted.\nfirst [admit | idtac]; admit; idtac.").unwrap(),
            vec![
                (2, "Admitted".into()),
                (3, "admit".into()),
                (3, "admit".into())
            ]
        );
        assert_eq!(scan("admit(* gap *);Admitted (* gap *).").unwrap().len(), 2);
    }

    #[test]
    fn unterminated_comments_and_strings_fail_closed() {
        for source in [
            "(* outer (* nested *)",
            "\"admit",
            "(* \"unterminated",
            "(* complete *)\n(* open",
        ] {
            assert!(scan(source).is_err(), "{source}");
        }
    }

    #[test]
    fn doubled_quotes_and_backslashes_follow_rocq_strings() {
        assert!(scan(r#""admit "" Admitted""""#).unwrap().is_empty());
        assert_eq!(
            scan(r#""backslash\" admit."#).unwrap(),
            vec![(1, "admit".into())]
        );
    }
}
