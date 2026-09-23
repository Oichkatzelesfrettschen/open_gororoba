use std::{env, path::Path};

fn main() {
    let mut arguments = env::args_os().skip(1);
    let root = arguments.next().unwrap_or_else(|| ".".into());
    let input = arguments
        .next()
        .unwrap_or_else(|| "registry/book_docs.toml".into());
    if arguments.next().is_some() {
        eprintln!("usage: book-docs-emit [repository-root] [registry-path]");
        std::process::exit(2);
    }
    if let Err(error) = book_docs::emit_legacy(Path::new(&root), Path::new(&input), false) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
