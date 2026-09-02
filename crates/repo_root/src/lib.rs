// SPDX-License-Identifier: GPL-2.0-or-later
//
// Runtime discovery of the workspace root.
//
// A binary that locates the repository through `env!("CARGO_MANIFEST_DIR")`
// carries the path of the checkout that compiled it. Cargo keys a
// workspace crate's fingerprint on its content, so a shared build-dir
// (REPO_SHARE_PRIMARY_CACHE=1 in the Makefile) hands a byte-identical
// crate compiled in one worktree to every other worktree, and the copied
// executable then reads the other worktree's registry, or a directory that
// no longer exists. The three strategies below run in the order that puts
// the invoking checkout first and the compile-time path last.

use std::env;
use std::path::{Path, PathBuf};

/// Environment variable naming the root explicitly. The Makefile exports it
/// as `$(CURDIR)` so every gate tool reads the checkout that runs the gate.
pub const ROOT_ENV: &str = "GOROROBA_REPO_ROOT";

/// A directory is the workspace root when its `Cargo.toml` declares a
/// `[workspace]` table and `AGENTS.md` sits beside it. The second marker
/// keeps a vendored or nested workspace from answering for the repository.
pub fn is_workspace_root(dir: &Path) -> bool {
    let manifest = dir.join("Cargo.toml");
    let Ok(text) = std::fs::read_to_string(&manifest) else {
        return false;
    };
    text.lines().any(|line| line.trim() == "[workspace]") && dir.join("AGENTS.md").is_file()
}

/// Walk from `start` upward and return the first workspace root.
pub fn find_root_above(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|dir| is_workspace_root(dir))
        .map(Path::to_path_buf)
}

/// Discover the root from the running process: `GOROROBA_REPO_ROOT` when it
/// names a workspace root, otherwise the nearest workspace root above the
/// current directory.
pub fn discover() -> Option<PathBuf> {
    if let Some(explicit) = env::var_os(ROOT_ENV) {
        let explicit = PathBuf::from(explicit);
        if is_workspace_root(&explicit) {
            return Some(explicit);
        }
    }
    env::current_dir()
        .ok()
        .and_then(|cwd| find_root_above(&cwd))
}

/// Resolve the root, falling back to the compile-time manifest directory
/// only when neither the environment nor the working directory names one.
/// Call through [`resolve!`] so the manifest directory is the caller's.
///
/// # Panics
///
/// When no strategy finds a workspace root. The message names all three,
/// because a silent fallback to a wrong checkout is the failure this crate
/// exists to remove.
pub fn resolve_from(compile_time_manifest_dir: &str) -> PathBuf {
    if let Some(root) = discover() {
        return root;
    }
    if let Some(root) = find_root_above(Path::new(compile_time_manifest_dir)) {
        return root;
    }
    panic!(
        "no workspace root: {ROOT_ENV} is unset or wrong, no ancestor of the current directory \
         holds a [workspace] Cargo.toml beside AGENTS.md, and the compile-time manifest \
         directory {compile_time_manifest_dir} has no such ancestor either"
    );
}

/// The workspace root for the calling crate.
#[macro_export]
macro_rules! resolve {
    () => {
        $crate::resolve_from(env!("CARGO_MANIFEST_DIR"))
    };
}

/// A path below the workspace root for the calling crate.
#[macro_export]
macro_rules! path {
    ($relative:expr) => {
        $crate::resolve_from(env!("CARGO_MANIFEST_DIR")).join($relative)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Tests mutate process-wide state (the environment and the working
    // directory), so they serialize on one lock.
    static PROCESS_STATE: Mutex<()> = Mutex::new(());

    fn make_workspace(dir: &Path) {
        std::fs::create_dir_all(dir.join("crates/tool")).unwrap();
        std::fs::write(dir.join("Cargo.toml"), "[workspace]\nmembers = [\"crates/tool\"]\n")
            .unwrap();
        std::fs::write(dir.join("AGENTS.md"), "# guide\n").unwrap();
        std::fs::write(
            dir.join("crates/tool/Cargo.toml"),
            "[package]\nname = \"tool\"\n",
        )
        .unwrap();
    }

    struct EnvGuard {
        previous: Option<std::ffi::OsString>,
        cwd: PathBuf,
    }

    impl EnvGuard {
        fn new() -> Self {
            Self {
                previous: env::var_os(ROOT_ENV),
                cwd: env::current_dir().unwrap(),
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: the process-state lock serializes every test that
            // touches the environment, and no other thread reads it here.
            unsafe {
                match &self.previous {
                    Some(value) => env::set_var(ROOT_ENV, value),
                    None => env::remove_var(ROOT_ENV),
                }
            }
            env::set_current_dir(&self.cwd).unwrap();
        }
    }

    fn set_root_env(value: Option<&Path>) {
        // SAFETY: see EnvGuard::drop.
        unsafe {
            match value {
                Some(path) => env::set_var(ROOT_ENV, path),
                None => env::remove_var(ROOT_ENV),
            }
        }
    }

    #[test]
    fn a_workspace_manifest_without_agents_md_is_not_a_root() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "[workspace]\n").unwrap();
        assert!(!is_workspace_root(tmp.path()));
        std::fs::write(tmp.path().join("AGENTS.md"), "").unwrap();
        assert!(is_workspace_root(tmp.path()));
    }

    #[test]
    fn a_package_manifest_is_not_a_root() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
        std::fs::write(tmp.path().join("AGENTS.md"), "").unwrap();
        assert!(!is_workspace_root(tmp.path()));
    }

    #[test]
    fn the_working_directory_wins_over_the_compile_time_path() {
        let _lock = PROCESS_STATE.lock().unwrap();
        let _guard = EnvGuard::new();
        let tmp = tempfile::tempdir().unwrap();
        let invoking = tmp.path().join("invoking");
        let compiled_in = tmp.path().join("compiled-in");
        make_workspace(&invoking);
        make_workspace(&compiled_in);
        set_root_env(None);
        env::set_current_dir(invoking.join("crates/tool")).unwrap();
        let root = resolve_from(compiled_in.join("crates/tool").to_str().unwrap());
        assert_eq!(root.canonicalize().unwrap(), invoking.canonicalize().unwrap());
    }

    #[test]
    fn the_environment_wins_over_the_working_directory() {
        let _lock = PROCESS_STATE.lock().unwrap();
        let _guard = EnvGuard::new();
        let tmp = tempfile::tempdir().unwrap();
        let explicit = tmp.path().join("explicit");
        let other = tmp.path().join("other");
        make_workspace(&explicit);
        make_workspace(&other);
        set_root_env(Some(&explicit));
        env::set_current_dir(&other).unwrap();
        assert_eq!(resolve_from("/nonexistent/crates/tool"), explicit);
    }

    #[test]
    fn an_environment_value_that_is_not_a_root_is_ignored() {
        let _lock = PROCESS_STATE.lock().unwrap();
        let _guard = EnvGuard::new();
        let tmp = tempfile::tempdir().unwrap();
        let real = tmp.path().join("real");
        make_workspace(&real);
        set_root_env(Some(&tmp.path().join("missing")));
        env::set_current_dir(real.join("crates")).unwrap();
        assert_eq!(
            resolve_from("/nonexistent").canonicalize().unwrap(),
            real.canonicalize().unwrap()
        );
    }

    #[test]
    fn the_compile_time_path_is_the_last_resort() {
        let _lock = PROCESS_STATE.lock().unwrap();
        let _guard = EnvGuard::new();
        let tmp = tempfile::tempdir().unwrap();
        let compiled_in = tmp.path().join("compiled-in");
        make_workspace(&compiled_in);
        let outside = tmp.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        set_root_env(None);
        env::set_current_dir(&outside).unwrap();
        assert_eq!(
            resolve_from(compiled_in.join("crates/tool").to_str().unwrap()),
            compiled_in
        );
    }

    #[test]
    #[should_panic(expected = "no workspace root")]
    fn no_strategy_panics_with_all_three_named() {
        let _lock = PROCESS_STATE.lock().unwrap();
        let _guard = EnvGuard::new();
        let tmp = tempfile::tempdir().unwrap();
        set_root_env(None);
        env::set_current_dir(tmp.path()).unwrap();
        let _ = resolve_from(tmp.path().join("gone/crates/tool").to_str().unwrap());
    }

    #[test]
    fn the_macros_resolve_this_repository_under_cargo() {
        let _lock = PROCESS_STATE.lock().unwrap();
        let root = crate::resolve!();
        assert!(root.join("AGENTS.md").is_file());
        assert_eq!(crate::path!("Cargo.toml"), root.join("Cargo.toml"));
    }
}
