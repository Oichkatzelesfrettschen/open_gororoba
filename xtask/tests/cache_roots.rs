// SPDX-License-Identifier: GPL-2.0-or-later
//
// Layout tests for mk/cache_roots.mk: the cache-owner resolution behind
// REPO_SHARE_PRIMARY_CACHE=1, the worktree-local validation state, the
// accounting set, and the guarded removal helper. Each test builds a real
// git layout under a temporary directory and drives the fragment through
// `make -f`, so the assertions hold for the shell that the gate runs.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fragment_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask sits one level below the repo root")
        .join("mk/cache_roots.mk")
}

fn tools_available() -> bool {
    let ok = |tool: &str| {
        Command::new(tool)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    };
    if ok("make") && ok("git") {
        true
    } else {
        eprintln!("skip: make and git are both required for cache_roots layout tests");
        false
    }
}

fn git(dir: &Path, args: &[&str]) {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args([
            "-c",
            "user.name=cache-roots-test",
            "-c",
            "user.email=cache-roots-test@example.invalid",
            "-c",
            "protocol.file.allow=always",
            "-c",
            "init.defaultBranch=main",
        ])
        .args(args)
        .output()
        .expect("git runs");
    assert!(
        output.status.success(),
        "git {:?} in {} failed:\n{}",
        args,
        dir.display(),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A minimal checkout: one Cargo.toml so the owner verification accepts it.
fn init_checkout(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("Cargo.toml"), "[workspace]\nmembers = []\n").unwrap();
    git(dir, &["init", "-q"]);
    git(dir, &["add", "Cargo.toml"]);
    git(dir, &["commit", "-q", "-m", "seed"]);
}

struct MakeRun {
    status: i32,
    stdout: String,
    stderr: String,
}

fn run_make(dir: &Path, target: &str, vars: &[(&str, &str)]) -> MakeRun {
    let mut cmd = Command::new("make");
    cmd.arg("-s")
        .arg("-C")
        .arg(dir)
        .arg("-f")
        .arg(fragment_path())
        .arg(target)
        .env_remove("REPO_SHARE_PRIMARY_CACHE")
        .env_remove("REPO_CACHE_OWNER")
        .env_remove("REPO_ALLOW_SHARED_CLEAN")
        .env_remove("MAKEFLAGS")
        .env_remove("MFLAGS");
    for (key, value) in vars {
        cmd.arg(format!("{key}={value}"));
    }
    let output = cmd.output().expect("make runs");
    MakeRun {
        status: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn roots(dir: &Path, vars: &[(&str, &str)]) -> BTreeMap<String, String> {
    let run = run_make(dir, "print-cache-roots", vars);
    assert_eq!(run.status, 0, "print-cache-roots failed:\n{}", run.stderr);
    run.stdout
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn canonical(dir: &Path) -> String {
    dir.canonicalize().unwrap().to_string_lossy().into_owned()
}

const SHARED: &[(&str, &str)] = &[("REPO_SHARE_PRIMARY_CACHE", "1")];

#[test]
fn default_mode_keys_every_root_on_the_checkout() {
    if !tools_available() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let primary = tmp.path().join("primary");
    init_checkout(&primary);
    let p = canonical(&primary);
    let r = roots(&primary, &[]);
    assert_eq!(r["REPO_CACHE_MODE"], "local");
    assert_eq!(r["REPO_CACHE_OWNER"], p);
    assert_eq!(r["REPO_CARGO_HOME"], format!("{p}/.cache/cargo-home"));
    assert_eq!(r["REPO_CARGO_TARGET_DIR"], format!("{p}/.cache/gate-target"));
    assert!(r["REPO_CARGO_BUILD_DIR"].starts_with(&format!("{p}/.cache/gate-cbuild/")));
    assert_eq!(
        r["VALIDATION_TOOLS_DIR"],
        format!("{p}/.cache/gate-target/validation-tools")
    );
}

#[test]
fn standard_layout_shares_with_itself() {
    if !tools_available() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let primary = tmp.path().join("primary");
    init_checkout(&primary);
    let p = canonical(&primary);
    let shared = roots(&primary, SHARED);
    let local = roots(&primary, &[]);
    assert_eq!(shared["REPO_CACHE_MODE"], "shared");
    assert_eq!(shared["REPO_CACHE_OWNER"], p);
    for key in [
        "REPO_PATH_HASH",
        "REPO_CARGO_HOME",
        "REPO_CARGO_TARGET_DIR",
        "REPO_CARGO_BUILD_DIR",
        "VALIDATION_TOOLS_DIR",
        "CACHE_CHECK_SENTINEL",
    ] {
        assert_eq!(shared[key], local[key], "{key} differs between modes on the primary");
    }
}

#[test]
fn linked_worktree_shares_build_dir_and_keeps_validation_state_local() {
    if !tools_available() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let primary = tmp.path().join("primary");
    init_checkout(&primary);
    let wt = tmp.path().join("wt-a");
    git(&primary, &["worktree", "add", "-q", wt.to_str().unwrap(), "-b", "wt-a"]);
    let p = canonical(&primary);
    let w = canonical(&wt);
    let owner = roots(&primary, &[]);
    let r = roots(&wt, SHARED);
    assert_eq!(r["REPO_CACHE_MODE"], "shared");
    assert_eq!(r["REPO_CACHE_OWNER"], p);
    assert_eq!(r["REPO_PATH_HASH"], owner["REPO_PATH_HASH"]);
    assert_eq!(r["REPO_CARGO_HOME"], owner["REPO_CARGO_HOME"]);
    assert_eq!(r["REPO_CARGO_BUILD_DIR"], owner["REPO_CARGO_BUILD_DIR"]);
    assert_eq!(r["REPO_CARGO_TARGET_DIR"], format!("{w}/.cache/gate-target"));
    assert_eq!(
        r["VALIDATION_TOOLS_DIR"],
        format!("{w}/.cache/gate-target/validation-tools")
    );
    assert_eq!(
        r["CACHE_CHECK_SENTINEL"],
        format!("{w}/.cache/gate-target/validation-tools/cache-check.last")
    );
    assert_eq!(
        r["VALIDATION_LOCK"],
        format!("{w}/.cache/gate-target/validation-tools/validation.lock")
    );
}

#[test]
fn two_worktrees_with_different_tool_sources_never_share_validation_state() {
    if !tools_available() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let primary = tmp.path().join("primary");
    init_checkout(&primary);
    let wt_a = tmp.path().join("wt-a");
    let wt_b = tmp.path().join("wt-b");
    git(&primary, &["worktree", "add", "-q", wt_a.to_str().unwrap(), "-b", "wt-a"]);
    git(&primary, &["worktree", "add", "-q", wt_b.to_str().unwrap(), "-b", "wt-b"]);
    // Divergent validation-tool sources in each worktree.
    for (wt, body) in [(&wt_a, "fn main() { println!(\"a\"); }"), (&wt_b, "fn main() { println!(\"b\"); }")] {
        std::fs::create_dir_all(wt.join("xtask/src")).unwrap();
        std::fs::write(wt.join("xtask/src/main.rs"), body).unwrap();
    }
    let a = roots(&wt_a, SHARED);
    let b = roots(&wt_b, SHARED);
    assert_eq!(a["REPO_CARGO_BUILD_DIR"], b["REPO_CARGO_BUILD_DIR"], "the build-dir is the shared part");
    assert_ne!(a["VALIDATION_TOOLS_DIR"], b["VALIDATION_TOOLS_DIR"]);
    assert!(a["VALIDATION_TOOLS_DIR"].starts_with(&canonical(&wt_a)));
    assert!(b["VALIDATION_TOOLS_DIR"].starts_with(&canonical(&wt_b)));
    assert_ne!(a["CACHE_CHECK_SENTINEL"], b["CACHE_CHECK_SENTINEL"]);
    assert_ne!(
        a["VALIDATION_SOURCE_IDENTITY"], b["VALIDATION_SOURCE_IDENTITY"],
        "different tool sources hash to different identities"
    );
    // Editing a source without advancing its mtime still changes the identity.
    std::fs::write(wt_a.join("xtask/src/main.rs"), "fn main() { println!(\"a2\"); }").unwrap();
    let a2 = roots(&wt_a, SHARED);
    assert_ne!(a["VALIDATION_SOURCE_IDENTITY"], a2["VALIDATION_SOURCE_IDENTITY"]);
}

#[test]
fn accounting_from_a_linked_worktree_covers_the_shared_roots() {
    if !tools_available() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let primary = tmp.path().join("primary");
    init_checkout(&primary);
    let wt = tmp.path().join("wt-a");
    git(&primary, &["worktree", "add", "-q", wt.to_str().unwrap(), "-b", "wt-a"]);
    let p = canonical(&primary);
    let w = canonical(&wt);
    let r = roots(&wt, SHARED);
    let dirs: Vec<&str> = r["CACHE_ACCOUNT_DIRS"].split_whitespace().collect();
    for expected in [
        format!("{p}/.cache/gate-target"),
        format!("{p}/.cache/gate-cbuild"),
        format!("{w}/.cache/gate-target"),
        format!("{w}/target"),
    ] {
        assert!(dirs.contains(&expected.as_str()), "missing {expected} in {dirs:?}");
    }
}

#[test]
fn bare_repository_worktree_is_rejected_without_an_explicit_owner() {
    if !tools_available() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let primary = tmp.path().join("primary");
    init_checkout(&primary);
    let bare = tmp.path().join("repos/project.git");
    std::fs::create_dir_all(bare.parent().unwrap()).unwrap();
    git(
        tmp.path(),
        &["clone", "-q", "--bare", primary.to_str().unwrap(), bare.to_str().unwrap()],
    );
    let bwt = tmp.path().join("repos/bare-wt");
    git(&bare, &["worktree", "add", "-q", bwt.to_str().unwrap(), "-b", "bare-wt"]);
    let run = run_make(&bwt, "print-cache-roots", SHARED);
    assert_ne!(run.status, 0, "bare layout must be rejected:\n{}", run.stdout);
    assert!(run.stderr.contains("rejected"), "stderr: {}", run.stderr);
    // The metadata parent must never appear as an owner.
    assert!(!run.stdout.contains("REPO_CACHE_OWNER="));
    // An explicit absolute owner is accepted.
    let p = canonical(&primary);
    let r = roots(&bwt, &[("REPO_SHARE_PRIMARY_CACHE", "1"), ("REPO_CACHE_OWNER", &p)]);
    assert_eq!(r["REPO_CACHE_OWNER"], p);
    assert!(r["REPO_CARGO_BUILD_DIR"].starts_with(&format!("{p}/.cache/gate-cbuild/")));
}

#[test]
fn separate_git_dir_layout_is_rejected() {
    if !tools_available() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let sep = tmp.path().join("sep");
    let gitdir = tmp.path().join("gitdir");
    std::fs::create_dir_all(&sep).unwrap();
    std::fs::write(sep.join("Cargo.toml"), "[workspace]\nmembers = []\n").unwrap();
    git(&sep, &["init", "-q", "--separate-git-dir", gitdir.to_str().unwrap()]);
    git(&sep, &["add", "Cargo.toml"]);
    git(&sep, &["commit", "-q", "-m", "seed"]);
    let run = run_make(&sep, "print-cache-roots", SHARED);
    assert_ne!(run.status, 0, "separate-git-dir layout must be rejected:\n{}", run.stdout);
    assert!(run.stderr.contains("rejected"), "stderr: {}", run.stderr);
}

#[test]
fn submodule_checkout_is_rejected() {
    if !tools_available() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let inner = tmp.path().join("inner");
    init_checkout(&inner);
    let outer = tmp.path().join("outer");
    init_checkout(&outer);
    git(&outer, &["submodule", "add", "-q", inner.to_str().unwrap(), "sub"]);
    let sub = outer.join("sub");
    let run = run_make(&sub, "print-cache-roots", SHARED);
    assert_ne!(run.status, 0, "submodule layout must be rejected:\n{}", run.stdout);
    assert!(run.stderr.contains("rejected"), "stderr: {}", run.stderr);
}

#[test]
fn explicit_owner_must_be_absolute_and_hold_a_manifest() {
    if !tools_available() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let primary = tmp.path().join("primary");
    init_checkout(&primary);
    let relative = run_make(
        &primary,
        "print-cache-roots",
        &[("REPO_SHARE_PRIMARY_CACHE", "1"), ("REPO_CACHE_OWNER", "relative/path")],
    );
    assert_ne!(relative.status, 0);
    assert!(relative.stderr.contains("absolute"), "stderr: {}", relative.stderr);
    let empty = tmp.path().join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let no_manifest = run_make(
        &primary,
        "print-cache-roots",
        &[("REPO_SHARE_PRIMARY_CACHE", "1"), ("REPO_CACHE_OWNER", empty.to_str().unwrap())],
    );
    assert_ne!(no_manifest.status, 0);
    assert!(no_manifest.stderr.contains("Cargo.toml"), "stderr: {}", no_manifest.stderr);
}

#[test]
fn guard_rejects_unsafe_paths_and_accepts_local_cache_paths() {
    if !tools_available() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let primary = tmp.path().join("primary");
    init_checkout(&primary);
    let p = canonical(&primary);
    let sibling = canonical(tmp.path());
    let rejected = [
        String::new(),
        "relative/.cache/x".to_string(),
        "/".to_string(),
        format!("{p}/.cache/../.cache/gate-target"),
        format!("{p}/.cache/../../elsewhere"),
        sibling.clone(),
        format!("{p}/Cargo.toml"),
        format!("{p}/.cache"),
    ];
    for path in &rejected {
        let run = run_make(&primary, "cache-guard-check", &[("GUARD_PATH", path)]);
        assert_ne!(run.status, 0, "guard accepted {path:?}:\n{}", run.stdout);
        assert!(run.stdout.contains("REJECT"), "path {path:?} stdout: {}", run.stdout);
    }
    let accepted = run_make(
        &primary,
        "cache-guard-check",
        &[("GUARD_PATH", &format!("{p}/.cache/gate-target"))],
    );
    assert_eq!(accepted.status, 0, "{}", accepted.stdout);
    assert!(accepted.stdout.contains("[cache-guard] OK"));
}

#[test]
fn guard_refuses_the_shared_owner_tree_from_a_worktree_unless_allowed() {
    if !tools_available() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let primary = tmp.path().join("primary");
    init_checkout(&primary);
    let wt = tmp.path().join("wt-a");
    git(&primary, &["worktree", "add", "-q", wt.to_str().unwrap(), "-b", "wt-a"]);
    let p = canonical(&primary);
    let owner_dir = format!("{p}/.cache/gate-cbuild/abc/debug");
    let refused = run_make(
        &wt,
        "cache-guard-check",
        &[("REPO_SHARE_PRIMARY_CACHE", "1"), ("GUARD_PATH", &owner_dir)],
    );
    assert_ne!(refused.status, 0);
    assert!(refused.stdout.contains("shared owner"), "{}", refused.stdout);
    let allowed = run_make(
        &wt,
        "cache-guard-check",
        &[
            ("REPO_SHARE_PRIMARY_CACHE", "1"),
            ("REPO_ALLOW_SHARED_CLEAN", "1"),
            ("GUARD_PATH", &owner_dir),
        ],
    );
    assert_eq!(allowed.status, 0, "{}", allowed.stdout);
}

#[test]
fn sweep_plan_names_only_the_shared_cache_and_never_a_sibling() {
    if !tools_available() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let primary = tmp.path().join("primary");
    init_checkout(&primary);
    let wt = tmp.path().join("wt-a");
    git(&primary, &["worktree", "add", "-q", wt.to_str().unwrap(), "-b", "wt-a"]);
    let p = canonical(&primary);
    let owner = roots(&primary, &[]);
    let hash = &owner["REPO_PATH_HASH"];
    let debug = format!("{p}/.cache/gate-cbuild/{hash}/debug");
    std::fs::create_dir_all(&debug).unwrap();
    std::fs::write(Path::new(&debug).join("artifact.o"), b"x").unwrap();
    // Decoys a careless glob could select.
    std::fs::create_dir_all(format!("{p}/.cache/other/debug")).unwrap();
    std::fs::create_dir_all(tmp.path().join("sibling/debug")).unwrap();

    let from_owner = run_make(&primary, "cache-sweep-plan", &[]);
    assert_eq!(from_owner.status, 0, "{}", from_owner.stderr);
    assert!(from_owner.stdout.contains(&format!("candidate {debug}")), "{}", from_owner.stdout);
    assert!(
        from_owner.stdout.contains(&format!("cargo-sweep target {p}/.cache/gate-target")),
        "{}",
        from_owner.stdout
    );

    let from_wt = run_make(&wt, "cache-sweep-plan", SHARED);
    assert_eq!(from_wt.status, 0, "{}", from_wt.stderr);
    assert!(from_wt.stdout.contains(&format!("skipped {debug}")), "{}", from_wt.stdout);
    assert!(from_wt.stdout.contains("shared owner"), "{}", from_wt.stdout);

    for run in [&from_owner, &from_wt] {
        assert!(!run.stdout.contains("/.cache/other"), "decoy selected: {}", run.stdout);
        assert!(!run.stdout.contains("/sibling"), "sibling selected: {}", run.stdout);
    }
    assert!(Path::new(&debug).join("artifact.o").exists(), "the plan removed a file");
}

/// Two worktrees of one owner hold identical tool sources, so the source hash
/// alone cannot separate them. The composite identity folds in the resolved
/// checkout path, so each worktree names its own stamp file.
#[test]
fn validation_identity_differs_across_worktrees_sharing_one_owner() {
    if !tools_available() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let primary = tmp.path().join("primary");
    init_checkout(&primary);
    let wt_a = tmp.path().join("wt-a");
    let wt_b = tmp.path().join("wt-b");
    git(&primary, &["worktree", "add", "-q", wt_a.to_str().unwrap(), "-b", "wt-a"]);
    git(&primary, &["worktree", "add", "-q", wt_b.to_str().unwrap(), "-b", "wt-b"]);
    let a = roots(&wt_a, SHARED);
    let b = roots(&wt_b, SHARED);
    assert_eq!(
        a["VALIDATION_SOURCE_IDENTITY"], b["VALIDATION_SOURCE_IDENTITY"],
        "identical tool sources hash the same, which is why the source hash cannot separate the worktrees"
    );
    assert_eq!(a["REPO_CACHE_OWNER"], b["REPO_CACHE_OWNER"]);
    assert_eq!(a["VALIDATION_CURDIR_REAL"], canonical(&wt_a));
    assert_ne!(
        a["VALIDATION_TOOL_IDENTITY"], b["VALIDATION_TOOL_IDENTITY"],
        "the composite identity must separate two worktrees of one owner"
    );
    assert!(a["VALIDATION_TOOL_IDENTITY_FILE"].ends_with(&a["VALIDATION_TOOL_IDENTITY"]));
    assert!(b["VALIDATION_TOOL_IDENTITY_FILE"].ends_with(&b["VALIDATION_TOOL_IDENTITY"]));
}

/// The incident: a tool compiled while one worktree existed is staged, that
/// worktree is removed, and the tool later resolves a path that is gone. The
/// stamp identity forces the rebuild decision; the byte scan proves the copy
/// is the wrong one.
#[test]
fn stale_worktree_tool_copy_is_rejected_and_rebuilt() {
    if !tools_available() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let worktrees_root = tmp.path().join("worktrees");
    std::fs::create_dir_all(&worktrees_root).unwrap();
    let primary = tmp.path().join("primary");
    init_checkout(&primary);
    let wt_a = worktrees_root.join("gone");
    let wt_b = worktrees_root.join("live");
    git(&primary, &["worktree", "add", "-q", wt_a.to_str().unwrap(), "-b", "gone"]);
    git(&primary, &["worktree", "add", "-q", wt_b.to_str().unwrap(), "-b", "live"]);
    let a = roots(&wt_a, SHARED);
    let b = roots(&wt_b, SHARED);
    let gone_path = canonical(&wt_a);

    // Compile in wt-a: an identity stamp plus a binary whose bytes embed the
    // compiling checkout, the way debuginfo and panic locations do.
    let a_tools = PathBuf::from(&a["VALIDATION_TOOLS_DIR"]);
    std::fs::create_dir_all(&a_tools).unwrap();
    let fake = |dir: &Path, embedded: &str| {
        let mut bytes = b"\x7fELF\x00".to_vec();
        bytes.extend_from_slice(format!("{embedded}/registry/claims.toml").as_bytes());
        bytes.push(0);
        std::fs::write(dir.join("provenance"), &bytes).unwrap();
    };
    fake(&a_tools, &gone_path);
    std::fs::write(&a["VALIDATION_TOOL_IDENTITY_FILE"], "staged\n").unwrap();

    // The shared build-dir hands the same artifact to wt-b.
    let b_tools = PathBuf::from(&b["VALIDATION_TOOLS_DIR"]);
    std::fs::create_dir_all(&b_tools).unwrap();
    std::fs::copy(a_tools.join("provenance"), b_tools.join("provenance")).unwrap();
    std::fs::copy(
        &a["VALIDATION_TOOL_IDENTITY_FILE"],
        b_tools.join(Path::new(&a["VALIDATION_TOOL_IDENTITY_FILE"]).file_name().unwrap()),
    )
    .unwrap();

    git(&primary, &["worktree", "remove", "--force", wt_a.to_str().unwrap()]);
    assert!(!Path::new(&gone_path).exists());

    // Resolution from wt-b: the stamp written for wt-a does not satisfy it,
    // and the rule that creates wt-b's stamp discards wt-a's.
    let run = run_make(&wt_b, &b["VALIDATION_TOOL_IDENTITY_FILE"], SHARED);
    assert_eq!(run.status, 0, "{}", run.stderr);
    assert!(Path::new(&b["VALIDATION_TOOL_IDENTITY_FILE"]).exists());
    assert!(
        !b_tools
            .join(Path::new(&a["VALIDATION_TOOL_IDENTITY_FILE"]).file_name().unwrap())
            .exists(),
        "the foreign identity file survived"
    );
    let stamp = std::fs::read_to_string(&b["VALIDATION_TOOL_IDENTITY_FILE"]).unwrap();
    for key in ["tool_identity=", "source_identity=", "curdir_real=", "owner="] {
        assert!(stamp.contains(key), "stamp missing {key}: {stamp}");
    }
    assert!(stamp.contains(&format!("curdir_real={}", canonical(&wt_b))));

    // The staged copy still holds the vanished path, and the scan rejects it.
    let hits = repo_utilities::validation_tool_paths::scan_tools_dir(
        &b_tools,
        &worktrees_root,
        Path::new(&canonical(&wt_b)),
    )
    .unwrap();
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert_eq!(hits[0].embedded, format!("{gone_path}/registry/claims.toml"));
}

/// Negative control: a binary that embeds only the running checkout's own path
/// is legitimate and must pass.
#[test]
fn path_scan_keeps_a_tool_that_names_only_the_live_worktree() {
    let tmp = tempfile::tempdir().unwrap();
    let worktrees_root = tmp.path().join("worktrees");
    let live = worktrees_root.join("live");
    let tools = live.join(".cache/gate-target/validation-tools");
    std::fs::create_dir_all(&tools).unwrap();
    let mut bytes = b"\x7fELF\x00".to_vec();
    bytes.extend_from_slice(format!("{}/crates/xtask/src/main.rs", live.display()).as_bytes());
    bytes.push(0);
    std::fs::write(tools.join("xtask"), &bytes).unwrap();
    let hits =
        repo_utilities::validation_tool_paths::scan_tools_dir(&tools, &worktrees_root, &live)
            .unwrap();
    assert!(hits.is_empty(), "{hits:?}");
}
