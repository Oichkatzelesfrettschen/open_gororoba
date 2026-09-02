# Cargo cache roots for the validation gate.
#
# Included by the root Makefile. Standalone use (tests, editors):
#   make -C <checkout> -f mk/cache_roots.mk print-cache-roots
#
# Three roots feed every gate lane:
#   REPO_CARGO_HOME       registry index and crate sources
#   REPO_CARGO_BUILD_DIR  Cargo build-dir: dependency rlibs, fingerprints,
#                         split-debuginfo; the sccache hash key covers
#                         this path (measured: a fresh build-dir missed
#                         82/82 crates the primary had just compiled,
#                         the same path hit 19/19 after a targeted clean)
#   REPO_CARGO_TARGET_DIR Cargo target-dir: uplifted final artifacts,
#                         the validation-tools copies, the cache-check
#                         sentinel and the validate-local lock
#
# Default mode keys all three on $(CURDIR), so a linked worktree compiles
# the dependency graph from nothing in each of the three gate profiles.
# REPO_SHARE_PRIMARY_CACHE=1 shares the home and the build-dir with the
# primary checkout and keeps the target-dir worktree-local. The build-dir
# carries the dependency artifacts, so sharing it alone gives the cache
# hits; the target-dir carries the copied validation executables and
# their stamps, so keeping it local means one worktree never executes
# another worktree's xtask or governance binaries.
#
# The primary checkout is resolved, then verified, from
# `git rev-parse --git-common-dir`. Bare repositories, separate git
# directories and submodules resolve to a metadata directory rather
# than a checkout; those layouts are rejected and require an explicit
# absolute REPO_CACHE_OWNER.

REPO_TMPDIR ?= $(or $(TMPDIR),/tmp)
REPO_SHARE_PRIMARY_CACHE ?=

# resolve_cache_owner prints the primary checkout path, or nothing.
# Each test names one layout it rejects.
define resolve_cache_owner
common=$$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || exit 0; \
[ "$$(git rev-parse --is-bare-repository 2>/dev/null)" = "false" ] || exit 0; \
[ -d "$$common" ] || exit 0; \
[ "$$(basename "$$common")" = ".git" ] || exit 0; \
owner=$$(dirname "$$common"); \
[ -d "$$owner/.git" ] || exit 0; \
top=$$(git -C "$$owner" rev-parse --show-toplevel 2>/dev/null) || exit 0; \
[ "$$top" = "$$owner" ] || exit 0; \
[ -f "$$owner/Cargo.toml" ] || exit 0; \
printf '%s' "$$owner"
endef

ifeq ($(REPO_SHARE_PRIMARY_CACHE),1)
ifeq ($(origin REPO_CACHE_OWNER),undefined)
REPO_CACHE_OWNER := $(shell $(resolve_cache_owner))
endif
ifeq ($(strip $(REPO_CACHE_OWNER)),)
$(error REPO_SHARE_PRIMARY_CACHE=1 needs a primary checkout with a standard <checkout>/.git directory; bare, separate-git-dir and submodule layouts are rejected. Pass an absolute REPO_CACHE_OWNER=<checkout> explicitly.)
endif
ifneq ($(patsubst /%,,$(REPO_CACHE_OWNER)),)
$(error REPO_CACHE_OWNER must be an absolute path, got '$(REPO_CACHE_OWNER)')
endif
ifeq ($(wildcard $(REPO_CACHE_OWNER)/Cargo.toml),)
$(error REPO_CACHE_OWNER '$(REPO_CACHE_OWNER)' holds no Cargo.toml)
endif
REPO_CACHE_MODE := shared
else
REPO_CACHE_OWNER := $(CURDIR)
REPO_CACHE_MODE := local
endif

# The path hash keys the shared build-dir and the tmp cargo root on the
# owner, so every worktree that shares the owner lands in one build-dir.
REPO_PATH_HASH ?= $(shell printf "%s" "$(REPO_CACHE_OWNER)" | sha256sum | cut -c1-16)
REPO_TMP_CARGO_ROOT ?= $(REPO_TMPDIR)/open_gororoba-cargo-build/gate/$(REPO_PATH_HASH)
REPO_CARGO_HOME ?= $(REPO_CACHE_OWNER)/.cache/cargo-home
REPO_CARGO_TARGET_DIR ?= $(CURDIR)/.cache/gate-target
REPO_CARGO_BUILD_DIR ?= $(REPO_CACHE_OWNER)/.cache/gate-cbuild/$(REPO_PATH_HASH)
REPO_CACHE_ROOT := $(REPO_CACHE_OWNER)/.cache
REPO_LOCAL_CACHE_ROOT := $(CURDIR)/.cache

# Worktree-local validation state. The stamps compare Make timestamps,
# so they also carry a content identity of the tool sources: the
# identity file is named by the hash, and a source edit that keeps an
# older mtime still renames it and rebuilds the tools.
VALIDATION_TOOLS_DIR := $(REPO_CARGO_TARGET_DIR)/validation-tools
CACHE_CHECK_SENTINEL := $(VALIDATION_TOOLS_DIR)/cache-check.last
VALIDATION_LOCK := $(VALIDATION_TOOLS_DIR)/validation.lock
VALIDATION_SOURCE_IDENTITY := $(shell find crates xtask -type f \( -name '*.rs' -o -name 'Cargo.toml' \) -print0 2>/dev/null | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | cut -c1-16)
VALIDATION_SOURCE_IDENTITY_FILE := $(VALIDATION_TOOLS_DIR)/source-identity.$(VALIDATION_SOURCE_IDENTITY)

$(VALIDATION_SOURCE_IDENTITY_FILE):
	@mkdir -p $(VALIDATION_TOOLS_DIR)
	@rm -f $(VALIDATION_TOOLS_DIR)/source-identity.*
	@touch $@

# Cache accounting covers every directory a gate lane on this checkout
# can grow: the local target-dir, the owner's target-dir when it differs,
# the owner's build-dir tree, and the residual target/ for cargo doc.
CACHE_ACCOUNT_DIRS := $(sort $(REPO_CARGO_TARGET_DIR) $(REPO_CACHE_ROOT)/gate-target $(REPO_CACHE_ROOT)/gate-cbuild $(CURDIR)/target)
# Age sweeps act on the local target-dir and on the owner's build-dir
# debug trees; both sit under a verified cache root.
CACHE_SWEEP_TARGET_DIR := $(REPO_CARGO_TARGET_DIR)
CACHE_SWEEP_CBUILD_ROOT := $(REPO_CACHE_ROOT)/gate-cbuild

# guarded_rm removes one path after four checks: the path is nonempty
# and absolute, its normalized form contains no parent segment, it sits
# beneath this checkout's .cache, the owner's .cache, or the repo tmp
# root, and a shared owner root is touched from the owner checkout only
# unless REPO_ALLOW_SHARED_CLEAN=1. Usage: $(call guarded_rm,<path>)
REPO_ALLOW_SHARED_CLEAN ?=
define guarded_rm
p="$(1)"; \
if [ -z "$$p" ]; then echo "[cache-guard] REJECT: empty path"; exit 1; fi; \
case "$$p" in /*) ;; *) echo "[cache-guard] REJECT: relative path '$$p'"; exit 1;; esac; \
case "$$p" in *"/../"*|*"/.."|"../"*) echo "[cache-guard] REJECT: parent segment in '$$p'"; exit 1;; esac; \
n=$$(realpath -m -- "$$p"); \
case "$$n" in /|/.|/..) echo "[cache-guard] REJECT: root path '$$p'"; exit 1;; esac; \
case "$$n" in \
  "$(REPO_LOCAL_CACHE_ROOT)/"*|"$(REPO_TMPDIR)/open_gororoba"*) ;; \
  "$(REPO_CACHE_ROOT)/"*) \
    if [ "$(REPO_CACHE_OWNER)" != "$(CURDIR)" ] && [ "$(REPO_ALLOW_SHARED_CLEAN)" != "1" ]; then \
      echo "[cache-guard] REJECT: '$$n' belongs to the shared owner $(REPO_CACHE_OWNER); set REPO_ALLOW_SHARED_CLEAN=1 to remove it from this worktree"; exit 1; \
    fi;; \
  *) echo "[cache-guard] REJECT: '$$n' is outside $(REPO_LOCAL_CACHE_ROOT), $(REPO_CACHE_ROOT) and $(REPO_TMPDIR)/open_gororoba*"; exit 1;; \
esac; \
rm -rf -- "$$n"
endef

.PHONY: print-cache-roots cache-guard-check cache-sweep-plan
print-cache-roots:
	@printf 'REPO_CACHE_MODE=%s\n' '$(REPO_CACHE_MODE)'
	@printf 'REPO_CACHE_OWNER=%s\n' '$(REPO_CACHE_OWNER)'
	@printf 'REPO_PATH_HASH=%s\n' '$(REPO_PATH_HASH)'
	@printf 'REPO_CARGO_HOME=%s\n' '$(REPO_CARGO_HOME)'
	@printf 'REPO_CARGO_TARGET_DIR=%s\n' '$(REPO_CARGO_TARGET_DIR)'
	@printf 'REPO_CARGO_BUILD_DIR=%s\n' '$(REPO_CARGO_BUILD_DIR)'
	@printf 'VALIDATION_TOOLS_DIR=%s\n' '$(VALIDATION_TOOLS_DIR)'
	@printf 'VALIDATION_SOURCE_IDENTITY=%s\n' '$(VALIDATION_SOURCE_IDENTITY)'
	@printf 'CACHE_CHECK_SENTINEL=%s\n' '$(CACHE_CHECK_SENTINEL)'
	@printf 'VALIDATION_LOCK=%s\n' '$(VALIDATION_LOCK)'
	@printf 'CACHE_ACCOUNT_DIRS=%s\n' '$(CACHE_ACCOUNT_DIRS)'
	@printf 'CACHE_SWEEP_TARGET_DIR=%s\n' '$(CACHE_SWEEP_TARGET_DIR)'
	@printf 'CACHE_SWEEP_CBUILD_ROOT=%s\n' '$(CACHE_SWEEP_CBUILD_ROOT)'

# cache-guard-check GUARD_PATH=<path>: dry-run the guard, printing the
# verdict and removing nothing.
GUARD_PATH ?=
cache-guard-check:
	@$(subst rm -rf -- "$$n",printf '[cache-guard] OK: %s\n' "$$n",$(call guarded_rm,$(GUARD_PATH)))

# cache-sweep-plan lists every directory a sweep may remove, each
# passed through the guard, and removes nothing.
cache-sweep-plan:
	@for d in $(CACHE_SWEEP_CBUILD_ROOT)/*/debug; do \
	    [ -d "$$d" ] || continue; \
	    ( $(subst rm -rf -- "$$n",printf '[cache-sweep-plan] candidate %s\n' "$$n",$(call guarded_rm,$$d)) ) || echo "[cache-sweep-plan] skipped $$d"; \
	done; \
	( $(subst rm -rf -- "$$n",printf '[cache-sweep-plan] cargo-sweep target %s\n' "$$n",$(call guarded_rm,$(CACHE_SWEEP_TARGET_DIR))) ) || echo "[cache-sweep-plan] skipped $(CACHE_SWEEP_TARGET_DIR)"
