# Rocq proof build system for open_gororoba formal verification.
#
# WHY:  Machine-checked proofs elevate claim evidence from "tested at
#       sample points" to "proven for all valid inputs."
# WHAT: Compiles .v files to kernel-checked .vo proof objects.
# HOW:  eval "$(opam env --switch rocq-9.1.1)" && make all
#
# REQUIRES: opam switch rocq-9.1.1 with rocq-core >= 9.1.1.

# Guard: prefer the pinned opam switch, but allow a valid Rocq 9.x binary on
# PATH so non-opam CI/user environments can still run the proof lane.
HAVE_ROCQ := $(shell command -v rocq >/dev/null 2>&1 && echo yes)
ifndef OPAM_SWITCH_PREFIX
ifneq ($(HAVE_ROCQ),yes)
$(error OPAM_SWITCH_PREFIX is not set and rocq is not on PATH. Run: eval "$$(opam env --switch rocq-9.1.1)")
endif
endif

ROCQMAKEFILE := rocq makefile
ROCQMAKEFILE_WARN_RE := [Ww]arning

# Detect PHYSICAL core count (not logical/SMT threads).
# WHY: Rocq proofs peak at ~500MB per file. Running on all SMT threads
#      causes L3 cache thrashing on chips like the 5600X3D (96MB 3D V-Cache
#      shared across SMT siblings). One process per physical core keeps each
#      rocqc in its own cache slice.
PHYS_CORES := $(shell lscpu --all --parse=CORE 2>/dev/null \
                | grep -v '^[[:punct:]]' | sort -u | wc -l)
# Build the CPU affinity list: first logical CPU of each physical core.
PHYS_CPUS := $(shell lscpu --all --parse=CPU,CORE 2>/dev/null \
               | grep -v '^[[:punct:]]' | sort -t, -k2 -n -u | cut -d, -f1 \
               | paste -sd, -)
JOBS ?= $(or $(PHYS_CORES),$(shell nproc 2>/dev/null),1)

# Pin to physical cores when taskset is available; fall back to unpinned.
TASKSET := $(shell command -v taskset 2>/dev/null)
ifdef TASKSET
  PINNED = $(TASKSET) -c $(PHYS_CPUS)
else
  PINNED =
endif

# OCaml 5.4.0 GC tuning: only `o=` (space_overhead) is effective.
# WHY: OCaml 5 ignores s= (initial heap), i= (minor heap size), O= (compaction).
#      o=200 lets major GC tolerate 200% overhead before collecting, reducing
#      pauses during large `ring` term construction (C912 peaks at ~300MB).
export OCAMLRUNPARAM ?= o=200

# Prefer _RocqProject (Rocq 9.x native), fall back to _CoqProject.
PROJ_FILE := $(shell if [ -f _RocqProject ]; then echo _RocqProject; else echo _CoqProject; fi)

Makefile.rocq: $(PROJ_FILE)
	@tmp_log=$$(mktemp); \
	if ! $(ROCQMAKEFILE) -f $(PROJ_FILE) -o Makefile.rocq >"$$tmp_log" 2>&1; then \
	  cat "$$tmp_log"; \
	  rm -f "$$tmp_log"; \
	  exit 1; \
	fi; \
	if grep -n "$(ROCQMAKEFILE_WARN_RE)" "$$tmp_log"; then \
	  cat "$$tmp_log"; \
	  echo "FAIL: rocq makefile emitted a warning; warnings are errors."; \
	  rm -f "$$tmp_log"; \
	  exit 1; \
	fi; \
	cat "$$tmp_log"; \
	rm -f "$$tmp_log"

rocq-makefile-check:
	@tmp_log=$$(mktemp); tmp_make=.Makefile.rocq.check.$$$$; \
	if ! $(ROCQMAKEFILE) -f $(PROJ_FILE) -o "$$tmp_make" >"$$tmp_log" 2>&1; then \
	  cat "$$tmp_log"; \
	  rm -f "$$tmp_log" "$$tmp_make" "$$tmp_make.conf" ".$$tmp_make.d"; \
	  exit 1; \
	fi; \
	if grep -n "$(ROCQMAKEFILE_WARN_RE)" "$$tmp_log"; then \
	  cat "$$tmp_log"; \
	  echo "FAIL: rocq makefile emitted a warning; warnings are errors."; \
	  rm -f "$$tmp_log" "$$tmp_make" "$$tmp_make.conf" ".$$tmp_make.d"; \
	  exit 1; \
	fi; \
	cat "$$tmp_log"; \
	rm -f "$$tmp_log" "$$tmp_make" "$$tmp_make.conf" ".$$tmp_make.d"; \
	echo "OK: rocq makefile completed without warnings."

all: Makefile.rocq
	$(PINNED) $(MAKE) -j$(JOBS) -f Makefile.rocq all

# --- Fast two-phase build pipeline ---
#
# Phase 1: vos (compiled interfaces) -- skips all proof bodies.
#   Checks that definitions, theorem statements, and cross-module
#   imports are well-formed. Seconds, not minutes.
#
# Phase 2: vok (proof checking) -- checks every proof body in parallel.
#   Each file is independent once .vos interfaces exist.
#   Saturates all physical cores.
#
# Workflow:
#   make vos   # instant interface check (iterative dev)
#   make vok   # full proof verification (CI / pre-commit)

vos: Makefile.rocq
	$(PINNED) $(MAKE) -j$(JOBS) -f Makefile.rocq vos

vok: Makefile.rocq
	$(PINNED) $(MAKE) -j$(JOBS) -f Makefile.rocq vok

# --- Standard targets ---

clean: Makefile.rocq
	$(MAKE) -f Makefile.rocq clean
	rm -f Makefile.rocq Makefile.rocq.conf .Makefile.rocq.d
	rm -f Makefile.coq Makefile.coq.conf .Makefile.coq.d

check: all
	@echo "Checking for admitted proofs..."
	@if grep -rn 'Admitted\.\|admit\.' theories/ verified/; then \
	  echo "FAIL: admitted proofs found"; exit 1; \
	else echo "CLEAN: all proofs complete"; fi

extract: all
	@echo "Extracting OCaml code from Rocq..."
	@mkdir -p extraction/golden
	@if [ -f extraction/ExtractQuaternion.v ]; then \
	  rocq compile -Q theories OpenGororoba -Q verified OpenGororobaVerified \
	    extraction/ExtractQuaternion.v; \
	fi

extract-check: extract
	@echo "Checking extraction golden file..."
	@if [ -f extraction/golden/extracted_quat.ml.golden ]; then \
	  diff -u extraction/golden/extracted_quat.ml.golden extraction/extracted_quat.ml && \
	  echo "MATCH: extraction output matches golden file" || \
	  (echo "DIFF: extraction output differs from golden file"; exit 1); \
	else echo "SKIP: no golden file yet"; fi

count:
	@echo "Theory files:  $$(ls theories/*.v 2>/dev/null | wc -l)"
	@echo "Verified files: $$(ls verified/*.v 2>/dev/null | wc -l)"
	@echo "Total .v files: $$(ls theories/*.v verified/*.v 2>/dev/null | wc -l)"
	@echo "Compiled .vo:   $$(ls theories/*.vo verified/*.vo 2>/dev/null | wc -l)"
	@echo "Interfaces .vos: $$(ls theories/*.vos verified/*.vos 2>/dev/null | wc -l)"
	@echo "Checked .vok:    $$(ls theories/*.vok verified/*.vok 2>/dev/null | wc -l)"

.PHONY: all vos vok clean check extract extract-check count rocq-makefile-check
