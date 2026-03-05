# Rocq proof build system for open_gororoba formal verification.
#
# WHY:  Machine-checked proofs elevate claim evidence from "tested at
#       sample points" to "proven for all valid inputs."
# WHAT: Compiles .v files to kernel-checked .vo proof objects.
# HOW:  eval "$(opam env --switch rocq-9.1.1)" && make all
#
# REQUIRES: opam switch rocq-9.1.1 with rocq-core >= 9.1.1.

# Guard: abort if OPAM_SWITCH_PREFIX is unset (opam env not sourced).
ifndef OPAM_SWITCH_PREFIX
$(error OPAM_SWITCH_PREFIX is not set. Run: eval "$$(opam env --switch rocq-9.1.1)")
endif

ROCQMAKEFILE := rocq makefile

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

# OCaml GC tuning: large initial heap + reduced minor GC pressure.
# WHY: cbv + ring on octonion polynomial identities peaks at ~500MB per file.
#      Generous minor heap (32MB) reduces GC pauses on heavy proofs.
export OCAMLRUNPARAM ?= s=1G,i=32M,O=3

# Prefer _RocqProject (Rocq 9.x native), fall back to _CoqProject.
PROJ_FILE := $(shell if [ -f _RocqProject ]; then echo _RocqProject; else echo _CoqProject; fi)

Makefile.rocq: $(PROJ_FILE)
	$(ROCQMAKEFILE) -f $(PROJ_FILE) -o Makefile.rocq

all: Makefile.rocq
	$(PINNED) $(MAKE) -j$(JOBS) -f Makefile.rocq all

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

.PHONY: all clean check extract extract-check count
