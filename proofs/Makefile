# Rocq proof build system for open_gororoba formal verification.
#
# WHY:  Machine-checked proofs elevate claim evidence from "tested at
#       sample points" to "proven for all valid inputs."
# WHAT: Compiles .v files to kernel-checked .vo proof objects.
# HOW:  eval "$(opam env --switch rocq-9.1.1)" && make all
#
# REQUIRES: opam switch rocq-9.1.1 with rocq-core >= 9.1.1.

# Guard: require an opam switch with the pinned Rocq toolchain and the
# matching Flocq compiled library. Compiled Flocq objects carry the OCaml
# runtime identity, so a Rocq binary from one switch cannot consume objects
# built by another switch.
ifndef OPAM_SWITCH_PREFIX
$(error OPAM_SWITCH_PREFIX is not set. Run: eval "$$(opam env --switch rocq-9.1.1)")
endif

# FP24Representable.v imports Flocq.Core. Keep the dependency in the active
# opam switch and map its physical installation to the Flocq logical root.
# FLOCQ_ROOT remains overrideable for reproducible CI or a hermetic build.
FLOCQ_ROOT ?= $(if $(OPAM_SWITCH_PREFIX),$(OPAM_SWITCH_PREFIX)/lib/coq/user-contrib/Flocq,$(shell opam var lib 2>/dev/null)/coq/user-contrib/Flocq)
FLOCQ_CORE_VO := $(FLOCQ_ROOT)/Core/Core.vo
ifeq ($(wildcard $(FLOCQ_CORE_VO)),)
$(error Flocq Core.vo is missing at $(FLOCQ_CORE_VO). Install coq-flocq.4.2.2 in the active Rocq switch.)
endif
ROCQ_EXTERNAL_ARGS := -R "$(FLOCQ_ROOT)" Flocq
# The dependency scanner must not turn installed Flocq sources into build
# targets under the opam prefix. Compiler invocations receive ROCQ_EXTERNAL_ARGS
# and consume the installed .vo objects directly.
ROCQ_PROJECT_FLAGS := -f _RocqProject

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

# GNU make's default behavior leaves a target file in place when its
# recipe exits non-zero, which means a Makefile.rocq written by
# rocq-makefile before the warning grep fires would survive across
# invocations and let `make vos` / `make vok` skip the warnings-as-
# errors gate on the next run. .DELETE_ON_ERROR forces make to delete
# any target whose recipe failed, restoring fail-closed semantics.
.DELETE_ON_ERROR:

Makefile.rocq: $(PROJ_FILE)
	@tmp_log=$$(mktemp); \
	if ! $(ROCQMAKEFILE) -f $(PROJ_FILE) -o Makefile.rocq >"$$tmp_log" 2>&1; then \
	  cat "$$tmp_log"; \
	  rm -f "$$tmp_log" Makefile.rocq Makefile.rocq.conf .Makefile.rocq.d; \
	  exit 1; \
	fi; \
	if grep -n "$(ROCQMAKEFILE_WARN_RE)" "$$tmp_log"; then \
	  cat "$$tmp_log"; \
	  echo "FAIL: rocq makefile emitted a warning; warnings are errors."; \
	  rm -f "$$tmp_log" Makefile.rocq Makefile.rocq.conf .Makefile.rocq.d; \
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
	$(PINNED) $(MAKE) -j$(JOBS) -f Makefile.rocq all \
		COQEXTRAFLAGS="$(ROCQ_EXTERNAL_ARGS)" VDFILE_FLAGS="$(ROCQ_PROJECT_FLAGS)"

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
	$(PINNED) $(MAKE) -j$(JOBS) -f Makefile.rocq vos \
		COQEXTRAFLAGS="$(ROCQ_EXTERNAL_ARGS)" VDFILE_FLAGS="$(ROCQ_PROJECT_FLAGS)"

vok: Makefile.rocq
	$(PINNED) $(MAKE) -j$(JOBS) -f Makefile.rocq vok \
		COQEXTRAFLAGS="$(ROCQ_EXTERNAL_ARGS)" VDFILE_FLAGS="$(ROCQ_PROJECT_FLAGS)"

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

# Verified Gallina -> C emission via CertiRocq (rocq-certirocq 0.9.1+9.1).
# Compiles the proven fp24_value_admit gate to Clight/C; output lands in
# extraction/ (gitignored, reproducible).  Run under the rocq-certirocq switch.
extract-c: all
	@echo "Emitting verified C from Rocq via CertiRocq..."
	@if [ -f extraction/ExtractFP24Admit.v ]; then \
	  rocq compile $(ROCQ_EXTERNAL_ARGS) -R theories OpenGororoba -R verified OpenGororobaVerified \
	    extraction/ExtractFP24Admit.v && \
	  mv -f OpenGororoba.FP24Representable.fp24_value_admit.c \
	        OpenGororoba.FP24Representable.fp24_value_admit.h extraction/ 2>/dev/null && \
	  echo "Emitted extraction/OpenGororoba.FP24Representable.fp24_value_admit.{c,h}"; \
	fi

# Verified Gallina -> C emission of the R2VB DP4 transform kernel.  Pinned
# command: rocq compile of extraction/ExtractR2VBDP4.v under the
# rocq-certirocq switch emits dp4 and dp4_operand_admit plus the Z/positive/
# bool constructor glue; outputs land in extraction/ and their hashes in
# extraction/golden/r2vb_dp4_extraction.sha256.
extract-r2vb-c:
	@echo "Emitting verified R2VB DP4 C from Rocq via CertiRocq..."
	@rocq compile $(ROCQ_EXTERNAL_ARGS) -R theories OpenGororoba -R verified OpenGororobaVerified 	  extraction/ExtractR2VBDP4.v && 	mv -f OpenGororoba.R2VBTransformDP4.dp4.c 	      OpenGororoba.R2VBTransformDP4.dp4.h 	      OpenGororoba.R2VBTransformDP4.dp4_operand_admit.c 	      OpenGororoba.R2VBTransformDP4.dp4_operand_admit.h 	      r2vb_dp4_glue.c r2vb_dp4_glue.h extraction/ && 	echo "Emitted extraction/OpenGororoba.R2VBTransformDP4.*.{c,h} and glue"

# Hash the emitted artifacts and compare against the retained golden hashes.
extract-r2vb-check: extract-r2vb-c
	@cd extraction && sha256sum 	  OpenGororoba.R2VBTransformDP4.dp4.c 	  OpenGororoba.R2VBTransformDP4.dp4_operand_admit.c 	  r2vb_dp4_glue.c > /tmp/r2vb_dp4_extraction.sha256.new && 	diff -u golden/r2vb_dp4_extraction.sha256 /tmp/r2vb_dp4_extraction.sha256.new && 	echo "MATCH: R2VB extraction hashes match golden" || 	(echo "DIFF: R2VB extraction hashes differ from golden"; exit 1)

# Build and run the correspondence harnesses: the extracted dp4 against a
# native integer reference over nine fixtures spanning the B = 181 admission
# boundary, and the extracted gate against its native predicate over five
# known-good and four known-bad rows.
test-r2vb-correspondence:
	@RT="$$(opam var lib)/coq/user-contrib/CertiRocq/Plugin/runtime"; 	gcc -O1 -std=gnu11 -Wno-int-conversion -Wno-implicit-function-declaration 	  -I "$$RT" -o extraction/test_r2vb_dp4 	  extraction/test_r2vb_dp4_correspondence.c 	  extraction/OpenGororoba.R2VBTransformDP4.dp4.c 	  extraction/r2vb_dp4_glue.c "$$RT"/gc_stack.o && 	gcc -O1 -std=gnu11 -Wno-int-conversion -Wno-implicit-function-declaration 	  -I "$$RT" -o extraction/test_r2vb_dp4_admit 	  extraction/test_r2vb_dp4_admit_correspondence.c 	  extraction/OpenGororoba.R2VBTransformDP4.dp4_operand_admit.c 	  extraction/r2vb_dp4_glue.c "$$RT"/gc_stack.o && 	./extraction/test_r2vb_dp4 && ./extraction/test_r2vb_dp4_admit

# Compile-check the emitted C against the CertiRocq runtime headers.  CertiCoq
# puns function pointers into value (long), valid CompCert-C, so int-conversion
# and implicit-declaration are permitted for this translation unit.
extract-c-check: extract-c
	@if [ ! -f extraction/OpenGororoba.FP24Representable.fp24_value_admit.c ]; then \
	  echo "SKIP: no FP24 C extraction emitted"; \
	  exit 0; \
	fi; \
	RT="$$(opam var lib)/coq/user-contrib/CertiRocq/Plugin/runtime"; \
	echo "Compile-checking emitted C against $$RT ..."; \
	gcc -c -std=gnu11 -Wno-int-conversion -Wno-implicit-function-declaration \
	    -I "$$RT" -o /dev/null \
	    extraction/OpenGororoba.FP24Representable.fp24_value_admit.c && \
	echo "MATCH: emitted C compiles clean" || \
	(echo "FAIL: emitted C did not compile"; exit 1)

count:
	@echo "Theory files:  $$(ls theories/*.v 2>/dev/null | wc -l)"
	@echo "Verified files: $$(ls verified/*.v 2>/dev/null | wc -l)"
	@echo "Total .v files: $$(ls theories/*.v verified/*.v 2>/dev/null | wc -l)"
	@echo "Compiled .vo:   $$(ls theories/*.vo verified/*.vo 2>/dev/null | wc -l)"
	@echo "Interfaces .vos: $$(ls theories/*.vos verified/*.vos 2>/dev/null | wc -l)"
	@echo "Checked .vok:    $$(ls theories/*.vok verified/*.vok 2>/dev/null | wc -l)"

.PHONY: all vos vok clean check extract extract-check extract-c extract-c-check count rocq-makefile-check
