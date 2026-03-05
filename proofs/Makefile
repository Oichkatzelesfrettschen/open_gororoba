# Rocq proof build system for open_gororoba formal verification.
#
# WHY:  Machine-checked proofs elevate claim evidence from "tested at
#       sample points" to "proven for all valid inputs."
# WHAT: Compiles .v files to kernel-checked .vo proof objects.
# HOW:  make all (build), make check (no Admitted), make clean.

ROCQMAKEFILE := rocq makefile
JOBS ?= $(shell nproc 2>/dev/null || echo 1)

Makefile.coq: _CoqProject
	$(ROCQMAKEFILE) -f _CoqProject -o Makefile.coq

all: Makefile.coq
	$(MAKE) -j$(JOBS) -f Makefile.coq all

clean: Makefile.coq
	$(MAKE) -f Makefile.coq clean
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
