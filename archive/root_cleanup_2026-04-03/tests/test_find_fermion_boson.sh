#!/bin/bash
echo "--- BOSON ---"
grep -rio 'boson' crates/
echo "--- FERMION ---"
grep -rio 'fermion' crates/
