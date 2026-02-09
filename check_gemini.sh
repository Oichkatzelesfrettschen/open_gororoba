#!/bin/bash
echo "--- WHICH GEMINI ---" > gemini_debug.txt
which -a gemini >> gemini_debug.txt 2>&1

echo "--- GEMINI VERSION ---" >> gemini_debug.txt
gemini --version >> gemini_debug.txt 2>&1

echo "--- NPM LIST ---" >> gemini_debug.txt
npm list -g @google/gemini-cli >> gemini_debug.txt 2>&1

echo "--- PACMAN QUERY ---" >> gemini_debug.txt
pacman -Q gemini >> gemini_debug.txt 2>&1

echo "--- DONE ---" >> gemini_debug.txt
