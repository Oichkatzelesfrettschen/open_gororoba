import os
from pathlib import Path
import re
import shutil

scandirs = ["docs", "reports", "experiments", "proofs", "curated", "data"]
dst_root = Path("crates/data_core/src/registry_mirrors")

def sanitize_name(name):
    stem = name.lower()
    stem = re.sub(r'[^a-z0-9_]', '_', stem)
    if not stem or not stem[0].isidentifier():
        stem = "md_" + stem
    # Truncate to 100 chars to avoid "File name too long"
    if len(stem) > 100:
        stem = stem[:100]
    return stem

for d in scandirs:
    p = Path(d)
    if not p.exists():
        continue
    
    for md_file in p.glob("**/*.md"):
        # Skip backups and venvs
        if any(x in str(md_file) for x in ["venv", ".git", "node_modules", "lambda_gororoba_backups", "intake"]):
            md_file.unlink()
            print(f"Deleted backup/crap md: {md_file}")
            continue
            
        content = md_file.read_text()
        rel = md_file.relative_to(Path("."))
        flat_name = sanitize_name(str(rel).replace("/", "_").replace(".md", ""))
        
        dst_path = dst_root / f"{flat_name}.rs"
        counter = 1
        while dst_path.exists():
            dst_path = dst_root / f"{flat_name}_{counter}.rs"
            counter += 1
            
        rustdoc_lines = []
        for line in content.split('\n'):
            if line == '':
                rustdoc_lines.append('//!')
            else:
                rustdoc_lines.append(f'//! {line}')
                
        dst_path.write_text('\n'.join(rustdoc_lines) + '\n')
        print(f"Converted {md_file} to {dst_path}")
        md_file.unlink()

# Update mod.rs
data_core_mod = Path("crates/data_core/src/registry_mirrors/mod.rs")
mods = []
for p in Path("crates/data_core/src/registry_mirrors").glob("*.rs"):
    if p.name != "mod.rs":
        mods.append(f"pub mod {p.stem};")
mods.sort()
data_core_mod.write_text('\n'.join(mods) + '\n')

print("Extermination complete.")
