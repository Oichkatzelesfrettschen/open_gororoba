import os
from pathlib import Path
import re

src_root = Path("docs/book/src")
dst_root = Path("crates/verified_core/src/monograph/book")

def convert_dir(src_dir, dst_dir):
    if not dst_dir.exists():
        dst_dir.mkdir(parents=True)
    
    mods = []
    for item in src_dir.iterdir():
        if item.is_dir():
            # Create a mod.rs for this directory
            convert_dir(item, dst_dir / item.name)
            mods.append(f"pub mod {item.name};")
        elif item.suffix == ".md":
            # Convert md to rs
            content = item.read_text()
            lines = content.split('\n')
            rustdoc_lines = []
            for line in lines:
                if line == '':
                    rustdoc_lines.append('//!')
                else:
                    rustdoc_lines.append(f'//! {line}')
            
            # Special case for SUMMARY.md -> summary.rs
            # introduction.md -> introduction.rs
            # glossary.md -> glossary.rs
            stem = item.stem.lower().replace('-', '_')
            target_file = dst_dir / f"{stem}.rs"
            target_file.write_text('\n'.join(rustdoc_lines) + '\n')
            mods.append(f"pub mod {stem};")
            
            # Remove original
            # item.unlink() # We will do git rm later
            
    # Write mod.rs
    mod_file = dst_dir / "mod.rs"
    mods.sort()
    mod_file.write_text('\n'.join(mods) + '\n')

convert_dir(src_root, dst_root)
print("Book conversion complete.")
