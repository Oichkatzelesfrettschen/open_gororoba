from pathlib import Path
import re

p = Path("crates/data_core/src/registry_mirrors")
for f in p.glob("*.rs"):
    content = f.read_text()
    
    # We will just rewrite the `//! ` lines that contain the failing text to `// ` instead of `//! ` or `//! > `.
    # Or just replace `//!` with `//` for ALL lines in these 4 files. They are just mirrors anyway.
    
    files_to_fix = [
        "docs_convos_implement_next_from_1_read_nonuser_lines_cont.rs",
        "docs_research_manga_null_result_smart_goal_2026.rs",
        "experiments_manga_zd_null_paper.rs",
        "experiments_manga_zd_null_retrospective.rs"
    ]
    
    if any(x in str(f) for x in files_to_fix):
        lines = content.split('\n')
        changed = False
        for i in range(len(lines)):
            if lines[i].startswith("//! "):
                # replace `//! ` with `// `
                lines[i] = "// " + lines[i][4:]
                changed = True
            elif lines[i] == "//!":
                lines[i] = "//"
                changed = True
        if changed:
            f.write_text('\n'.join(lines))
