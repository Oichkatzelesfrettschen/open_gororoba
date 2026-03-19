import re
import toml

# --- Define the new entries to be added ---
new_papers = [
    {
        "citation": "**Herbschleb, E. D., et al.** (2019). *Ultra-long coherence times at room temperature in isotopically engineered diamond*. Nature Communications.",
        "notes": ["Achieved T2 = 2.4 ms and T2* = 1.5 ms in P-doped, 12C-enriched diamond."],
        "dois": ["10.1038/s41467-019-11776-8"],
    },
    {
        "citation": "**Bar-Gill, N., et al.** (2013). *Solid-state electronic spin coherence time approaching one second*. Nature Communications.",
        "notes": ["Demonstrated T2 ~ 0.6 s at 77 K using CPMG dynamical decoupling on NV centers."],
        "dois": ["10.1038/ncomms2771"],
    },
    {
        "citation": "**Jarmola, A., et al.** (2012). *Temperature- and Magnetic-Field-Dependent Spin-Lattice Relaxation of Nitrogen-Vacancy Centers in Diamond*. Physical Review Letters.",
        "notes": ["Established the T1 temperature dependence, identifying Orbach (73 meV) and Raman two-phonon processes."],
        "dois": ["10.1103/PhysRevLett.108.197601"],
    },
    {
        "citation": "**Maurer, P. C., et al.** (2012). *Room-Temperature Quantum Bit Memory Exceeding One Second*. Science.",
        "notes": ["Achieved >1s quantum memory at room temperature using 13C nuclear spins near an NV center with QEC."],
        "dois": ["10.1126/science.1220513"],
    },
    {
        "citation": "**Dréau, A., et al.** (2018). *Quantum frequency conversion of single NV-center photons to the telecom C-band*. Physical Review Applied.",
        "notes": ["Canonical demonstration of DFG in PPLN to convert 637 nm NV photons to 1588 nm with 17% efficiency."],
        "dois": ["10.1103/PhysRevApplied.9.064031"],
    },
    {
        "citation": "**Kaczmarek, K. T., et al.** (2018). *High-speed noise-free optical quantum memory*. Physical Review A.",
        "notes": ["Introduced the ORCA protocol in warm cesium vapor, achieving GHz bandwidth and ultra-low noise."],
        "dois": ["10.1103/PhysRevA.97.042306"],
    },
    {
        "citation": "**Pompili, M., et al.** (2021). *Realization of a multi-node quantum network of remote solid-state qubits*. Science.",
        "notes": ["Demonstrated the first three-node quantum network (Alice, Bob, Charlie) with NV centers, enabling entanglement swapping."],
        "dois": ["10.1126/science.abg1919"],
    },
    {
        "citation": "**Hermans, S. L. J., et al.** (2022). *Qubit teleportation between non-neighbouring nodes in a quantum network*. Nature.",
        "notes": ["Realized qubit teleportation across the three-node Delft network, a key primitive for quantum communication."],
        "dois": ["10.1038/s41586-022-04697-y"],
    },
    {
        "citation": "**Hensen, B., et al.** (2015). *Loophole-free Bell inequality violation using electron spins separated by 1.3 kilometres*. Nature.",
        "notes": ["Landmark loophole-free Bell test using NV centers, measuring S = 2.42 ± 0.20."],
        "dois": ["10.1038/nature15759"],
    },
]

# --- Script Logic ---
filepath = "registry/bibliography.toml"

with open(filepath, "r") as f:
    content = f.read()

# Find the highest existing entry ID
entries = re.findall(r'id = "BIB-(\d+)"', content)
if not entries:
    max_id = 0
else:
    max_id = max([int(x) for x in entries])

# Prepare new TOML entries
new_toml_entries = ""
for i, paper in enumerate(new_papers):
    entry_id = max_id + i + 1
    new_toml_entries += f"""
[[entry]]
id = "BIB-{entry_id:04d}"
order_index = {entry_id}
group = "Primary Research & Data Sources"
section = "Quantum Information & Optics"
citation_markdown = "{paper['citation']}"
notes = {repr(paper['notes'])}
urls = []
dois = {repr(paper.get('dois', []))}
source_line = 0
"""

# Append to the file
with open(filepath, "a") as f:
    f.write(new_toml_entries)

# Update the entry_count at the top of the file
new_count = len(entries) + len(new_papers)
content = re.sub(r'(entry_count = )(\d+)', f'\g<1>{new_count}', open(filepath).read(), count=2)
with open(filepath, "w") as f:
    f.write(content)

print(f"Appended {len(new_papers)} new entries to {filepath}. New total count: {new_count}")

