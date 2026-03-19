import toml
import os

filepath = "registry/bibliography.toml"
with open(filepath, "r") as f:
    content = f.read()

import re
entries = re.findall(r'id = "BIB-(\d+)"', content)
max_id = max([int(x) for x in entries])

new_entries = f"""

[[entry]]
id = "BIB-{max_id+1:04d}"
order_index = {max_id+1}
group = "Primary Research & Data Sources"
section = "Materials & Optics"
citation_markdown = "**Patsalas, P., Kalfagiannis, N., & Kassavetis, S.** (2015). *Optical Properties and Plasmonic Performance of Titanium Nitride*. Materials 8(6), 3128-3154. DOI: 10.3390/ma8063128."
notes = ["*Review of TiN optical properties, Drude-Lorentz fitting, screened/unscreened plasma energies, and mean free path.*"]
urls = ["https://pmc.ncbi.nlm.nih.gov/articles/PMC5455719/"]
dois = ["10.3390/ma8063128"]
source_line = 0

[[entry]]
id = "BIB-{max_id+2:04d}"
order_index = {max_id+2}
group = "Primary Research & Data Sources"
section = "Materials & Optics"
citation_markdown = "**Kinsey, N., Ferrera, M., Shalaev, V. M., & Boltasseva, A.** (2015). *Examining nanophotonics for integrated circuitry: plasmonics or silicon photonics?* (or similar related to ZrN). Note: ZrN Drude parameters extracted from D-NB thesis/report."
notes = ["*ZrN plasma frequency and Drude damping parameters (ωp = 7.456 eV, γ = 0.62 eV).*"]
urls = ["https://d-nb.info/1241792763/34"]
dois = []
source_line = 0

[[entry]]
id = "BIB-{max_id+3:04d}"
order_index = {max_id+3}
group = "Primary Research & Data Sources"
section = "Materials & Optics"
citation_markdown = "**Braic, M. et al. / Erbium.nl** (2019). *Tunable plasmonic HfN nanoparticles and arrays*. Nanoscale."
notes = ["*HfN Drude parameters (ωp = 8.19 eV, γ = 0.48 eV, ε∞ = 4.62) from ellipsometry for metal-like films.*"]
urls = ["https://www.erbium.nl/wp-content/uploads/2019/11/Tunable-plasmonic-HfN-nanoparticles-and-arrays-Nanoscale.pdf"]
dois = []
source_line = 0

[[entry]]
id = "BIB-{max_id+4:04d}"
order_index = {max_id+4}
group = "Primary Research & Data Sources"
section = "Materials & Optics"
citation_markdown = "**Naik, G. V., Shalaev, V. M., & Boltasseva, A.** (2013). *Alternative Plasmonic Materials: Beyond Gold and Silver*. Advanced Materials 25(24), 3264-3294. (Includes TCOs like AZO/GZO/ITO)."
notes = ["*TCO plasmonics, carrier density criterion (> 10^20 cm^-3) for NIR metallic behavior.*"]
urls = ["https://arxiv.org/pdf/1211.5988"]
dois = ["10.1002/adma.201205076"]
source_line = 0

[[entry]]
id = "BIB-{max_id+5:04d}"
order_index = {max_id+5}
group = "Primary Research & Data Sources"
section = "Materials & Optics"
citation_markdown = "**Gerfin, T., et al.** (1993/2016). *Optical properties of transparent and infra-red reflecting ITO films in the 0.2 - 50 μm range*."
notes = ["*ITO optical properties, carrier densities up to 8x10^20 cm^-3, effective mass ~0.4 m0.*"]
urls = ["https://www.academia.edu/23283814/Optical_properties_of_transparent_and_infra_red_reflecting_ITO_films_in_the_0_2_50_%CE%BCm_range"]
dois = []
source_line = 0
"""

with open(filepath, "a") as f:
    f.write(new_entries)

# Let's fix the entry_count at the top of the file
new_count = len(entries) + 5
content = re.sub(r'entry_count = \d+', f'entry_count = {new_count}', open(filepath).read(), count=2)
with open(filepath, "w") as f:
    f.write(content)
