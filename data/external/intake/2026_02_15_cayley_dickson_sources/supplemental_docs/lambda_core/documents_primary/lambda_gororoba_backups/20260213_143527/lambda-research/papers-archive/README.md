# Lambda Calculus Papers Archive

A systematic collection of academic papers, historical documents, and research materials for lambda calculus and type theory, organized for research and educational purposes.

## Archive Structure

### Organization Principles

1. **By Type System / Calculus Variant**: Primary organization follows the theoretical hierarchy
2. **By Era**: Historical (pre-1980), Classical (1980-2000), Modern (2000-2020), Recent (2020+)
3. **By Access Type**: Open access, author copies, historical public domain
4. **By Document Type**: Papers, books, theses, surveys, tutorials

### Directory Structure

```
papers-archive/
├── README.md                          # This file
├── CITATION_INDEX.md                  # Master citation index
├── metadata/                          # Metadata and indexing files
│   ├── bibliography.bib              # Master BibTeX database
│   ├── author_index.json             # Author cross-reference
│   ├── topic_tags.json               # Topic classification system
│   └── download_sources.json         # URL tracking and status
├── scripts/                           # Automation and management tools
│   ├── download_papers.py            # Automated paper retrieval
│   ├── update_metadata.py            # Metadata synchronization
│   ├── generate_index.py             # Index generation
│   └── verify_access.py              # Link validation
├── historical/                       # Foundational papers (pre-1980)
│   ├── church-lambda-calculus/        # Alonzo Church's original works
│   ├── curry-combinatory-logic/       # Curry-Feys combinatory logic
│   ├── scott-domain-theory/           # Dana Scott's domain theory
│   └── README.md                      # Historical collection overview
├── classical/                        # Established theory (1980-2000)
│   ├── girard-system-f/              # Girard's System F and Linear Logic
│   ├── martin-lof-type-theory/        # Martin-Löf constructive type theory
│   ├── polymorphism-parametricity/    # Reynolds, Wadler parametricity
│   └── README.md                      # Classical period overview
├── modern/                           # Developments (2000-2020)
│   ├── dependent-types/              # Modern dependent type systems
│   ├── homotopy-type-theory/          # HoTT and univalent foundations
│   ├── linear-session-types/          # Linear types and session types
│   └── README.md                      # Modern developments overview
├── recent/                           # Current research (2020+)
│   ├── quantum-calculi/              # Quantum lambda calculi
│   ├── probabilistic-types/           # Probabilistic and gradual typing
│   ├── categorical-semantics/         # Category theory applications
│   └── README.md                      # Recent research overview
└── surveys/                          # Survey papers and tutorials
    ├── historical-surveys/           # Historical perspectives
    ├── tutorial-materials/           # Educational resources
    └── comparative-studies/          # Comparative analyses
```

## Document Classification

### Access Categories

- **OA**: Open Access (freely available)
- **AP**: Author Permission (personal copies from authors)
- **PD**: Public Domain (historical works, expired copyright)
- **IR**: Institutional Repository (university archives)
- **AR**: arXiv Preprints (e-print archives)

### Quality Indicators

- **Core**: Essential foundational papers
- **Survey**: Comprehensive overview papers
- **Tutorial**: Educational materials
- **Historical**: Historically significant
- **Recent**: Latest developments (2020+)

## Metadata Standards

### BibTeX Requirements

All papers must include:
- Complete bibliographic information
- DOI when available
- arXiv identifier for preprints
- URL to original source
- Local file path
- Access classification
- Topic tags
- Quality indicator

### File Naming Convention

Format: `{year}_{first_author_surname}_{short_title}_{access_type}.pdf`

Examples:
- `1936_church_lambda_calculus_pd.pdf`
- `1987_girard_linear_logic_oa.pdf`
- `2024_author_quantum_lambda_ar.pdf`

## Copyright and Fair Use

### Guidelines

1. **Historical Papers**: Pre-1924 works are in public domain
2. **Open Access**: Explicitly licensed for distribution
3. **Author Copies**: Personal copies with author permission
4. **Fair Use**: Educational and research purposes only
5. **No Redistribution**: Archive for personal research only

### Sources Prioritized

1. **arXiv.org**: Primary source for preprints
2. **Author Homepages**: Personal copies and recent works
3. **Institutional Repositories**: University archives
4. **Open Access Journals**: PLOS, ACM Open Access, etc.
5. **Historical Archives**: Internet Archive, Google Books

## Usage Guidelines

### Research Use

- Papers organized by theoretical development
- Cross-references through citation index
- Full-text search capabilities (when implemented)
- Author and topic-based browsing

### Educational Use

- Structured progression from historical to modern
- Tutorial materials in dedicated section
- Annotated bibliographies for learning paths
- Comparative studies for understanding evolution

## Maintenance

### Regular Tasks

1. **Weekly**: Check for new arXiv papers
2. **Monthly**: Validate download links
3. **Quarterly**: Update metadata and indices
4. **Annually**: Review archive structure and expand

### Update Process

1. Identify new papers from conferences/journals
2. Verify open access status
3. Download and organize according to structure
4. Update metadata and regenerate indices
5. Commit changes to version control

## Technical Implementation

### Dependencies

- Python 3.8+ for automation scripts
- BibTeX tools for bibliography management
- JSON for metadata storage
- Markdown for documentation

### Integration

- Compatible with existing lambda-research structure
- Automated integration with build system
- Cross-references with implementation directories
- Searchable through repository-wide tools