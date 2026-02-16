# Lambda Research Repository - Index System

**Central navigation hub for discovering and exploring repository content**

## Overview

This directory contains seven specialized indices that provide different views into the lambda calculus research repository. Each index offers a unique perspective for discovering papers, implementations, and research areas.

## Available Indices

### 📚 [Comprehensive Index](comprehensive.md)
**Complete repository structure and content overview**
- Organized by 31 research categories (01-31)
- Shows paper counts and key topics per category
- Links to bibliographies and implementations
- Best for: Understanding repository organization and finding specific research areas

### 👤 [Author Index](by-author.md)
**Alphabetical listing by author**
- A-Z directory of researchers and their contributions
- Multiple papers per author grouped together
- Cross-references to categories and topics
- Best for: Finding work by specific researchers or tracking author contributions

### 📅 [Chronological Index](chronological.md)
**Timeline of lambda calculus research (1918-2025)**
- Historical progression of ideas
- Decade-by-decade organization
- Shows evolution of the field
- Best for: Understanding historical development and finding papers by year

### 🏷️ [Topic Index](by-topic.md)
**Organized by research area and theme**
- Major topics: Type systems, semantics, logic, computation
- Cross-category topic clusters
- Related concepts grouped together
- Best for: Finding papers on specific technical topics

### 🔗 [Citation Index](by-citation.md)
**Citation network analysis**
- Highly-cited foundational papers
- Citation relationships between works
- Influence and impact metrics
- Best for: Finding influential papers and understanding citation networks

### 🔓 [Access Type Index](access-type.md)
**Sorted by accessibility**
- Open access papers (freely available)
- Paywalled journal articles
- ArXiv preprints
- Historical scans and archives
- Best for: Finding freely accessible papers

### 📊 [Statistics](statistics.md)
**Repository metrics and summaries**
- Total paper counts
- Papers per category
- Coverage statistics
- Growth over time
- Best for: Quick overview of repository scope

## Quick Statistics

- **Total Research Categories**: 31
- **Total Papers**: 700+ citations
- **Time Span**: 1918-2025 (107 years)
- **Major Authors**: Church, Curry, Scott, Reynolds, Girard, Pierce, and many more

## How to Use These Indices

### Finding Papers

1. **By Author**: Know the researcher? → [Author Index](by-author.md)
2. **By Topic**: Know the subject? → [Topic Index](by-topic.md)
3. **By Year**: Know the time period? → [Chronological Index](chronological.md)
4. **By Category**: Know the lambda variant? → [Comprehensive Index](comprehensive.md)

### Research Workflows

**Starting New Research**: 
- Check [Topic Index](by-topic.md) for related work
- Review [Citation Index](by-citation.md) for foundational papers
- Browse [Comprehensive Index](comprehensive.md) for relevant category

**Literature Review**:
- Start with [Citation Index](by-citation.md) for key papers
- Use [Chronological Index](chronological.md) to track development
- Check [Author Index](by-author.md) for major contributors

**Finding Implementations**:
- Navigate via [Comprehensive Index](comprehensive.md) to category
- Check category's `implementations/` directory
- See external references in Implementations Catalog

**Accessing Papers**:
- Use [Access Type Index](access-type.md) to find open access papers
- Check papers-archive/ for downloaded PDFs
- Follow DOI links for official sources

## Index Maintenance

### Update Frequency
- **Automated**: Statistics regenerated on each build
- **Manual**: Indices updated when new categories or major papers added
- **Validation**: `./validate-repository.py` checks link integrity

### Contributing
When adding new papers or categories:
1. Update category bibliography.md
2. Run `./validate-repository.py`
3. Indices auto-update from bibliography files
4. Manual review for cross-references

## Related Documentation

- **[Comprehensive Bibliography](../comprehensive-bibliography.md)**: Full citation list
- **[Research Highlights](../research-highlights.md)**: Featured papers and key findings
- **[Papers Archive](../../papers-archive/)**: Downloaded paper collection
- **[Implementations Catalog](../implementations/)**: Code implementations reference

## Index Structure

```
docs/indices/
├── README.md              ← You are here
├── comprehensive.md       ← Repository structure (31 categories)
├── by-author.md          ← Alphabetical by researcher
├── chronological.md      ← Timeline 1918-2025
├── by-topic.md           ← By research theme
├── by-citation.md        ← Citation network
├── access-type.md        ← By availability
└── statistics.md         ← Repository metrics
```

## Navigation Tips

- **Cross-links**: Follow related topic links between indices
- **Search**: Use browser search (Ctrl+F / Cmd+F) within any index
- **MkDocs Search**: Use site-wide search for specific terms
- **Git History**: See `git log` for index evolution

---

**Last Updated**: 2025-12-24  
**Maintained by**: Repository automation and manual curation  
**Questions?**: See repository README.md or open an issue
