# Gemini Context: Lambda Calculus Research Repository

## Directory Overview

This repository is a comprehensive, non-code, academic knowledge base dedicated to lambda calculus, type theory, and related fields in theoretical computer science. It is structured as a curated collection of research papers, implementations, tutorials, and bibliographies.

The primary purpose of this repository is to serve as a centralized, high-quality resource for researchers, students, and practitioners. The content is organized into a detailed thematic structure, and it uses **MkDocs** with the Material theme to generate a static documentation website.

## Key Files and Directories

*   **Numbered Directories (`01-` to `31-`):** Each directory represents a specific topic (e.g., `01-untyped-lambda-calculus`, `06-linear-lambda-calculus`). Inside each, you will find:
    *   `papers/`: Academic papers related to the topic.
    *   `implementations/`: Descriptions or links to software implementations.
    *   `tutorials/`: Educational content.
    *   `historical/`: Foundational works.

*   **`papers-archive/`:** A directory containing a large, organized collection of research papers. It has its own automation for maintenance.

*   **`docs/`:** The source directory for the MkDocs-generated website content.

*   **`mkdocs.yml`:** The main configuration file for the **MkDocs** static site generator. It defines the site's structure, navigation, theme, and plugins (including `bibtex` for citations and `with-pdf` for PDF generation).

*   **`CROSS_REFERENCE_SYSTEM.md`:** A key document that outlines the theoretical connections and relationships between the different topics covered in the repository. It serves as a conceptual map of the domain.

*   **`standardize_bibliography.py`:** A Python script used to enforce a consistent citation style across all bibliography files in the repository.

*   **`validate-repository.py`:** A Python script for checking the integrity and structure of the repository, ensuring that files and directories adhere to the established conventions.

*   **`Makefile`:** Located in the `papers-archive/` directory, this file automates tasks such as downloading papers, running searches, and generating indices.

## Usage and Development Workflows

This is a non-code project focused on documentation and knowledge management.

### Building the Documentation Site

The repository uses MkDocs to generate a static website.

*   **To serve the site locally (with live-reloading):**
    ```bash
    mkdocs serve
    ```
    The site will be available at `http://127.0.0.1:8000`.

*   **To build the static site for deployment:**
    ```bash
    mkdocs build
    ```
    The output will be generated in the `site/` directory.

### Maintaining the Repository

*   **Standardizing Bibliographies:** To ensure all bibliography files are consistent, run the standardization script. It takes a glob pattern to find the target files.
    ```bash
    python3 standardize_bibliography.py '**/bibliography.md'
    ```

*   **Automating Paper Management:** The `papers-archive/` directory contains a `Makefile` for managing the paper collection.
    ```bash
    cd papers-archive/
    make # See available commands
    make download-high # Download high-priority papers
    make search QUERY="dependent types" # Search for papers
    ```
