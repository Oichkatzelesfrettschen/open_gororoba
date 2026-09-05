# Research source redistribution boundary

The audit covers the exact 43 members of the original research archive.
The original entry-list SHA256 is
`787272e61730bde312f63cef89eea5d6ae301dbbc8a60d53a9140b60fcaa2573`.
The original compressed archive SHA256 is
`4cddd2b33ade6c94753ef9faaaef8eed4744d146a2dc404305206f0dd4214fdd`.
The public archive contains 24 members (2812107 source bytes). Private
retention contains the other 19 members (1844239 bytes), with their public
hash inventory in `calibration-research-private.sha256`.

## Full-text permissions supported by primary evidence

| Work and retained files | Primary license evidence | Public disposition |
| --- | --- | --- |
| Daniel Lakens, *Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses* (2017), `lakens-equivalence.xml` and `.txt` | The XML permissions element states Creative Commons Attribution 4.0 and permits reproduction and distribution with attribution. | Retain XML and derived text with this attribution and the license link below. |
| scikit-learn developers, *Common pitfalls and recommended practices*, `sklearn-common-pitfalls.html` and `.txt` | The captured HTML footer identifies the developers and BSD License; upstream `COPYING` supplies BSD 3-Clause terms. | Retain the captured HTML and extracted text with the complete notice below. |
| Christoph Bergmeir, Rob J Hyndman and Bonsoo Koo, *A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction*, author preprint dated 2017-07-23, PDF and extracted text | The inspected preprint and author publication page contain no redistribution grant. Publisher-deposited Crossref metadata identifies an Elsevier text-and-data-mining license, not a general redistribution license for the author preprint. | Preserve both files privately; publish citation and hashes. |
| Vitor Cerqueira, Luis Torgo and Igor Mozetic, *Evaluating time series forecasting models: An empirical study on performance estimation methods*, arXiv:1905.11744v1, PDF and extracted text | The version-specific arXiv page links `nonexclusive-distrib/1.0`. That grant to arXiv does not establish general third-party republication permission. Publisher-deposited metadata identifies Springer text-and-data-mining terms. | Preserve both files privately; publish citation and hashes. |

Lakens attribution: Daniel Lakens (2017), DOI
[10.1177/1948550617697177](https://doi.org/10.1177/1948550617697177),
retrieved from the [Europe PMC full-text endpoint](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5502906/fullTextXML),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The XML bytes are retained unchanged. The `.txt` member is an extracted-text
derivative; extraction changes markup and layout.

The scikit-learn document comes from
[the official common-pitfalls page](https://scikit-learn.org/stable/common_pitfalls.html).
The HTML bytes are retained unchanged; the text member is an extracted-text
derivative. The upstream license was checked at
[scikit-learn COPYING](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/COPYING).

The inspected Bergmeir primary surfaces are
[the author-hosted PDF](https://robjhyndman.com/papers/cv-wp.pdf) and
[the author publication page](https://robjhyndman.com/publications/cv-time-series/).
The journal DOI is [10.1016/j.csda.2017.11.003](https://doi.org/10.1016/j.csda.2017.11.003).
The Cerqueira license observation is tied to
[arXiv:1905.11744v1](https://arxiv.org/abs/1905.11744v1), rather than inferred
from another version or from download availability. The journal DOI is
[10.1007/s10994-020-05910-7](https://doi.org/10.1007/s10994-020-05910-7).

## Other full-page captures

The variable-description PDF and its extracted text, plus eight Berkeley
technical pages, lack an explicit redistribution grant in the inspected
captures. Those ten files remain private. Funding or university hosting
does not by itself supply a document-specific license.

The Auster publisher response is an article landing-page capture. Its
publisher metadata supplies text-and-data-mining terms without an explicit
general redistribution grant. The two UCL challenge pages and the two
Staples publisher challenge responses also remain private. In particular,
`staples-paper-publisher.pdf` and `staples-paper.xml` contain HTML challenge
responses rather than article PDF/XML. Their filename suffixes do not
establish document identity or permissions.

`calibration-research-private.entries` enumerates all 19 excluded files.
The 24-member public whitelist consists of the four licensed full-text
members above, factual provider/index/citation metadata, the independently
verified V2 catalog, and project-authored findings and research notes.
The official dataset license review belongs to its separate retained
provenance record; this audit does not repeat that review.

Exact private files and the original 43-member archive are retained under
`.cache/empirical-claim-private-research` in both the working checkout and
the primary checkout. Directory mode is 0700 and private-file mode is 0600.
Copies refused differing existing destinations and passed byte comparison.
The original archive and original member manifests are retained there too.
Public and private entry lists partition the original 43-file set. The
intake archive's 684 files remain unchanged, preserving all 727 original
source members across the public/private boundary.

## License lookup identities

The following primary-source response hashes identify the inspected license
surfaces. These small lookup captures remain in the worktree's ignored
`.cache/calibration-source-license-audit` directory and in the primary
checkout's private research `license-lookups` directory. The operative
license statements and source URLs are recorded above.

| Capture | SHA256 |
| --- | --- |
| `cerqueira-arxiv-v1.html` | `0d45bfa14808bcfe04da50519878192789b5bacb697913f82e3b0ebb7292f075` |
| `bergmeir-author-publication.html` | `f6cfbb82a24ab11a23c8a747ecef7e2ba3e607f1a56e9fdbbe6cbee461db8119` |
| `sklearn-COPYING` | `50d6a9d340f19ab355609917993114daf5f47e3161067bcf34955bbd05cd9cb0` |

## scikit-learn BSD notice

BSD 3-Clause License

Copyright (c) 2007-2026 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
