<!-- AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT. -->
<!-- Source of truth: registry/external_sources.toml -->
<!-- Canonical write path: registry/canonical/control_plane.sqlite3 -->
<!-- Source label: XS-020 -->
<!-- Regenerate with: cargo run -p gororoba_cli_data --bin provenance -- export-external-sources -->

# SOHO Archive Mirror Audit (2026-03-09)

This note records a host-level audit of current and historical SOHO archive
endpoints. The goal is not to duplicate mission documentation, but to identify
which endpoints are practical for retrieval from this machine and how they
should be classified inside the repo.

## Recommended Source Ladder

1. GSFC archive status pages for freshness and archive topology.
2. ESA/ESAC command-line and mission-long documentation for current European
   archive workflow, with live archive access treated as host-dependent because
   direct archive endpoints reset from this machine.
3. MEDOC/IAS and VSO for discovery and cross-archive lookup.
4. Instrument-side direct access where available, especially NRL LASCO for bulk
   wget-friendly retrieval.
5. UKSSDC and Torino only as historical breadcrumbs, not as primary byte
   sources.

## Tested Endpoints

### Current and reachable here

- GSFC status/index:
  - <https://soho.nascom.nasa.gov/data/archive/index_gsfc.html>
  - HTTP 200 from this host.
  - Contains the current archive status table and still mentions IAS and Torino
    under the European mirror context.
- GSFC compact chooser:
  - <https://soho.nascom.nasa.gov/data/archive/>
  - HTTP 200 from this host.
  - Contains current archive freshness information and links onward to ESAC and
    VSO.
- GSFC full landing page:
  - <https://soho.nascom.nasa.gov/data/archive.html>
  - HTTP 200 from this host.
  - Includes mission-long files, deprecated archive pointers, and
    instrument-side catalog links.
- GSFC holdings dashboard:
  - <https://soho.nascom.nasa.gov/data/.dash/sohoar-dashboard.html>
  - HTTP 200 from this host.
  - The dashboard HTML currently reports total holdings as:
    - files: `4647328`
    - total size (TB): `9.9881729`
- ESA SOHO command-line documentation:
  - <https://www.cosmos.esa.int/web/soho/command-line>
  - HTTP 200 from this host.
  - Confirms TAP-based command-line access.
- ESA SOHO TAP functionality page:
  - <https://www.cosmos.esa.int/web/soho/tap-main-functionalities>
  - HTTP 200 from this host.
  - Mentions TAP and `50 GB` delayed-download guidance.
- ESA SOHO mission-long files:
  - <https://www.cosmos.esa.int/web/soho/mission-long-files>
  - HTTP 200 from this host.
- VSO search:
  - <https://sdac.virtualsolar.org/cgi/search>
  - HTTP 200 from this host.
  - Page text confirms VSO instances at Home, NSO, and Stanford. Treat this as
    a federated discovery surface, not a single byte-serving mirror.
- MEDOC/IAS portal:
  - <https://idoc-medoc.ias.u-psud.fr/sitools/client-user/index.html?project=Medoc-Solar-Portal>
  - HTTP 200 from this host.
  - Reachable as a solar-archive discovery portal.
- NRL LASCO direct access:
  - <https://lasco-www.nrl.navy.mil/index.php?p=get_data>
  - HTTP 200 from this host.
  - Page text explicitly mentions `wget` and `level_05`; this is the strongest
    instrument-side bulk-download candidate in this audit.
- EIT catalog:
  - <https://umbra.nascom.nasa.gov/eit/eit-catalog.html>
  - HTTP 200 from this host.
- SUMER image database host:
  - <https://www2.mps.mpg.de/>
  - HTTP 200 from this host.

### Documented current surfaces that reset from this host

- ESA/ESAC SOHO Science Archive search:
  - <https://ssa.esac.esa.int/ssa/#/pages/search>
  - Connection reset by peer from this host.
- ESA/ESAC SOHO archive root:
  - <https://ssa.esac.esa.int/ssa/>
  - Connection reset by peer from this host.
- ESA/ESAC TAP tables:
  - <https://ssa.esac.esa.int/ssa-sl-tap/tap/tables>
  - Connection reset by peer from this host.
- ESA/ESAC TAP capabilities:
  - <https://ssa.esac.esa.int/ssa-sl-tap/tap/capabilities>
  - Connection reset by peer from this host.

Interpretation: ESA documentation is current and reachable, but the live ESAC
archive and TAP byte/query endpoints are not usable from this machine right now.
Keep ESAC in the source ladder, but mark it as host-dependent until a working
session or alternate route is demonstrated.

### Historical or weak fallbacks

- UKSSDC SOHO hub:
  - <https://www.ukssdc.ac.uk/solar/soho/data.html>
  - HTTP 200 from this host.
  - Still useful as a map of older mirror relationships.
- Torino mirror URLs:
  - <http://solar.to.astro.it/>
  - <http://solar.oato.inaf.it/>
  - Both fail name resolution from this host.
  - Treat Torino as historical or intermittent, not dependable.

### Legacy breadcrumbs that are no longer useful here

- SEAL CGI:
  - <https://seal.nascom.nasa.gov/cgi-bin/gui_seal>
  - HTTP 404 from this host.
- Old ESAC mirror-style pages:
  - <http://soho.esac.esa.int/data/archive/index_ssa.html>
  - <http://soho.esac.esa.int/data/archive/index_gsfc.html>
  - HTTP 404 from this host.

## Practical Inclusion Guidance

- Use GSFC archive pages as the first freshness and topology reference.
- Use ESA command-line and mission-long docs as the current documentation anchor
  for European access.
- Prefer MEDOC and VSO for search/discovery.
- Prefer NRL LASCO direct access for LASCO bulk download workflows.
- Keep UKSSDC and Torino only as context for historical mirror discussions.
- Treat direct ESAC archive/TAP access as "documented but flaky from this host"
  until a working retrieval path is demonstrated.
