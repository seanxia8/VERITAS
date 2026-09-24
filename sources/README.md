# `sources/` — research notes and raw API dumps

Notes gathered while researching testbeds and prior work. The prose notes are
hand-written and committed; `raw/` holds the unedited API responses they were
built from (arXiv Atom XML, OpenAlex JSON, CERN Open Data JSON). `raw/` is
**gitignored** (regenerable, ~3.6 MB) — re-issue the queries to recreate it.

| file | what |
|---|---|
| `atlas_cms_detector_layout_research_2026-09-18.md` | CMS/ATLAS same-physics/different-layout simulation, open datasets, and representation-transfer prior work (Tier 3b) |
| `raw/arxiv_*.xml` | arXiv API responses (queries on foundation models, cross-detector transfer, Delphes, Key4hep, TrackML, MLPF, domain shift, …) |
| `raw/abs_*.xml` | arXiv abstracts pulled for the key papers |
| `raw/openalex_*.json` | OpenAlex works queries |
| `raw/cernopendata_*.json`, `raw/cod_*.json` | CERN Open Data records API responses |
| `raw/s2_detector_transfer.json` | Semantic Scholar query (rate-limited; kept for the record) |

Re-running the queries needs no key for arXiv/OpenAlex/CERN; see the commands in
the note or re-issue with `curl`. Do not treat `raw/` as curated — the prose
notes are the record.
