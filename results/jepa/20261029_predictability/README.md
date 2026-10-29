# Predictability extension, 29 October 2026

registered: false. Grid remains paused. The cohort is the original first 20 dev probe-test runs, all 1,000 events (340 photon, 340 muon, 320 electron), split by runs 10/10. H-a–H-c use the photon rows; the other domains are descriptive controls. See `DECISIONS.md` for the author clarification, checkpoint provenance, engineering choices, and resource accounting.

The runbook baseline photon N_min exclusion was set to zero in this author-clarified extension solely to retain all 1,000 frozen events; counts below the original N_min are reported in each summary. Raw realisations are not retained.

Large arrays remain on Deepthought under `/ceph/dwong/jepa/returns/ext_predictability_20261029/budget/`:

- `nominal_A.npz`: `6c17097658aa1e9d88a4a68cafbb65916d534359155d225d8aede26b00d45b15`
- `nominal_B.npz`: `ad06f30dd42c83018f3bd95616583cd0483ae0c81b8de89fec4c6ee1e7c0009d`
- `strong_drift_A.npz`: `984a5ab7bc364a3d15f7ff3cb5b69b772531eb72d11f2eeea0f759256f83a62a`
- `strong_drift_B.npz`: `302871a73ef5afff674ba81259795365f3e19e879f9589fe348f8151dad9db8d`

All four hashes are SHA-256. The small summaries and probes were verified against their host SHA-256 before copying. Part 2 is pending at this commit.
