# Sealed access log (r2-sparse)

Canonical copy: `/ceph/dwong/jepa/returns/r2_sparse_2026-11-24/SEALED_ACCESS_LOG.md` (appended under a lock by the writing script). Format: UTC | commit | what | purpose | script. Passes before `shards_w2b` were superseded (no metrics in any).

- 2026-11-23T23:38:10Z | c3a00e463fb5 | wcte sealed_test runs 0:600 | W2 re-pack from the lucid_v1 LUCiD cache, window [-110, +50] ns (packing only, no metrics) | scripts/sparse_w2_repack.py
- 2026-11-23T23:44:43Z | 3d6cfd5b824e | wcte sealed_test runs 0:600 | W2 re-pack from the lucid_v1 LUCiD cache, window [-110, +50] ns (packing only, no metrics) | scripts/sparse_w2_repack.py
- 2026-11-23T23:53:24Z | 63ea330ec583 | wcte sealed_test runs 0:600 | W2 re-pack from the lucid_v1 LUCiD cache, window [-110, +50] ns (packing only, no metrics) | scripts/sparse_w2_repack.py
- 2026-11-24T00:01:01Z | 78c7b2244de6 | wcte sealed_test runs 0:600 | W2 re-pack from the lucid_v1 LUCiD cache, window [-110, +50] ns (packing only, no metrics) | scripts/sparse_w2_repack.py
- 2026-11-24T00:06:18Z | ddde5ca2990d | wcte sealed_test | W2 structural validation v2 vs v1 shards and cache (event ids, truths, hit identity, triggers, dark counts; no model metrics) | scripts/sparse_w2_pack.py
- 2026-11-24T00:10:44Z | 1a75f3b47e8f | wcte sealed_test | pack lucid_v1_w2 sealed shards into wcte_lucid_v1_w2_sealed.h5 (packing + byte verification only, no metrics) | scripts/sparse_w2_pack.py
