# Reproduction status — investigation closed, 22 September 2026

The current manuscript is an isolated review draft, not a replacement of the authoritative manuscript. Its scientific results and production defaults are unchanged.

| Route | What is verified | What remains limited |
|---|---|---|
| Archived manuscript rebuild | 24 archive input hashes, 17 numerical table groups at displayed precision, 10 figures from saved arrays/metrics, compiled PDF | Does not test fitting or model inference |
| Supplied historical checkpoints | Selected Ex1/Ex8 seed-0 reconstruction checks passed original rtol=1e-5, atol=1e-6 | Requires external checkpoints; not every saved model was reevaluated |
| Fresh training, original settings | Complete corrected Ex8 ARFF K128/N80000/seed-0 seven-fit path executed | Two fresh processes differed; strict fresh-process repeatability is not established. One matched the archive, the other failed |
| Optional autotune-disabled policy | Three minimal processes (twice each), then two full seven-fit processes: bit-identical scientific outputs within the tested scope | Neither full run matched the historical archive within original tolerances; no general determinism or historical-cause conclusion |

All full checks used packaged corrected data, not data generation. Other configurations, methods, hardware and whole-study distributional reproduction are unverified by these checks. All failed comparisons and original tolerances remain preserved. Detailed runtime, GPU UUID, configuration and the exact optional command are in [REPRODUCING.md](REPRODUCING.md); diagnostic evidence is in [the policy report](docs/autotune0_policy_v1/REPORT.md). No additional numerical runs are authorized or launched by this closure.

## Remaining release tasks

- Select and authorize a durable hosting destination and license/access terms; upload the immutable bundle and verify a downloaded copy before advertising a URL.
- Transfer the bundle to genuinely independent storage and retain a destination checksum receipt. No verified off-KW61146 backup or public hosting identifier is recorded. The older same-server backup does not qualify.
- Publish the verified artifact identifier and availability alongside the repository when those tasks are complete. Historical source/selection gaps and author decisions remain disclosed; release preparation must not claim they are solved.

The prepared bundle is `../artifact_exports/independent_reproduction_v1.tar`: **6,978,959,360 bytes (6.50 GiB)**, 29,009 payload files, SHA-256 **54c939eb8d8add8ca92e3aa1c5906cc7379803124dd642ac426d69b9a06a51a4**. Its source checksum was reverified during closure. [BUNDLE.json](docs/independent_reproduction_v1/BUNDLE.json) and [BACKUP.md](docs/independent_reproduction_v1/BACKUP.md) give layout, transfer and payload-verification instructions. No transfer/upload was performed.

## Closure verification

The existing archive-only rebuild ran in the isolated pinned environment into a new output directory; no new predictions, fitting, stochastic data or compiler-policy tests were performed. All existing numerical checks passed. The isolated manuscript received only a reproducibility disclosure; compilation has no undefined references/citations or overfull-box warnings. The disclosure page was visually inspected. Existing numerical tables, coefficient arrays and figures were not replaced. The build record is [REPRODUCTION_CLOSURE_CHECKS.json](results/conservative_manuscript_revision_v1/REPRODUCTION_CLOSURE_CHECKS.json).
