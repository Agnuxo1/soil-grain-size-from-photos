# Reproduction boundaries and commands

This source archive preserves the executed algorithms and selected aggregate evidence. It is not
a clean-room end-to-end rebuild of the recovered 1,794-feature physical cache. That cache and the
authorised competition inputs are prerequisites. They are not redistributed here.

Files under source/ are byte-preserved snapshots. Restore them to the original audit layout
described by the scripts before executing; their presence in this archive does not configure
the private input locations. The archived scripts use their original Windows D:/PROJECTS paths and managed Debian project
directories. Preserve the recorded source when auditing it; adapt paths in a separate copy if
rebuilding elsewhere. The raw historical script is archived as .py.txt: its main reads cached
test features and writes a CSV/report. Do not run that main against an original workspace.

1. Obtain authorised competition data and rebuild the original feature caches through their
   documented project pipeline. Preserve source/data hashes and original outputs.
2. Reproduce the retained recipe with source/patch/reproduce_best.py. It checks the original 20 metrics
   and byte-exact existing CSV; it does not select a new model. Its input proof/working layout
   comes from the private audit workspace.
3. The managed GPU sources validate source/data/model/runtime/order, reject optimised Python,
   and save atomic checkpoints under exclusive locks. Patch uses smallDINOv2; large uses a
   four-photo FP16/FP32 precision smoke before processing 127 training photographs. These are feature
   extraction tasks, with no label fitting or test images in their submitted packages.
4. Reproduce the comparisons with each evaluate*.py in a restored audit workspace. Internal
   manifests intentionally reference private input arrays, job receipts and complete reports
   omitted from this public source-only archive. Rebuild and freeze those proofs from authorised
   inputs before execution; do not treat missing proofs as permission to bypass the guards.
5. Compare the public aggregates in results/summary.json. Do not interpret archived training
   predictions as official leaderboard results. No candidate in this batch qualified for test.

Each comparison used 20 fixed five-fold seeds 20260925 through 20260944. Four threads were used for CPU
linear algebra. See included baseline runtime metadata and GPU RUN_CONTRACT.json files for exact
versions. Model provenance is pinned in source/large/MODEL_PROVENANCE.json and source contracts.

Primary model references:
- https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m/tree/07c9addcbc7d8ca698afe88784a057056aec490d
- https://huggingface.co/timm/vit_large_patch14_reg4_dinov2.lvd142m/tree/f3c408e77602bb412aa65fb03dfa0d5f95cb3832
- https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md

The public notebook only verifies/extracts this data-free source archive and displays the
aggregates. It has GPU/TPU/internet disabled and never reads competition inputs or submits.
