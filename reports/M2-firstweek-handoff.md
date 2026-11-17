# M2 first-week handoff — Ramez's tasks (T01, T08, T09)

**Prepared:** 2026-11-17 · **Owner of this package:** Ramez Hesham · **Status vocabulary:** PREPARED = ready in this package, not yet in the repo · UPLOADED = already on `main` · NEEDS REAL DATA = code is tested but must run on the real EuroSAT_MS dataset.

## What changed vs current `main` (commit 2d8b26d)

| File | Status | What it is |
|---|---|---|
| `reports/band_correlation_interpretation.md` | PREPARED (new) | M2-T01 interpretation of the existing band-correlation artifacts, with verified provenance (single-image pixel-level method, AnnualCrop_1.tif) and verified band order (B8A stored last). |
| `src/compute_band_correlation.py` | PREPARED (new), NEEDS REAL DATA to produce numbers | Dataset-level band correlation (per-image band means across images). Train-split only by default (no test leakage), fails clearly if the split file is missing/stale, `--split all` = disclosed report-only mode. |
| `tests/test_dataset_ms.py` | PREPARED (new) | 13 synthetic regression tests for the 13-band MS path (M2-T09's "with tests"). |
| `tests/test_compute_band_correlation.py` | PREPARED (new) | 4 planted-signal tests for the correlation script: exact reference recovery, val/test exclusion, clear failure on missing/stale split. |
| `reports/M2-firstweek-handoff.md` | PREPARED (new) | This handoff. |
| `src/model.py` | PREPARED (modified, overwrite) | `build_model()` now honors `in_channels` (default 3, RGB entry point unchanged); `resnet18` preserved, 3-channel only with a clear error for 13 until M3-T03. |
| `src/evaluate.py` | PREPARED (modified, overwrite) | Reusable `validate_checkpoint` helper: rejects checkpoints from a different model/channel/class contract (model name, channel count, class list order/count, then tensor shapes as fallback) with a clear error instead of a cryptic state_dict failure. No behavior change for matching checkpoints. |
| `tests/test_model_wiring.py` | PREPARED (new) | 13 model-contract tests calling the real helper: 13-band forward, channel pass-through, RGB default preserved, ResNet18 preserved (offline via monkeypatched torchvision constructor, pretraining request asserted), checkpoint round-trip through the actual guard, and rejection of wrong model name / channel count / tensor shape / class order / class count. |

The two modified files are non-destructive compatible edits: nothing was deleted, and the RGB path behavior is unchanged. They must OVERWRITE their existing versions when copying; the other six files are pure additions.

## Test evidence

`python -m pytest tests/` → **35/35 passed** (2026-11-18, CPU): 13 MS loader tests + 4 correlation-script tests + 13 model-wiring tests + 5 pre-existing RGB-side tests (unchanged). All tests use synthetic fixtures; no real-data training or metrics are claimed. Model proof: a 13-band SmallCNN forward pass returns the right shape; a checkpoint saved in the training format round-trips through evaluation's real `validate_checkpoint` guard; wrong model name, channel count, tensor shape, and class list (order or count) checkpoints are all rejected, not silently loaded. ResNet18 tests run offline against a monkeypatched constructor so no weight download can mask behavior.

## Model contract (SmallCNN vs ResNet18) — what the conflict was

- `build_model("cnn")` ignored channels and always built a 3-channel SmallCNN, while `src/train.py`/`src/evaluate.py` build a 13-channel SmallCNN directly — so "CNN" had two different channel contracts depending on entry point, and `build_model("cnn")` crashed on 13-band input. Fixed: the helper now takes `in_channels` (default 3, RGB unchanged).
- `build_model("resnet18")` is 3-channel only (torchvision, ImageNet weights, 11.2M params vs SmallCNN's 292k). It cannot consume 13 bands without the planned first-conv adaptation — that is M3-T03 (Nov 26, Ramez): adapt ResNet18 3->13 channels and run the SAME split + seed comparison against SmallCNN, then M3-T04 (Nov 27) records the team decision.
- Current state: SmallCNN-13 is the only architecture that runs on the MS path today and is the provisional baseline. Which model is BETTER is not known yet — the recorded 93.85% is historical and unverified, and no fair same-split comparison exists. No accuracy claim is made here.
- `src/evaluate.py` now refuses cross-contract checkpoints (e.g. an RGB 3-channel checkpoint in the 13-band evaluation) with a clear error.

## Task status against the delivered plan

- **M2-T01** (band correlation interpretation, Ramez, Nov 19): interpretation PREPARED; heatmap/CSV already UPLOADED with the migration commit. Remaining: upload the new files + run `src/compute_band_correlation.py` on real data for dataset-level numbers.
- **M2-T08** (repo cleanup, Ramez, review Hazem, Nov 19): UPLOADED — dead submodule links removed in commit 2d8b26d; `temp_verify.py` kept, treated as scratch per plan. Remaining: Hazem's review.
- **M2-T09** (13-band port beside RGB, Ramez, review Malak+Hazem, Nov 21): UPLOADED (loader/train/evaluate MS path) + tests now PREPARED. Remaining: Malak/Hazem review.

## Next steps and owners

1. **Ramez — one commit:** copy the ZIP contents into the repo root (the two modified files `src/model.py` and `src/evaluate.py` must overwrite). Suggested message: `Add T01 interpretation + dataset-level correlation, MS path tests, and model channel-contract fix (M2-T01, M2-T09)`.
2. **Ramez (or anyone with the dataset) — real-data run:**
   `python -m src.compute_band_correlation --data-dir data/raw/EuroSAT_MS --max-per-class 300 --seed 42`
   Default is train-only. Dataset sources: already on Malak's machine (per `reports/tables/eda_summary.csv`), or `python -m src.download_eurosat --kind ms` (zenodo). Then update the interpretation's numbers with the dataset-level values.
3. **Hazem:** review T08 (and the stale band-order line in `notes visualization.txt` — wavelength order listed, storage order is B8A-last).
4. **Malak + Hazem:** review T09 (MS path + new tests).
5. **Rawda (M2-T03, Nov 24):** fit PCA on the TRAIN split only; the interpretation's two high-covariation groups are the hypothesis to test, not a conclusion.
6. **Ramez:** mark T01/T08/T09 on the Notion board as done-pending-review only after upload.

## Known limits (unchanged)

- Existing correlation CSV/heatmap describe ONE image's pixels (AnnualCrop_1.tif) — descriptive only.
- 93.85% remains a historical, unverified number; no new training, metrics, or checkpoints exist.
- RGB/MS path unification (single config/entry point) is still open project work, outside T09's "beside RGB" acceptance.
