# Example: 14 May 2025 session, galvo scan → find_nv → optimize → CW ESR

The measurement files from one hardware session of the Quantum Sensing Agent on the Yacoby-lab
NV confocal setup, Harvard (run `run_20250514_082859`, 08:28–11:53 local time).

- **Agent log:** [`projects/NVExperiment/runs/run_20250514_082859/logs/`](../../projects/NVExperiment/runs/run_20250514_082859/logs/)
- **Configs the agent wrote:** [`projects/NVExperiment/runs/run_20250514_082859/configs/`](../../projects/NVExperiment/runs/run_20250514_082859/configs/)
- **Data (this folder):** `data/`, 38 files (PNG plots rendered by the lab toolkit, and JSON data)
- **Figure script:** `make_hero_figure.py` builds `docs/figures/hero_2025-05-14.png`

## What happened

1. The researcher asked the agent to "characterize all the NV in the field of view". It ran a
   galvo scan with the default config and analyzed the plot with the vision model.
2. The researcher read six candidate spots off the scan and gave them to the agent.
3. For each candidate the agent wrote configs and ran `find_nv`, then (for candidates 1–2) z `optimize`,
   then CW ESR sweeps between 2.72 and 2.92 GHz at −8 or 0 dBm.
4. Candidates 1 and 2 showed no clear resonance. The agent listed possible causes (microwave delivery,
   NV orientation and charge state). The researcher checked the microwave line and moved on.
5. On candidate 3 a narrower sweep (2.85–2.89 GHz, 0 dBm, `esr_config_NV3_modified2.json`) shows two dips at
   ≈2.863 and 2.878 GHz, on either side of D = 2.87 GHz. The researcher logged "good we
   see ESR, let's proceed to the next NV" (11:49:53).
6. The session ended after `find_nv` on candidate 4.

In total the agent launched 20 measurements in this session: 1 galvo scan, 9 find_nv, 3 optimize, 7 ESR.
**No spectrum was fitted at the time**: `fit_params` is `null` in every ESR JSON file. A two-Lorentzian fit
of the candidate-3 sweep, done afterwards, is in the technical report (and `analysis/scripts/esr_fit.py`).

## Run → file map

File timestamps are the save times (end of each measurement).

| # | Start (log) | Script | Config | Output timestamp |
|---|---|---|---|---|
| 1 | 08:30:20 | galvo_scan | `default_galvo_scan_config.json` | `083103` |
| 2 | 09:03:43 | find_nv | `find_nv_config_NV1.json` | `090401` |
| 3 | 09:21:38 | find_nv | `find_nv_config_NV1_refined.json` | `092156` |
| 4 | 09:23:20 | optimize | `default_optimize_config.json` | `092342` |
| 5 | 09:24:13 | optimize | `default_optimize_config.json` | `092436` |
| 6 | 09:31:23 | ESR | `esr_config_NV1.json` | `094532` |
| 7 | 09:46:48 | ESR | `esr_config_NV1_lower.json` | `100103` |
| 8 | 10:04:35 | find_nv | `find_nv_config_NV2.json` | `100452` |
| 9 | 10:06:21 | find_nv | `find_nv_config_NV2_refined.json` | `100640` |
| 10 | 10:07:38 | find_nv | `find_nv_config_NV2_refined2.json` | `100757` |
| 11 | 10:08:23 | optimize | `default_optimize_config.json` | `100849` |
| 12 | 10:09:55 | ESR | `esr_config_NV2.json` | `102414` |
| 13 | 10:25:13 | ESR | `esr_config_NV2_lower.json` | `103929` |
| 14 | 10:42:21 | ESR | `esr_config_NV2_higher_power.json` | `105639` |
| 15 | 11:21:59 | find_nv | `find_nv_config_NV3.json` | `112216` |
| 16 | 11:23:08 | find_nv | `find_nv_config_NV3_refined.json` | `112326` |
| 17 | 11:25:23 | ESR | `esr_config_NV3.json` | `113937` |
| 18 | 11:45:57 | ESR | `esr_config_NV3_modified2.json` | `114852` |
| 19 | 11:50:48 | find_nv | `find_nv_config_NV4.json` | `115105` |
| 20 | 11:52:32 | find_nv | `find_nv_config_NV4_refined.json` | `115250` |

Runs 2–5 have no timestamped PNG. Their plots were saved under fixed names (`FindNV_plot.png`,
`Optimization_plot.png`) that later runs overwrote.

## Data format and caveats

- Positions are galvo/piezo **control voltages** (V). The repository has no V→µm calibration.
- Fluorescence is in kcounts/s.
- 1D arrays in the JSON files (ESR `frequency`, `data`, `data_err`; optimize `fluor_vector`,
  `original_fluor_vector`) are complete.
- 2D arrays (`image_data` in GalvoScan/FindNV files) were **truncated** by numpy's repr (`...`) when saved.
  Use the PNGs for the images.
- The ESR "fit" overlays in the toolkit PNGs come from an automatic Gaussian-dip routine; none was accepted as a fit.

## Provenance

The session's `data/` folder was excluded by `.gitignore`. These 38 files were uploaded to the repository
on 20 May 2025 in commit `b316b58` (by rholanx, at the path `run_20250514_082859/data/`) and removed the
same day in the folder reorganization `83526d2`. They were recovered with
`git show b316b58:run_20250514_082859/data/<file>`; each copy here is byte-identical to that blob.
Before release they were scanned for credentials (none found).
