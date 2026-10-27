# Analysis scripts for the technical report

These scripts derive every log-based number and figure in the Quantum Sensing Agent technical report from the
session logs in `projects/NVExperiment/runs/` and the 14 May 2025 data in `examples/2025-05-14-nv-odmr/`.

```bash
cd analysis
pip install matplotlib numpy scipy pillow
sh run_all.sh        # writes stats.json, data/*.csv and figures/ here
```

Definitions used throughout:

- **session**: a contiguous part of one log with no gap longer than 3 h;
- **reached hardware**: a logged `RUN OUTPUT` entry contains `PB connected` or a `[Runner] Saved` line;
- **completed measurement**: a `[Runner] Saved <GalvoScan|FindNV|ESR|Optimization> data to:` line at the start of a
  line **inside a logged `RUN OUTPUT` entry**. Such lines inside an LLM reply (invented tool output) do not count.

Expected totals: 31 logs, 36 sessions, 16 reaching hardware on 10 days, 73 completed measurements
(20 galvo / 27 find_nv / 8 optimize / 18 ESR). `guidance_check.py` checks the 22 parametric operator instructions;
`esr_fit.py` fits the 14 May candidate-3 spectrum (fit done for the report, not by the agent).
