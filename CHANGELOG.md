# Changelog

## v1.0: public release (2026)

- Repository cleaned up for release: development notes moved to `docs/dev-notes/`, test scripts to `tests/`,
  superseded files removed (`configs_OLD/`, `agent_old.py`, `agent_mac.py`, empty `gemini_engine.py`,
  duplicate root logs). They remain in the development history; if the public history is rewritten, they are
  only in the archival copy of that history. The superseded Feb–Mar configs referenced by the old logs are kept in
  `projects/NVExperiment/runs/legacy_configs_2025-02_03/`.
- API keys removed from the code. Keys are read from the environment / a `.env` file (`env_template.txt`).
  `anthropic_engine.py` no longer loads `.env` from a hard-coded lab-PC path.
- Added `README.md`, MIT `LICENSE`, this changelog, and `examples/2025-05-14-nv-odmr/` (the measurement files
  of the 14 May 2025 session, with the script for the README figure).
- `.gitignore` and `requirements.txt` completed.
- Fixed `save_conversation_embeddings()` in `agent/agent.py`: it used `conversation_for_embedding` before assigning
  it (since 14 Jun 2025), so embedding on exit always failed. Added `tests/test_save_embeddings.py`.
- Added `analysis/`: the scripts that reproduce the technical report's counts and figures from the logs.

## 2025 history (summary)

Dates are commit dates and run-folder timestamps.

- **Feb–Mar 2025.** First agent sessions on the NV confocal setup (logs from 18–19 Feb, 28 Feb – 14 Mar):
  the agent writes JSON configs, runs the four measurement scripts (galvo scan, find_nv, optimize, ESR)
  with the researcher's permission, and reads the resulting plots with a vision model.
- **3 Apr 2025.** Code imported into this repository (`init`).
- **8–16 Apr 2025.** Per-run folder structure (`projects/NVExperiment/runs/run_<timestamp>/{configs,data,logs}`)
  and path fixes.
- **19 Apr 2025.** Full galvo → find_nv → optimize → ESR session with researcher guidance.
- **2 May 2025.** Retrieval over past sessions (RAG): conversations are embedded at exit
  (all-MiniLM-L6-v2) and relevant past context is retrieved for new sessions.
- **14 May 2025.** Session with the full chain on several NV candidates; ESR dips observed on one candidate
  (see `examples/2025-05-14-nv-odmr/`).
- **20 May 2025.** `agent.py` updated ("based on local test and features"); wrapper scripts, default configs, embeddings and
  session logs uploaded; folders reorganized.
- **3–14 Jun 2025.** Chunked conversation embeddings with overlap and RAG as an agent tool (`rag_search`),
  prompt caching, clearer script descriptions, **assistant and auto modes**, one action per turn,
  re-analysis of historical plots, cleaner CLI, environment template.
- **16 Jun 2025.** `.env` loading fix and a short auto-mode test session on the setup: one galvo scan completed;
  the model then wrote invented find_nv/optimize/ESR output into its own replies, and the real optimize and ESR
  runs failed.
