# Archived helper scripts

One-off or superseded tools, kept for reference. Not part of any live path.

- `clean_repo.sh` — early bootstrap hygiene; stale (would overwrite
  `.gitignore` and move `models/` under `runtime/`, contradicting the
  current layout). Do not run.
- `download_sequential.py` — legacy urllib shard downloader; superseded by
  `scripts/download_models.py` (huggingface_hub resume).
- `fetch_pe_t2i.sh` — one-off PE-T2I fetch; use `download_models.py --model`.
- `measure_heretic_peak.py` — one-off GB10 peak measurement (run A/B);
  results recorded in `modules/backends/image_21.py` comments and the
  calibrated memory table.
