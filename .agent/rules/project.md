# MedDriftGuard project rules
- Project: multi-turn, Hinglish-capable medical RAG assistant for general health INFORMATION (never diagnosis, never dosing).
- Language: Python 3.11+, type hints, ruff + black, pytest. Config in YAML under configs/. No hardcoded paths or API keys (use .env).
- Architecture is fixed: Triage Gate -> Normalizer/Rewriter -> Patient-State Ledger -> Hybrid Retriever -> Evidence-bound Generator -> Claim Verifier. Do not merge the Ledger and the evidence index. The retriever must NEVER receive raw chat history, only the standalone query.
- Safety: emergency detection is deterministic code + a small classifier, never left to the generator LLM. Every answer must carry citations to KB chunk IDs, a "when to see a doctor" line, and a disclaimer. If evidence is insufficient, abstain.
- Training data must contain NO drug doses, lab thresholds or other hard medical facts. We fine-tune behaviour/format/language only.
- Privacy: no real patient data, no PII logging. Synthetic data only. Log only turn IDs and metrics.
- Reproducibility: fixed seeds, every experiment writes results/<exp_id>/metrics.json + config.yaml. Never overwrite old results.
- Every phase ends with: tests passing, a short docs/phase_X.md, and a git commit message "phase X: ...".
- Before coding a phase, produce a plan artifact; list files you will create/modify; ask before deleting anything.
