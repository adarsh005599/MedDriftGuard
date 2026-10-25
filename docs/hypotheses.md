# MedDriftGuard: Pre-Registered Scientific Hypotheses

| ID | Hypothesis | Success Criterion | Target Metric |
|---|---|---|---|
| **H1** | In standard conversational RAG (B1), the unsupported-claim rate (UCR) rises steadily with turn depth. | Statistically significant positive slope of UCR vs. turn depth for B1 | UCR slope > 0 (p < 0.05) |
| **H2** | MedDriftGuard (B3) decouples patient-state from evidence retrieval, keeping UCR flat across turns. | Slope of UCR vs. turn depth not significantly different from 0; UCR at turn ≥ 8 is ≥ 50% lower than B1 | UCR at turn 8+ reduced by ≥ 50% |
| **H3** | The decoupled ledger + current-turn retrieval is the primary driver of drift mitigation, not the verifier alone. | Ablation B3 − noLedger accounts for the majority of the performance drop vs B3 | UCR degradation in B3 − noLedger |
| **H4** | A trained rewriter (M1) reduces the script gap for Roman-Hinglish queries compared to a generic prompted rewriter. | Retrieval Recall@5 gap (English − Roman Hinglish) is significantly narrower with M1 | Recall@5 Script Gap Reduction |
| **H5** | The deterministic red-flag triage gate reliably detects emergencies across English, Devanagari, and Roman Hindi. | Emergency recall ≥ 0.95 across each script; false-alarm rate ≤ 5% on benign messages | Recall ≥ 0.95, False Alarm ≤ 0.05 |
| **H6** | The claim verifier introduces safety gains at acceptable latency and abstention cost. | Abstention rate ≤ 15%; turn latency acceptable for local interaction | Abstention ≤ 15% |
