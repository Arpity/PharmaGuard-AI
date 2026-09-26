"""Production-traffic simulation for the monitoring demo. Everything goes through the real pipeline / stores and is flagged
synthetic, so it can be excluded from dashboards."""
from __future__ import annotations

import numpy as np

from src.assistant.pipeline import investigate
from src.observability.demo import generate_demo_traffic
from src.observability.recorder import RequestContext

INCIDENT_WEIGHTS = {"normal_demo": .20, "live_llm_ok": .08, "injection": .22, "llm_failure": .15, "unsafe_llm_reply": .12,
                    "retrieval_failure": .08, "internal_error": .07, "unauthorized": .08}


def simulate_production(recorder, df, kb, cfg, n: int = 200, incident: bool = False, seed: int = 11, days: float = 3.0) -> dict:
    """Steady-state traffic over the last `days`, plus (optionally) an incident burst in the last 45 minutes:
    prompt-injection spree, LLM provider failures, unsafe model replies, retrieval failures, internal errors, access denials."""
    counts = generate_demo_traffic(n, df, kb, cfg, recorder, seed=seed, days=days)
    if incident:
        burst = generate_demo_traffic(max(n // 5, 30), df, kb, cfg, recorder, seed=seed + 1, weights=INCIDENT_WEIGHTS, window_hours=0.75)
        for k, v in burst.items():
            counts[f"incident:{k}"] = v
    return counts


def seed_demo_reviews(review_store, df, kb, cfg, n: int = 20, seed: int = 5) -> dict:
    """Persist n demo AI runs (user 'Demo Analyst') and record human decisions on ~70% of them ('Demo Reviewer').
    These appear in the QA Review queue and are clearly named demo entries."""
    rng = np.random.default_rng(seed)
    ids = df["Batch_ID"].sample(n, random_state=seed).tolist()
    out = {"runs": 0, "approved": 0, "rejected": 0, "more_analysis": 0}
    for bid in ids:
        inv = investigate("Investigate this batch.", bid, df, kb, cfg, None, None,
                          RequestContext(user="Demo Analyst", role="quality_analyst", store=review_store, synthetic=True))
        if not inv.found:
            continue
        out["runs"] += 1
        if rng.random() < 0.7:
            d = str(rng.choice(["approved", "rejected", "more_analysis"], p=[.7, .15, .15]))
            review_store.add_review(inv.run_id, "Demo Reviewer", "qa_reviewer", d, "Demo review (simulated)")
            out[d] += 1
    return out
