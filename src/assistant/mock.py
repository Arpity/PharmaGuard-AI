"""Demo-mode answer writer: a deterministic template that turns the Python analysis and the
retrieved knowledge into the same style of answer an LLM would give. No numbers are computed here."""
from __future__ import annotations

from .intents import sections_for
from .knowledge import Hit


def _bullets(text: str) -> list[str]:
    return [l[2:].strip() for l in text.splitlines() if l.startswith("- ")]


def build_demo_answer(analysis: dict, hits: list[Hit], intents: list[str]) -> str:
    b, h, dv = analysis["batch"], analysis["history"], analysis["deviations"]
    oos, abn = analysis["out_of_spec_parameters"], analysis["abnormal_parameters"]
    params = {p["parameter"]: p for p in analysis["parameters"]}
    lines = []

    head = (f"**Summary** - Batch {b['Batch_ID']} ({b['Product_Name']}, {b['Plant']}, {b['Operator_Shift']} shift, "
            f"{b['Manufacturing_Date']}) is recorded as **{b['Quality_Status']}**. ")
    head += (f"{len(oos)} attribute(s) are outside specification ({', '.join(oos)})." if oos
             else "No attribute is outside its specification limits.")
    if abn:
        head += f" Abnormal signals: {', '.join(abn)}."
    lines.append(head)

    for sec in sections_for(intents):
        if sec == "why_risk":
            lines.append("\n**Why this batch shows risk**")
            lines += [f"- {i}" for i in analysis["risk_indicators"]] or \
                     ["- No risk indicator was triggered by the deterministic checks."]
        elif sec == "abnormal":
            lines.append("\n**Abnormal parameters**")
            if not abn:
                lines.append("- None: all measured parameters are within specification and within normal range for this product.")
            for name in abn:
                if name == "Deviation_Count":
                    lines.append(f"- Deviation count: {dv['count']} (peer average {dv['peer_mean']}, higher than "
                                 f"{dv['percentile']}% of peer batches).")
                    continue
                p = params[name]
                z = f", z-score {p['z_score']} vs {p['peer_n']} peer batches" if p["z_score"] is not None else ""
                lim = f" - {p['breach']} the limit by {p['breach_amount']}" if p["breach"] else ""
                lines.append(f"- {name}: {p['value_text']} (specification {p['spec_text']}){lim}{z}; flag: {p['flag']}.")
        elif sec == "compare":
            lines.append("\n**Comparison with historical batches**")
            lines.append(f"- Peer group: {h['peer_group']} ({h['peer_batches']} other batches); their failure rate is "
                         f"{h['peer_fail_rate_pct']}% and risk rate {h['peer_risk_rate_pct']}% (all products: "
                         f"{h['overall_fail_rate_pct']}% failure).")
            if h["similar_fail_rate_pct"] is not None:
                near = ", ".join(f"{c['Batch_ID']} ({c['Quality_Status']})" for c in h["closest_batches"][:3])
                lines.append(f"- Of the {h['similar_batches_used']} most similar historical batches, "
                             f"{h['similar_fail_rate_pct']}% failed. Closest: {near}.")
            for o in h["out_of_spec_history"]:
                lines.append(f"- Historically {o['batches_out_of_spec']} batches were out of specification for "
                             f"{o['parameter']}; {o['fail_rate_pct']}% of them failed.")
        elif sec == "procedure":
            lines.append("\n**What QA should review**")
            if not hits:
                lines.append("- No relevant procedure was retrieved from the knowledge base.")
            for i, hit in enumerate(hits[:4], 1):
                for bl in _bullets(hit.chunk.text)[:2]:
                    lines.append(f"- {bl} [K{i}]")

    gaps = analysis["data_notes"]
    notes = []
    if gaps["missing_critical_results"]:
        notes.append(f"critical results not available: {', '.join(gaps['missing_critical_results'])}")
    if gaps["imputed_parameters"]:
        notes.append(f"values estimated during data cleaning (confirm against batch record): {', '.join(gaps['imputed_parameters'])}")
    if notes:
        lines.append("\n**Data gaps** - " + "; ".join(notes) + ".")
    lines.append("\n_Decision support only: final batch disposition rests with authorised QA personnel._")
    return "\n".join(lines)
