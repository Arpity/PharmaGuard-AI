"""Governance report: one structured document rendered to Markdown and standalone HTML."""
from __future__ import annotations

import html
from datetime import datetime, timezone
from dataclasses import dataclass, field

from src.security.controls import Control, summary


@dataclass
class Report:
    title: str
    blocks: list = field(default_factory=list)      # (kind, payload)

    def h(self, text, level=2): self.blocks.append(("h", (level, text)))
    def p(self, text): self.blocks.append(("p", text))
    def bullets(self, items): self.blocks.append(("ul", list(items)))
    def table(self, header, rows): self.blocks.append(("table", (list(header), [list(map(str, r)) for r in rows])))

    def markdown(self) -> str:
        out = [f"# {self.title}", ""]
        for kind, x in self.blocks:
            if kind == "h": out += [f"{'#' * x[0]} {x[1]}", ""]
            elif kind == "p": out += [x, ""]
            elif kind == "ul": out += [f"- {i}" for i in x] + [""]
            else:
                hd, rows = x
                esc = lambda s: s.replace("|", "\\|").replace("\n", " ")
                out += ["| " + " | ".join(hd) + " |", "|" + "---|" * len(hd)] + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows] + [""]
        return "\n".join(out)

    def html(self) -> str:
        e = html.escape
        body = [f"<h1>{e(self.title)}</h1>"]
        for kind, x in self.blocks:
            if kind == "h": body.append(f"<h{x[0]}>{e(x[1])}</h{x[0]}>")
            elif kind == "p": body.append(f"<p>{e(x)}</p>")
            elif kind == "ul": body.append("<ul>" + "".join(f"<li>{e(i)}</li>" for i in x) + "</ul>")
            else:
                hd, rows = x
                body.append("<table><thead><tr>" + "".join(f"<th>{e(c)}</th>" for c in hd) + "</tr></thead><tbody>"
                            + "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table>")
        css = ("body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#1e293b}"
               "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.9rem}th,td{border:1px solid #e2e8f0;padding:6px 8px;text-align:left;vertical-align:top}"
               "th{background:#f1f5f9}h1{border-bottom:2px solid #2563eb;padding-bottom:.3rem}")
        return f"<!doctype html><html><head><meta charset='utf-8'><title>{e(self.title)}</title><style>{css}</style></head><body>{''.join(body)}</body></html>"


def build_report(meta: dict, model: dict, evaluation: dict, fingerprints: list[dict], controls: list[Control],
                 env_rows: list[dict], generated_by: str = "", now: datetime | None = None) -> Report:
    s, now = meta["system"], now or datetime.now(timezone.utc)
    r = Report(f"AI Governance Report - {s['name']}")
    r.p(f"Version {s['version']} · generated {now:%Y-%m-%d %H:%M} UTC" + (f" by {generated_by}" if generated_by else "")
        + " · produced from configured project metadata and live checks.")
    r.h("1. Summary")
    cs = summary(controls)
    r.table(["Item", "Status"], [["Approval status", f"{meta['approval']['status']} ({meta['approval']['stage']})"],
                                 ["Evaluation", evaluation["status"] + (f" - overall score {evaluation['overall_score'] * 100:.1f}%, {evaluation['cases_passed']}/{evaluation['cases']} cases" if evaluation["available"] else "")],
                                 ["Security controls", f"{cs['pass']} pass, {cs['warn']} warning, {cs['fail']} fail, {cs['info']} not run (of {cs['total']})"],
                                 ["Mode", model["mode"]]])
    r.h("2. AI system")
    r.table(["Field", "Value"], [["AI use case", s["use_case"].strip()], ["Business owner", s["business_owner"]], ["Technical owner", s["technical_owner"]],
                                 ["Environment", s.get("environment", "")], ["Out of scope", "; ".join(s.get("out_of_scope", []))]])
    r.h("Intended users", 3)
    r.table(["Role", "Use"], [[u["role"], u["use"]] for u in s["intended_users"]])
    r.h("3. Data")
    r.table(["Source", "Description", "Classification", "Personal data"],
            [[d["name"], d["description"], d["classification"], "Yes" if d["personal_data"] else "No"] for d in meta["data_sources"]])
    r.p("Current classification: " + meta["data_classification"]["current"])
    r.p("If real data were used: " + meta["data_classification"]["if_real_data"])
    r.h("4. Model and prompt")
    r.table(["Field", "Value"], [["Current mode", model["mode"]], ["Model / LLM", model["model"]], ["Model version", model["model_version"]],
                                 ["Version pinned", model["pinned"]], ["Data leaves environment", model["data_leaves_environment"]],
                                 ["Demo mode", meta["model"]["demo_mode"]], ["Live mode", meta["model"]["live_mode"]],
                                 ["Prompt", f"{meta['prompt']['name']} v{meta['prompt']['version']}"], ["Prompt change control", meta["prompt"]["change_control"]]])
    r.h("Configuration fingerprints (drift detection)", 3)
    r.table(["Artifact", "Current", "Baseline", "Status"], [[f["artifact"], f["current"], f["baseline"], f["status"]] for f in fingerprints])
    r.h("5. Risks")
    r.table(["ID", "Risk", "Likelihood", "Impact", "Mitigation", "Residual"],
            [[k["id"], k["risk"], k["likelihood"], k["impact"], k["mitigation"], k["residual"]] for k in meta["risks"]])
    r.h("6. Known limitations")
    r.bullets(meta["limitations"])
    r.h("7. Guardrails")
    r.table(["Control", "Implementation", "Purpose"], [[g["control"], g["implementation"], g["purpose"]] for g in meta["guardrails"]])
    r.h("8. Human oversight")
    r.bullets(meta["human_oversight"])
    r.h("9. Evaluation status")
    if evaluation["available"]:
        m = evaluation["metrics"]
        r.p(f"Run {evaluation['generated_at']} UTC in {evaluation['mode']}; {evaluation['status']}.")
        r.table(["Metric", "Value", "Target met"], [[k.replace("_", " "), f"{m[k]:.3f}" if k != "latency_p95_ms" else f"{m[k]:.0f} ms", "yes" if ok else "NO"]
                                                    for k, ok in evaluation["target_status"].items()])
    else:
        r.p("No evaluation results available. Run scripts/run_evaluation.py.")
    r.h("10. Approval status")
    r.p(f"{meta['approval']['status']} - {meta['approval']['stage']}.")
    r.table(["Approver role", "Status", "Signature / date"], [[a["role"], a["status"], "________"] for a in meta["approval"]["approvals"]])
    r.p("Conditions for approval:")
    r.bullets(meta["approval"]["conditions"])
    r.h("11. Change history")
    r.table(["Version", "Date", "Category", "Change", "Owner"], [[c["version"], c["date"], c["category"], c["change"], c["owner"]] for c in meta["change_history"]])
    r.h("12. Security controls")
    r.table(["ID", "Category", "Control", "Status", "Evidence"], [[c.id, c.category, c.name, c.status.upper(), c.evidence] for c in controls])
    todo = [c for c in controls if c.status in ("warn", "fail") and c.remediation]
    if todo:
        r.h("Open security actions", 3)
        r.bullets([f"{c.id} {c.name}: {c.remediation}" for c in todo])
    r.h("Environment variables (values never shown)", 3)
    r.table(["Variable", "Secret", "Status", "Purpose"], [[e["variable"], "yes" if e["secret"] else "no", e["status"], e["purpose"]] for e in env_rows])
    r.h("13. Secret-management guidance")
    r.bullets(SECRET_GUIDANCE)
    return r


SECRET_GUIDANCE = [
    "Local development: keep secrets in a git-ignored .env file (copy from .env.example); never commit it or paste keys into code, notebooks or chat.",
    "Shared / CI environments: inject secrets from the platform secret store (for example GitHub Actions secrets) as environment variables.",
    "Production: use a managed vault or KMS (Azure Key Vault, AWS Secrets Manager, HashiCorp Vault) with workload identity, not long-lived keys on disk.",
    "Least privilege: separate keys per environment and per service, with spend limits and only the scopes needed.",
    "Rotate keys on a schedule and immediately if exposure is suspected; revoke first, then investigate.",
    "Never log, store or display secrets: logs are redacted and only hashes and lengths of free text are recorded.",
    "Run secret scanning and dependency audits in CI and before every release.",
    "Before enabling a live LLM with real data, complete the data-classification and data-processing review.",
]
