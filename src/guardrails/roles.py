"""Role / permission concepts (demo). In production these come from SSO / IAM, not a dropdown."""
from __future__ import annotations

ROLES = {
    "viewer": "Viewer",
    "quality_analyst": "Quality Analyst",
    "qa_reviewer": "QA Reviewer (authorized Quality)",
    "compliance_admin": "Compliance / IT Admin",
}
PERMISSIONS = {
    "viewer": {"view_runs", "view_governance"},
    "quality_analyst": {"view_runs", "view_governance", "ask_assistant"},
    "qa_reviewer": {"view_runs", "view_governance", "ask_assistant", "review_ai_finding", "export_governance_report", "view_observability"},
    # Admins run security scans and export reports; they cannot ask the AI or review findings (separation of duties).
    "compliance_admin": {"view_runs", "view_governance", "run_security_scan", "export_governance_report", "view_observability",
                         "generate_demo_traces"},
}
# Nobody - human or AI - holds these here. Batch disposition happens outside this application.
FORBIDDEN_FOR_AI = {"approve_batch", "reject_batch", "release_batch", "disposition_batch", "modify_batch_data"}


class PermissionDenied(PermissionError):
    pass


def can(role: str, permission: str) -> bool:
    return permission in PERMISSIONS.get(role, set())


def require(role: str, permission: str) -> None:
    if not can(role, permission):
        raise PermissionDenied(f"Role '{ROLES.get(role, role)}' is not permitted to {permission.replace('_', ' ')}.")
