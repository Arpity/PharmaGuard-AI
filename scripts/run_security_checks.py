"""Entry point: python scripts/run_security_checks.py [--audit]   (--audit also runs pip-audit; needs network)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT  # noqa: E402
from src.security import controls, dependencies  # noqa: E402

if __name__ == "__main__":
    if "--audit" in sys.argv:
        controls.save_pip_audit(dependencies.run_pip_audit(PROJECT_ROOT / "requirements.lock"))
    cs = controls.run_controls(pip_audit=controls.load_pip_audit())
    for c in cs:
        print(f"{c.status.upper():5s} {c.id} {c.name} - {c.evidence}")
    print(controls.summary(cs))
    sys.exit(1 if any(c.status == "fail" for c in cs) else 0)
