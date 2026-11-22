"""Baseline policies. Each policy maps observations (JSON) to actions (JSON) and never sees the env."""
from .base import Policy
from .random_explorer import RandomExplorer
from .reconciliation import SimpleReconciliation
from .flag_all import FlagAll
from .oracle import OracleAdapter, oracle_findings
from .scripted_auditor import ScriptedAuditor
from .boundary_guess import BoundaryGuesser

BASELINES = {
    "random": RandomExplorer,
    "reconciliation": SimpleReconciliation,
    "flag_all": FlagAll,
    "oracle": OracleAdapter,
}
ALL_POLICIES = dict(BASELINES, scripted_auditor=ScriptedAuditor, boundary_guess=BoundaryGuesser)

__all__ = ["Policy", "RandomExplorer", "SimpleReconciliation", "FlagAll", "OracleAdapter", "ScriptedAuditor",
           "BoundaryGuesser",
           "oracle_findings", "BASELINES", "ALL_POLICIES"]
