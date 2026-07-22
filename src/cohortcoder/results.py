from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ResultsContract:
    external_human_reference: bool
    group_disjoint_test: bool
    no_test_terminology_leakage: bool
    no_test_tuning: bool
    non_synthetic: bool
    provenance_recorded: bool

    @property
    def reportable(self) -> bool:
        return all(asdict(self).values())

    @property
    def status(self) -> str:
        if self.reportable:
            return "reportable"
        if not self.non_synthetic:
            return "synthetic_smoke_test"
        if not self.no_test_terminology_leakage:
            return "oracle_diagnostic"
        return "non_reportable"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["reportable"] = self.reportable
        out["status"] = self.status
        return out


def build_results_contract(metadata: Mapping[str, Any]) -> ResultsContract:
    """Conservatively decide whether a benchmark result is reportable.

    Missing audit fields default to False, so omitted metadata can never make a
    result reportable by accident.
    """
    return ResultsContract(
        external_human_reference=bool(metadata.get("external_human_reference", False)),
        group_disjoint_test=bool(metadata.get("group_disjoint_test", False)),
        no_test_terminology_leakage=bool(metadata.get("no_test_terminology_leakage", False)),
        no_test_tuning=bool(metadata.get("no_test_tuning", False)),
        non_synthetic=bool(metadata.get("non_synthetic", False)),
        provenance_recorded=bool(metadata.get("provenance_recorded", False)),
    )
