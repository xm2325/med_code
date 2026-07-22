from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
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
    return ResultsContract(
        external_human_reference=bool(metadata.get("external_human_reference", False)),
        group_disjoint_test=bool(metadata.get("group_disjoint_test", False)),
        no_test_terminology_leakage=bool(metadata.get("no_test_terminology_leakage", False)),
        no_test_tuning=bool(metadata.get("no_test_tuning", False)),
        non_synthetic=bool(metadata.get("non_synthetic", False)),
        provenance_recorded=bool(metadata.get("provenance_recorded", False)),
    )


def contract_from_benchmark_metadata(*, external_human_reference: bool, group_disjoint_test: bool,
                                     candidate_dictionary_source: str, test_used_for_selection_or_tuning: bool,
                                     data_is_synthetic: bool, provenance_recorded: bool) -> ResultsContract:
    leakage_safe = candidate_dictionary_source not in {"all_gold_oracle", "dataset_derived_all_labels"}
    return ResultsContract(
        external_human_reference=bool(external_human_reference),
        group_disjoint_test=bool(group_disjoint_test),
        no_test_terminology_leakage=bool(leakage_safe),
        no_test_tuning=not bool(test_used_for_selection_or_tuning),
        non_synthetic=not bool(data_is_synthetic),
        provenance_recorded=bool(provenance_recorded),
    )


def write_results_contract(path: str | Path, contract: ResultsContract) -> dict[str, Any]:
    payload = contract.to_dict()
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
