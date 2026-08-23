from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path


RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_adecoding_pt_experiment.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("adecoding_pt_runner", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClient:
    def complete_json(self, messages, schema, max_tokens):
        payload = json.loads(messages[-1]["content"])
        properties = schema.get("properties", {})
        if "ranked_term_ids" in properties:
            return {"ranked_term_ids": properties["ranked_term_ids"]["items"]["enum"]}
        text = payload.get("clinical_text", "headache")
        item = properties["events"]["items"]["properties"]
        event = {"quote": text, "assertion": "affirmed"}
        if "term_ids" in item:
            event["term_ids"] = ["PTL0001"]
        return {"events": [event]}


def terms(module):
    return [
        module.Term("PTL0001", "headache", "", "UNKNOWN", "UNRESOLVED_NO_VERSIONED_PT_ASC"),
        module.Term("PTL0002", "nausea", "", "UNKNOWN", "UNRESOLVED_NO_VERSIONED_PT_ASC"),
    ]


def test_three_methods_are_independent_and_grounded():
    module = load_runner()
    experiment = module.ThreeMethodExperiment(FakeClient(), terms(module))
    prompt = experiment.prompt_only("headache")
    lexical = experiment.extract_lexical("headache")
    rag = experiment.hybrid_rag("headache")

    assert prompt["selected_pt_terms"] == ["headache"]
    assert lexical["selected_pt_terms"] == ["headache"]
    assert rag["selected_pt_terms"] == ["headache"]
    assert rag["candidates"][0]["reranked_candidates"]
    assert all(
        result["events"][0]["offset_source"] == "SERVER_UNIQUE_EXACT_QUOTE_MATCH"
        for result in (prompt, lexical, rag)
    )


def test_event_quote_must_be_a_unique_exact_substring():
    module = load_runner()
    accepted, errors = module.validate_events(
        "headache then headache",
        {"events": [{"quote": "headache", "assertion": "affirmed"}]},
    )
    assert accepted == []
    assert errors == ["EVENT_0_AMBIGUOUS_QUOTE"]


def test_frozen_rerank_cannot_change_candidate_set():
    module = load_runner()

    class FrozenChangingClient(FakeClient):
        def complete_json(self, messages, schema, max_tokens):
            properties = schema.get("properties", {})
            if "ranked_term_ids" in properties:
                allowed = properties["ranked_term_ids"]["items"]["enum"]
                return {"ranked_term_ids": [*allowed[:-1], "PTL9999"]}
            return super().complete_json(messages, schema, max_tokens)

    experiment = module.ThreeMethodExperiment(FrozenChangingClient(), terms(module))
    result = experiment.hybrid_rag("headache")
    assert result["execution_status"] == "VALIDATION_FAILED"
    assert result["events"][0]["selected"] is None
    assert "FROZEN_CANDIDATE_SET_CHANGED" in result["validation_errors"]


def test_public_scripts_contain_no_private_machine_identifiers():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "scripts" / "run_adecoding_pt_experiment.py",
        root / "scripts" / "prepare_adecoding_pt_inputs.py",
        root / "scripts" / "verify_adecoding_pt_outputs.py",
        root / "slurm" / "run_adecoding_pt_experiment.sbatch",
    ]
    forbidden = ("/Users/", "/Downloads/", "@roihu-gpu")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden), path
        assert re.search(r"project_[0-9]{7,}", text) is None, path


if __name__ == "__main__":
    test_three_methods_are_independent_and_grounded()
    test_event_quote_must_be_a_unique_exact_substring()
    test_frozen_rerank_cannot_change_candidate_set()
    test_public_scripts_contain_no_private_machine_identifiers()
    print("adecoding PT experiment tests passed")
