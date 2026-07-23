import pandas as pd

from cohortcoder.selective_policy import (
    apply_frozen_threshold,
    one_sided_binomial_lower_bound,
    select_threshold_by_accuracy_lower_bound,
)


def test_one_sided_lower_bound_behaves_conservatively():
    assert one_sided_binomial_lower_bound(0, 20) == 0.0
    assert one_sided_binomial_lower_bound(100, 100) > 0.95
    assert one_sided_binomial_lower_bound(95, 100) < 0.95


def test_threshold_requires_lower_bound_not_only_empirical_accuracy():
    # Lowest threshold has 95% empirical accuracy but insufficient lower bound.
    rows = []
    for i in range(100):
        rows.append({"confidence": 0.5 + i / 1000, "correct": 0 if i < 5 else 1})
    frame = pd.DataFrame(rows)
    selection = select_threshold_by_accuracy_lower_bound(
        frame, target_accuracy=0.95, alpha=0.05, min_auto=20
    )
    assert selection.threshold is not None
    assert selection.one_sided_lower_bound >= 0.95
    assert selection.n_auto < 100


def test_no_feasible_policy_disables_auto_instead_of_relaxing_target():
    frame = pd.DataFrame(
        [{"confidence": 0.9 - i / 1000, "correct": int(i % 10 != 0)} for i in range(40)]
    )
    selection = select_threshold_by_accuracy_lower_bound(
        frame, target_accuracy=0.99, alpha=0.05, min_auto=20
    )
    assert selection.threshold is None
    frozen = apply_frozen_threshold(frame, selection.threshold)
    assert frozen["n_auto"] == 0
    assert frozen["coverage"] == 0.0
