"""Hand-made cases for evaluate.py. Run: python -m pytest test_evaluate.py -q"""
import pandas as pd
import pytest

from evaluate import candidates_to_predictions, cluster_bootstrap_ci, mcnemar, per_code_scores, summarise


@pytest.fixture
def gold():
    return pd.DataFrame({
        "source_code": ["LAA00", "LAA01", "LBC20", "MAB20", "TND11"],
        "chapter": ["L", "L", "L", "M", "T"],
        "group": ["LAA", "LAA", "LBC", "MAB", "TND"],
        "mapping_type": ["1:1", "1:1", "1:n", "unanswerable", "1:n"],
        "index_targets": [{"1"}, {"2"}, {"3", "4"}, set(), {"5", "6", "7"}],
    })


def row(scores, code):
    return scores.set_index("source_code").loc[code]


def test_single_correct_prediction(gold):
    r = row(per_code_scores({"LAA00": ["1"]}, gold), "LAA00")
    assert (r.hit, r.precision, r.recall, r.f1, r.complete) == (1, 1, 1, 1, 1)


def test_wrong_prediction(gold):
    r = row(per_code_scores({"LAA01": ["999"]}, gold), "LAA01")
    assert (r.hit, r.precision, r.recall, r.f1, r.complete) == (0, 0, 0, 0, 0)


def test_one_of_two_targets(gold):
    r = row(per_code_scores({"LBC20": ["3"]}, gold), "LBC20")
    assert r.hit == 1 and r.precision == 1 and r.recall == 0.5 and r.complete == 0
    assert r.f1 == pytest.approx(2 / 3)


def test_both_targets_with_extra(gold):
    r = row(per_code_scores({"LBC20": ["4", "3", "999"]}, gold), "LBC20")
    assert r.complete == 1 and r.recall == 1 and r.precision == pytest.approx(2 / 3)


def test_unanswerable_is_a_miss_even_with_prediction(gold):
    r = row(per_code_scores({"MAB20": ["123"]}, gold), "MAB20")
    assert (r.hit, r.recall, r.complete) == (0, 0, 0)


def test_missing_prediction_counts_as_miss(gold):
    scores = per_code_scores({}, gold)
    assert len(scores) == 5 and scores.hit.sum() == 0


def test_duplicates_are_ignored(gold):
    r = row(per_code_scores({"TND11": ["5", "5", "5"]}, gold), "TND11")
    assert r.precision == 1 and r.recall == pytest.approx(1 / 3)


def test_unknown_code_raises(gold):
    with pytest.raises(ValueError):
        per_code_scores({"ZZZ99": ["1"]}, gold)


def test_denominator_includes_every_gold_code(gold):
    summary = summarise(per_code_scores({"LAA00": ["1"], "LBC20": ["3"]}, gold), ci=False)
    assert summary.loc["all", "n"] == 5
    assert summary.loc["all", "hit"] == pytest.approx(2 / 5)
    assert summary.loc["all", "recall"] == pytest.approx(1.5 / 5)


def test_top_k_predictions():
    cands = pd.DataFrame({"source_code": ["A"] * 3, "rank": [2, 1, 3], "concept_id": ["y", "x", "z"]})
    assert candidates_to_predictions(cands, 2) == {"A": ["x", "y"]}


def test_bootstrap_ci_brackets_mean(gold):
    scores = per_code_scores({"LAA00": ["1"], "LBC20": ["3"]}, gold)
    low, high = cluster_bootstrap_ci(scores, "hit", n_boot=500)
    assert low <= scores.hit.mean() <= high


def test_paired_bootstrap_difference(gold):
    from evaluate import paired_bootstrap
    a = per_code_scores({"LAA00": ["1"], "LAA01": ["2"]}, gold)   # 2 of 5 correct
    b = per_code_scores({"LAA00": ["1"]}, gold)                   # 1 of 5 correct
    result = paired_bootstrap(a, b, n_boot=500)
    assert result["diff"] == pytest.approx(0.2)
    assert result["low"] <= result["diff"] <= result["high"]
    assert result["n_groups"] == 4  # LAA, LBC, MAB, TND


def test_paired_bootstrap_identical_systems(gold):
    from evaluate import paired_bootstrap
    a = per_code_scores({"LAA00": ["1"]}, gold)
    result = paired_bootstrap(a, a, n_boot=200)
    assert result["diff"] == 0 and result["low"] == 0 and result["high"] == 0 and not result["excludes_zero"]


def test_mcnemar_counts_discordant_pairs(gold):
    a = per_code_scores({"LAA00": ["1"], "LAA01": ["2"]}, gold)
    b = per_code_scores({"LAA00": ["1"]}, gold)
    result = mcnemar(a, b)
    assert (result["only_a"], result["only_b"]) == (1, 0)
