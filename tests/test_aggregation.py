from common.aggregation import aggregate_hypothesis_scores, weighted_max_mean


def test_weighted_max_mean() -> None:
    assert weighted_max_mean([0.2, 0.8], 0.7, 0.3) == 0.7 * 0.8 + 0.3 * 0.5


def test_counter_hypothesis_reduces_score() -> None:
    rows = [
        {
            "dimension_code": "d1",
            "construct_code": "c1",
            "hypothesis_kind": "positive",
            "label_probability": 0.90,
        },
        {
            "dimension_code": "d1",
            "construct_code": "c1",
            "hypothesis_kind": "negative",
            "label_probability": 0.80,
        },
    ]
    dimensions, constructs = aggregate_hypothesis_scores(
        rows,
        {
            "max_weight": 0.7,
            "mean_weight": 0.3,
            "counter_weight": 0.5,
            "construct_max_weight": 0.7,
            "construct_mean_weight": 0.3,
        },
    )
    assert round(constructs["d1.c1"], 6) == 0.5
    assert round(dimensions["d1"], 6) == 0.5

