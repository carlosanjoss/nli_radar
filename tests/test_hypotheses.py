from pathlib import Path

from common.hypotheses import load_hypotheses


ROOT = Path(__file__).resolve().parents[1]


def test_layer3_hypothesis_count() -> None:
    hypotheses = load_hypotheses(ROOT / "layer3" / "hypotheses.yaml")
    assert len(hypotheses) == 40
    assert len({(h.dimension_code, h.construct_code, h.hypothesis_code) for h in hypotheses}) == 40


def test_layer4_hypothesis_count() -> None:
    hypotheses = load_hypotheses(ROOT / "layer4" / "hypotheses.yaml")
    assert len(hypotheses) == 63
    assert len({(h.dimension_code, h.construct_code, h.hypothesis_code) for h in hypotheses}) == 63
