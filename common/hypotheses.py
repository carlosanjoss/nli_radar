from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.config import load_yaml


VALID_KINDS = {"positive", "negative", "neutral"}


@dataclass(frozen=True)
class Hypothesis:
    dimension_code: str
    dimension_name: str
    construct_code: str
    construct_name: str
    hypothesis_code: str
    hypothesis_kind: str
    text: str


def load_hypotheses(path: Path) -> list[Hypothesis]:
    raw = load_yaml(path)
    dimensions = raw.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError(f"Nenhuma dimensão encontrada em {path}")

    flattened: list[Hypothesis] = []
    seen_codes: set[tuple[str, str, str]] = set()
    for dimension in dimensions:
        dimension_code = str(dimension["code"])
        dimension_name = str(dimension["name"])
        constructs = dimension.get("constructs", [])
        if not constructs:
            raise ValueError(f"Dimensão sem construtos: {dimension_code}")

        for construct in constructs:
            construct_code = str(construct["code"])
            construct_name = str(construct["name"])
            entries = construct.get("hypotheses", [])
            if not entries:
                raise ValueError(
                    f"Construto sem hipóteses: {dimension_code}/{construct_code}"
                )

            for entry in entries:
                hypothesis_code = str(entry["code"])
                kind = str(entry.get("kind", "positive")).lower()
                if kind not in VALID_KINDS:
                    raise ValueError(
                        f"Tipo de hipótese inválido em {dimension_code}/{construct_code}/{hypothesis_code}: {kind}"
                    )
                unique_key = (dimension_code, construct_code, hypothesis_code)
                if unique_key in seen_codes:
                    raise ValueError(f"Código de hipótese duplicado: {unique_key}")
                seen_codes.add(unique_key)

                text = str(entry["text"]).strip()
                if not text:
                    raise ValueError(f"Hipótese vazia: {unique_key}")
                flattened.append(
                    Hypothesis(
                        dimension_code=dimension_code,
                        dimension_name=dimension_name,
                        construct_code=construct_code,
                        construct_name=construct_name,
                        hypothesis_code=hypothesis_code,
                        hypothesis_kind=kind,
                        text=text,
                    )
                )

    return flattened

