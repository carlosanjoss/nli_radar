from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


TEXT_CANDIDATES = (
    "cleaned_body",
    "body",
    "text",
    "texto",
    "comment",
    "comentario",
    "comentário",
    "content",
)

ID_CANDIDATES = (
    "clean_id",
    "comment_id",
    "id",
    "external_id",
    "row_id",
)


@dataclass(frozen=True)
class LoadedInput:
    dataframe: pd.DataFrame
    text_column: str
    id_column: str | None
    original_rows: int
    empty_rows_removed: int
    gate_rows_removed: int
    encoding: str


def _read_csv(path: Path) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            dataframe = pd.read_csv(path, sep=None, engine="python", encoding=encoding)
            if len(dataframe.columns) == 1:
                # O Sniffer pode falhar em arquivos pequenos. Testa separadores usuais.
                for separator in (",", ";", "\t"):
                    candidate = pd.read_csv(path, sep=separator, encoding=encoding)
                    if len(candidate.columns) > len(dataframe.columns):
                        dataframe = candidate
            return dataframe, encoding
        except Exception as exc:  # tenta a próxima codificação
            errors.append(f"{encoding}: {exc}")
    raise ValueError(
        f"Não foi possível ler o CSV {path}. Tentativas: " + " | ".join(errors)
    )


def _detect_column(columns: Iterable[str], requested: str | None, candidates: tuple[str, ...]) -> str | None:
    available = list(columns)
    if requested and requested.lower() != "auto":
        if requested not in available:
            raise ValueError(
                f"Coluna '{requested}' não existe. Colunas disponíveis: {available}"
            )
        return requested

    by_lower = {column.lower(): column for column in available}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def _normalize_gate_values(values: Iterable[object]) -> set[str]:
    return {str(value).strip().lower() for value in values}


def load_input_csv(
    path: Path,
    text_column: str | None = "auto",
    id_column: str | None = "auto",
    sample_limit: int | None = None,
    gate_column: str | None = None,
    gate_positive_values: Iterable[object] = (1, True, "hate", "sim", "positivo"),
) -> LoadedInput:
    if not path.exists():
        raise FileNotFoundError(
            f"CSV não encontrado: {path}. Coloque o arquivo na pasta data ou informe --input."
        )

    dataframe, encoding = _read_csv(path)
    original_rows = len(dataframe)

    detected_text = _detect_column(dataframe.columns, text_column, TEXT_CANDIDATES)
    if detected_text is None:
        raise ValueError(
            "Não foi possível detectar a coluna de texto. Informe --text-column. "
            f"Colunas disponíveis: {list(dataframe.columns)}"
        )
    detected_id = _detect_column(dataframe.columns, id_column, ID_CANDIDATES)

    gate_rows_removed = 0
    if gate_column:
        if gate_column not in dataframe.columns:
            raise ValueError(
                f"A coluna de gate '{gate_column}' não existe no CSV. "
                f"Colunas disponíveis: {list(dataframe.columns)}"
            )
        positive = _normalize_gate_values(gate_positive_values)
        mask = dataframe[gate_column].map(lambda value: str(value).strip().lower() in positive)
        gate_rows_removed = int((~mask).sum())
        dataframe = dataframe.loc[mask].copy()

    text_series = dataframe[detected_text].fillna("").astype(str).str.strip()
    nonempty_mask = text_series.str.len() >= 5
    empty_rows_removed = int((~nonempty_mask).sum())
    dataframe = dataframe.loc[nonempty_mask].copy()
    dataframe["_text"] = text_series.loc[nonempty_mask]

    if sample_limit is not None:
        if sample_limit <= 0:
            raise ValueError("sample_limit precisa ser maior que zero.")
        dataframe = dataframe.head(sample_limit).copy()

    dataframe["_source_row_index"] = dataframe.index
    dataframe.reset_index(drop=True, inplace=True)
    if dataframe.empty:
        raise ValueError(
            "Nenhuma linha válida restou após os filtros de gate, texto vazio e limite."
        )
    dataframe["_row_id"] = range(1, len(dataframe) + 1)
    if detected_id:
        dataframe["_source_id"] = dataframe[detected_id].astype(str)
    else:
        dataframe["_source_id"] = dataframe["_source_row_index"].astype(str)

    return LoadedInput(
        dataframe=dataframe,
        text_column=detected_text,
        id_column=detected_id,
        original_rows=original_rows,
        empty_rows_removed=empty_rows_removed,
        gate_rows_removed=gate_rows_removed,
        encoding=encoding,
    )
