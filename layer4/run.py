from __future__ import annotations

import argparse
from pathlib import Path

from common.runner import run_classifier


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")

DEFAULT_TEXT_COLUMN = "text_analyzed"
DEFAULT_ID_COLUMN = "review_id"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa a camada 4 de dimensões discursivas em um CSV."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Caminho para o arquivo config.yaml da camada 4.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Caminho do CSV de entrada. "
            "Quando omitido, será usado o caminho definido no config.yaml."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Diretório de saída. "
            "Quando omitido, será usado o diretório definido no config.yaml."
        ),
    )

    parser.add_argument(
        "--text-column",
        type=str,
        default=DEFAULT_TEXT_COLUMN,
        help=(
            "Coluna que contém o texto analisado. "
            f"Padrão: {DEFAULT_TEXT_COLUMN}"
        ),
    )

    parser.add_argument(
        "--id-column",
        type=str,
        default=DEFAULT_ID_COLUMN,
        help=(
            "Coluna usada como identificador da amostra. "
            f"Padrão: {DEFAULT_ID_COLUMN}"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita a quantidade de registros processados.",
    )

    parser.add_argument(
        "--gate-column",
        type=str,
        default=None,
        help=(
            "Coluna opcional usada para filtrar quais registros "
            "devem passar pela camada 4."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescreve os resultados existentes.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    predictions, model = run_classifier(
        layer_name="layer4",
        config_path=args.config,
        input_override=args.input,
        output_override=args.output_dir,
        text_column_override=args.text_column,
        id_column_override=args.id_column,
        sample_limit_override=args.limit,
        gate_column_override=args.gate_column,
        overwrite=args.overwrite,
    )

    print()
    print(f"Camada 4 concluída: {predictions}")
    print(
        f"Modelo: {model.model_name} | "
        f"dispositivo: {model.device}"
    )


if __name__ == "__main__":
    main()