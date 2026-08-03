from __future__ import annotations

import argparse
from pathlib import Path

from common.runner import run_classifier
from layer3.postprocessing import finalize_layer3


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")

DEFAULT_TEXT_COLUMN = "text_analyzed"
DEFAULT_ID_COLUMN = "review_id"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa a camada 3 de intensidade/toxicidade em um CSV."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Caminho para o arquivo de configuração YAML.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Caminho do CSV de entrada.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Diretório onde os resultados serão salvos.",
    )

    parser.add_argument(
        "--text-column",
        type=str,
        default=DEFAULT_TEXT_COLUMN,
        help=(
            "Coluna que contém o texto a ser classificado. "
            f"Padrão: {DEFAULT_TEXT_COLUMN}"
        ),
    )

    parser.add_argument(
        "--id-column",
        type=str,
        default=DEFAULT_ID_COLUMN,
        help=(
            "Coluna usada como identificador de cada comentário. "
            f"Padrão: {DEFAULT_ID_COLUMN}"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Quantidade máxima de registros a processar.",
    )

    parser.add_argument(
        "--gate-column",
        type=str,
        help="Coluna usada como filtro de entrada para a camada.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescreve resultados existentes.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    predictions, model = run_classifier(
        layer_name="layer3",
        config_path=args.config,
        input_override=args.input,
        output_override=args.output_dir,
        text_column_override=args.text_column,
        id_column_override=args.id_column,
        sample_limit_override=args.limit,
        gate_column_override=args.gate_column,
        overwrite=args.overwrite,
        finalizer=finalize_layer3,
    )

    print(f"Camada 3 concluída: {predictions}")
    print(
        f"Modelo: {model.model_name} | "
        f"dispositivo: {model.device}"
    )


if __name__ == "__main__":
    main()