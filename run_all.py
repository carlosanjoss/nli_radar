from __future__ import annotations

import argparse
from pathlib import Path

from common.config import load_yaml
from common.nli import NLIModel
from common.runner import run_classifier
from evaluation.compare_layers import compare_layer_outputs
from layer3.postprocessing import finalize_layer3


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa as camadas 3 e 4 separadamente e gera a verificação cruzada."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "reddit_tupye_2500_psicologia_sem_gabarito.csv",
    )
    parser.add_argument("--text-column")
    parser.add_argument("--id-column")
    parser.add_argument("--gate-column")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    layer3_config_path = ROOT / "layer3" / "config.yaml"
    layer4_config_path = ROOT / "layer4" / "config.yaml"
    shared_model = NLIModel(load_yaml(layer3_config_path)["model"])

    layer3_predictions, _ = run_classifier(
        layer_name="layer3",
        config_path=layer3_config_path,
        input_override=args.input,
        text_column_override=args.text_column,
        id_column_override=args.id_column,
        sample_limit_override=args.limit,
        gate_column_override=args.gate_column,
        overwrite=args.overwrite,
        finalizer=finalize_layer3,
        nli_model=shared_model,
    )
    layer4_predictions, _ = run_classifier(
        layer_name="layer4",
        config_path=layer4_config_path,
        input_override=args.input,
        text_column_override=args.text_column,
        id_column_override=args.id_column,
        sample_limit_override=args.limit,
        gate_column_override=args.gate_column,
        overwrite=args.overwrite,
        nli_model=shared_model,
    )
    comparison, summary = compare_layer_outputs(
        layer3_predictions,
        layer4_predictions,
        ROOT / "outputs" / "evaluation",
    )
    print(f"Camada 3: {layer3_predictions}")
    print(f"Camada 4: {layer4_predictions}")
    print(f"Verificação cruzada: {comparison}")
    print(f"Síntese: {summary}")


if __name__ == "__main__":
    main()

