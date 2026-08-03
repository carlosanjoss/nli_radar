from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common.config import load_yaml, resolve_project_path
from common.hypotheses import load_hypotheses
from common.postgres import (
    PostgresStore,
    classifier_config_sha256,
    file_sha256,
    load_postgres_uri,
)
from common.runner import run_classifier
from layer3.postprocessing import finalize_layer3


ROOT = Path(__file__).resolve().parent
CONFIG_PATHS = {
    "layer3": ROOT / "layer3" / "config.yaml",
    "layer4": ROOT / "layer4" / "config.yaml",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Executa as hipóteses NLI sobre posts e comentários classificados como "
            "hate pelo Gemma e persiste as camadas 3 e 4 separadamente no PostgreSQL."
        )
    )
    parser.add_argument(
        "--postgres-uri",
        help=(
            "URI PostgreSQL. Prefira definir POSTGRES_URI no ambiente ou no .env "
            "para não expor a senha no histórico do terminal."
        ),
    )
    parser.add_argument(
        "--layer",
        choices=("both", "layer3", "layer4"),
        default="both",
        help="Camadas a executar. Padrão: both.",
    )
    parser.add_argument(
        "--platform",
        action="append",
        default=[],
        help="Filtra a plataforma de origem; pode ser informado mais de uma vez.",
    )
    parser.add_argument(
        "--content-kind",
        choices=("all", "comment", "post"),
        default="all",
        help="Tipo de conteúdo Gemma a processar. Padrão: all.",
    )
    parser.add_argument(
        "--gemma-model",
        help="Processa somente avaliações produzidas por este modelo Gemma.",
    )
    parser.add_argument(
        "--gemma-prompt-version",
        help="Processa somente avaliações desta versão de prompt do Gemma.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limita os conteúdos pendentes por camada (útil para teste).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "postgres",
        help="Diretório dos CSVs de auditoria; o PostgreSQL continua sendo a saída principal.",
    )
    parser.add_argument(
        "--migrate-only",
        action="store_true",
        help="Cria/atualiza as tabelas e encerra sem executar o modelo.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Mostra o progresso dos runs recentes e encerra.",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        help="Com --status, mostra somente este run_id.",
    )
    return parser


def _layers(selected: str) -> list[str]:
    return ["layer3", "layer4"] if selected == "both" else [selected]


def _content_kinds(selected: str) -> tuple[str, ...]:
    return ("comment", "post") if selected == "all" else (selected,)


def _safe_print(message: str) -> None:
    """Não interrompe a inferência se o terminal de origem for fechado."""
    try:
        print(message, flush=True)
    except (OSError, ValueError):
        return


def _show_status(
    store: PostgresStore, *, layer_name: str, run_id: int | None
) -> None:
    filters: list[str] = []
    params: list[object] = []
    if layer_name != "both":
        filters.append("layer_name = %s")
        params.append(layer_name)
    if run_id is not None:
        filters.append("run_id = %s")
        params.append(run_id)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    params.append(20)
    rows = store.conn.execute(
        f"""
        SELECT
            run_id, layer_name, status, total_selected, processed_count,
            started_at, finished_at, error_message
        FROM radar_nli_runs
        {where}
        ORDER BY run_id DESC
        LIMIT %s
        """,
        params,
    ).fetchall()
    if not rows:
        _safe_print("Nenhum run encontrado.")
        return
    for row in rows:
        total = int(row["total_selected"])
        processed = int(row["processed_count"])
        percentage = (processed / total * 100.0) if total else 100.0
        message = (
            f"run_id={row['run_id']} | {row['layer_name']} | "
            f"status={row['status']} | {processed}/{total} "
            f"({percentage:.2f}%) | início={row['started_at']}"
        )
        if row["finished_at"]:
            message += f" | fim={row['finished_at']}"
        if row["error_message"]:
            message += f" | erro={row['error_message']}"
        _safe_print(message)


def _run_layer(
    *,
    store: PostgresStore,
    layer_name: str,
    args: argparse.Namespace,
    shared_model: Any | None,
) -> tuple[Any | None, int | None]:
    config_path = CONFIG_PATHS[layer_name]
    config = load_yaml(config_path)
    hypotheses_path = resolve_project_path(ROOT, config["hypotheses_file"])
    hypotheses = load_hypotheses(hypotheses_path)
    dimension_codes = list(dict.fromkeys(item.dimension_code for item in hypotheses))
    hypotheses_hash = file_sha256(hypotheses_path)
    config_hash = classifier_config_sha256(config)
    model_name = str(config["model"]["name"])
    content_kinds = _content_kinds(args.content_kind)

    loaded = store.fetch_pending_hate_comments(
        layer_name=layer_name,
        nli_model_name=model_name,
        hypotheses_hash=hypotheses_hash,
        config_hash=config_hash,
        platforms=args.platform,
        gemma_model=args.gemma_model,
        gemma_prompt_version=args.gemma_prompt_version,
        limit=args.limit,
        content_kinds=content_kinds,
    )
    selected = len(loaded.dataframe)
    if selected == 0:
        _safe_print(f"{layer_name}: nenhum conteúdo hate pendente.")
        return shared_model, None

    run_id = store.create_run(
        layer_name=layer_name,
        nli_model_name=model_name,
        hypotheses_hash=hypotheses_hash,
        config_hash=config_hash,
        total_selected=selected,
        platforms=args.platform,
        gemma_model=args.gemma_model,
        gemma_prompt_version=args.gemma_prompt_version,
    )
    output_dir = args.output_dir.resolve() / layer_name / f"run_{run_id}"
    _safe_print(
        f"{layer_name}: run_id={run_id} | conteúdos hate pendentes={selected}"
    )

    try:
        if shared_model is not None and shared_model.model_name != model_name:
            shared_model = None
        persisted_total = 0

        def persist_batch(
            source_chunk: Any,
            predictions: list[dict[str, Any]],
            hypothesis_scores: list[dict[str, Any]],
        ) -> None:
            nonlocal persisted_total
            inserted = store.save_layer_batch(
                layer_name=layer_name,
                run_id=run_id,
                source_chunk=source_chunk,
                predictions=predictions,
                hypothesis_scores=hypothesis_scores,
                dimension_codes=dimension_codes,
                nli_model_name=model_name,
                hypotheses_hash=hypotheses_hash,
                config_hash=config_hash,
            )
            persisted_total += inserted

        predictions_path, shared_model = run_classifier(
            layer_name=layer_name,
            config_path=config_path,
            output_override=output_dir,
            overwrite=False,
            finalizer=finalize_layer3 if layer_name == "layer3" else None,
            nli_model=shared_model,
            loaded_input_override=loaded,
            input_source_description=(
                "PostgreSQL: conteúdo Gemma cuja classificação mais recente possui "
                "success e is_hate=true"
            ),
            batch_callback=persist_batch,
            show_progress=False,
        )
        store.finish_run(run_id, persisted_total)
        _safe_print(
            f"{layer_name}: concluída | salvos no PostgreSQL={persisted_total} | "
            f"auditoria={output_dir}"
        )
        return shared_model, run_id
    except Exception as exc:
        store.fail_run(run_id, str(exc))
        raise


def main() -> None:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit precisa ser maior que zero.")

    store = PostgresStore(load_postgres_uri(args.postgres_uri))
    try:
        if args.status:
            _show_status(store, layer_name=args.layer, run_id=args.run_id)
            return
        store.migrate()
        _safe_print("Migrações PostgreSQL aplicadas.")
        if args.migrate_only:
            return

        shared_model: Any | None = None
        failures: list[str] = []
        for layer_name in _layers(args.layer):
            try:
                shared_model, _ = _run_layer(
                    store=store,
                    layer_name=layer_name,
                    args=args,
                    shared_model=shared_model,
                )
            except Exception as exc:
                failures.append(f"{layer_name}: {exc}")
                _safe_print(f"{layer_name}: falhou: {exc}")
        if failures:
            raise RuntimeError(" | ".join(failures))
    finally:
        store.close()


if __name__ == "__main__":
    main()
