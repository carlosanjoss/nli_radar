from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"O arquivo YAML precisa conter um objeto na raiz: {path}")
    return data


def project_root_from_config(config_path: Path) -> Path:
    """Os arquivos de configuração ficam em <raiz>/<camada>/config.yaml."""
    return config_path.resolve().parent.parent


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()

