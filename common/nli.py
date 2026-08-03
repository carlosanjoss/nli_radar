from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class NLIScore:
    entailment_probability_3way: float
    neutral_probability_3way: float
    contradiction_probability_3way: float
    label_probability: float
    inference_time_seconds: float


class NLIModel:
    """Inferência NLI batelada com mapeamento explícito dos rótulos do checkpoint."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_name = str(config["name"])
        requested_device = str(config.get("device", "auto")).lower()
        self.device = self._resolve_device(requested_device)
        self.batch_size = int(config.get("batch_size", 8))
        self.max_length = int(config.get("max_length", 512))
        self.score_mode = str(
            config.get("score_mode", "entailment_vs_contradiction")
        )
        self.local_files_only = bool(config.get("local_files_only", False))
        self.use_amp = bool(config.get("use_amp", True)) and self.device.type == "cuda"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        self.model.eval()
        self.model.to(self.device)

        configured_ids = config.get("label_ids", {})
        self.contradiction_id = self._label_id("contradiction", configured_ids)
        self.neutral_id = self._label_id("neutral", configured_ids)
        self.entailment_id = self._label_id("entailment", configured_ids)

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA foi solicitada, mas não está disponível.")
        return torch.device(requested)

    def _label_id(self, label: str, configured_ids: dict[str, Any]) -> int:
        if label in configured_ids:
            return int(configured_ids[label])

        id2label = {
            int(index): str(value).lower()
            for index, value in self.model.config.id2label.items()
        }
        aliases = {
            "entailment": ("entailment", "entail", "implication"),
            "neutral": ("neutral",),
            "contradiction": ("contradiction", "contradict"),
        }
        for index, value in id2label.items():
            if any(alias in value for alias in aliases[label]):
                return index

        raise ValueError(
            f"O checkpoint não informou o índice de '{label}'. id2label={id2label}. "
            "Defina model.label_ids no YAML. Para joeddav/xlm-roberta-large-xnli, "
            "a ordem documentada é contradiction=0, neutral=1, entailment=2."
        )

    def score_pairs(self, premises: list[str], hypotheses: list[str]) -> list[NLIScore]:
        if len(premises) != len(hypotheses):
            raise ValueError("Premises e hypotheses precisam ter o mesmo tamanho.")
        output: list[NLIScore] = []

        for start in range(0, len(premises), self.batch_size):
            batch_premises = premises[start : start + self.batch_size]
            batch_hypotheses = hypotheses[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                padding=True,
                truncation="only_first",
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.use_amp
                else nullcontext()
            )
            started = perf_counter()
            with torch.inference_mode(), amp_context:
                logits = self.model(**encoded).logits
            elapsed_per_pair = (perf_counter() - started) / len(batch_premises)

            probabilities_3way = torch.softmax(logits.float(), dim=1)
            binary_logits = logits.float()[:, [self.contradiction_id, self.entailment_id]]
            probabilities_binary = torch.softmax(binary_logits, dim=1)

            for index in range(len(batch_premises)):
                entailment_3way = float(
                    probabilities_3way[index, self.entailment_id].cpu()
                )
                neutral_3way = float(
                    probabilities_3way[index, self.neutral_id].cpu()
                )
                contradiction_3way = float(
                    probabilities_3way[index, self.contradiction_id].cpu()
                )
                if self.score_mode == "entailment_vs_contradiction":
                    label_probability = float(probabilities_binary[index, 1].cpu())
                elif self.score_mode == "three_way_entailment":
                    label_probability = entailment_3way
                else:
                    raise ValueError(
                        "model.score_mode deve ser 'entailment_vs_contradiction' "
                        "ou 'three_way_entailment'."
                    )
                output.append(
                    NLIScore(
                        entailment_probability_3way=entailment_3way,
                        neutral_probability_3way=neutral_3way,
                        contradiction_probability_3way=contradiction_3way,
                        label_probability=label_probability,
                        inference_time_seconds=elapsed_per_pair,
                    )
                )
        return output

