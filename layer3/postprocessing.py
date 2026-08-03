from __future__ import annotations

from statistics import mean
from typing import Any


def _contains_any(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _apply_lexical_boosts(
    text: str, scores: dict[str, float], config: dict[str, Any]
) -> dict[str, float]:
    rules = config.get("lexical_rules", {})
    if _contains_any(text, rules.get("sexual_violence", [])):
        scores["violence"] = max(scores.get("violence", 0.0), 0.95)
        scores["threat"] = max(scores.get("threat", 0.0), 0.75)
    if _contains_any(text, rules.get("physical_threat", [])):
        scores["threat"] = max(scores.get("threat", 0.0), 0.95)
        scores["violence"] = max(scores.get("violence", 0.0), 0.85)
    if _contains_any(text, rules.get("dehumanization", [])):
        scores["dehumanization"] = max(scores.get("dehumanization", 0.0), 0.90)
    return scores


def _block_false_positive_insult(text: str, scores: dict[str, float]) -> dict[str, float]:
    lowered = text.lower()
    neutral_patterns = [
        "não concordo", "nao concordo", "discordo", "não gosto", "nao gosto",
        "prefiro", "acho errado", "critico", "crítica", "critica",
        "definida juridicamente", "lei nº", "lei n", "lei do racismo", "pl ",
        "projeto de lei", "misoginia é definida", "misoginia nao existiria",
        "misoginia não existiria", "algoritmos", "reddit perguntar", "exclua subs",
        "comentário foi removido", "comentario foi removido",
    ]
    explicit_insults = [
        "idiota", "burro", "imbecil", "otário", "otario", "babaca", "canalha",
        "verme", "lixo", "lunático", "lunatico", "maluca", "cadelinha",
    ]
    has_neutral = any(pattern in lowered for pattern in neutral_patterns)
    has_insult = any(pattern in lowered for pattern in explicit_insults)
    if has_neutral and not has_insult:
        caps = {
            "insult": 0.15,
            "disrespect": 0.20,
            "humiliation": 0.15,
            "identity_attack": 0.25,
            "threat": 0.15,
            "violence": 0.15,
        }
        for code, cap in caps.items():
            scores[code] = min(scores.get(code, 0.0), cap)
    return scores


def _apply_contextual_downweight(text: str, scores: dict[str, float]) -> dict[str, float]:
    lowered = text.lower()
    quotation_markers = [
        '"', "'", "“", "”", "‘", "’", "&gt;", "r:", "fulano disse",
        "quando falam", "quando fala", "a gente fala", "vcs chama",
        "vocês chamam", "chamam a gente", "foi chamado de", "chamou de",
        "acusou de", "disse que", "falar que", "dizer que",
    ]
    meta_markers = [
        "misoginia", "misógina", "misogino", "misógino", "racismo", "islamofobia",
        "lei", "pl ", "projeto de lei", "algoritmo", "algoritmos", "reddit", "sub",
        "subs", "moderação", "moderacao", "comentário", "comentario", "post",
        "declaração", "declaracao", "termo", "definição", "definicao",
    ]
    direct_attack_markers = [
        "você é", "vc é", "tu é", "tu és", "voce é", "vcs são", "vocês são",
        "eles são", "elas são", "esse povo é", "essa mulher é", "esse cara é",
        "essa gente é", "tem que matar", "vou te matar", "deveria morrer",
        "deveria ser estupr", "deveria ser expulso", "não são humanos",
        "nao sao humanos", "é uma praga", "são uma praga", "sao uma praga",
    ]
    has_context = any(p in lowered for p in quotation_markers + meta_markers)
    has_direct_attack = any(p in lowered for p in direct_attack_markers)
    if has_context and not has_direct_attack:
        multipliers = {
            "disrespect": 0.55,
            "insult": 0.55,
            "humiliation": 0.55,
            "identity_attack": 0.60,
            "threat": 0.50,
            "violence": 0.50,
        }
        for code, multiplier in multipliers.items():
            scores[code] = scores.get(code, 0.0) * multiplier
    return scores


def finalize_layer3(
    text: str, dimension_scores: dict[str, float], config: dict[str, Any]
) -> dict[str, Any]:
    scores = dict(dimension_scores)
    scores = _apply_lexical_boosts(text, scores, config)
    scores = _block_false_positive_insult(text, scores)
    scores = _apply_contextual_downweight(text, scores)
    # O runner já gravou os scores NLI. Estes campos explicitam os valores finais.
    dimension_scores.update(scores)

    severe = max(
        scores.get("violence", 0.0), scores.get("threat", 0.0),
        scores.get("dehumanization", 0.0), scores.get("exclusion", 0.0),
    )
    identity = max(
        scores.get("identity_attack", 0.0), scores.get("dehumanization", 0.0),
        scores.get("exclusion", 0.0),
    )
    incivility = max(
        scores.get("disrespect", 0.0), scores.get("insult", 0.0),
        scores.get("humiliation", 0.0),
    )
    all_codes = [
        "disrespect", "insult", "humiliation", "dehumanization", "violence",
        "exclusion", "identity_attack", "threat",
    ]
    mean_all = mean(scores.get(code, 0.0) for code in all_codes)
    weights = config.get("toxicity_weights", {})
    toxicity = (
        float(weights.get("severe", 0.45)) * severe
        + float(weights.get("identity", 0.25)) * identity
        + float(weights.get("incivility", 0.20)) * incivility
        + float(weights.get("mean_all", 0.10)) * mean_all
    )
    thresholds = config["thresholds"]
    if toxicity >= float(thresholds["toxicity_high"]):
        label = "alta"
    elif toxicity >= float(thresholds["toxicity_medium"]):
        label = "media"
    elif toxicity >= float(thresholds["toxicity_low"]):
        label = "baixa"
    else:
        label = "nao_toxico"

    active = [
        code
        for code, score in scores.items()
        if score >= float(thresholds.get("dimensions", {}).get(code, thresholds["default"]))
    ]
    primary = max(scores, key=scores.get) if scores else ""
    return {
        "primary_dimension": primary,
        "primary_score": scores.get(primary, 0.0),
        "active_dimensions": "|".join(active),
        "active_dimension_count": len(active),
        "toxicity_score": toxicity,
        "toxicity_label": label,
    }

