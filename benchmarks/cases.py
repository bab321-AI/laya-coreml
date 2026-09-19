"""Validation fixtures from laya-mlx fc1df62 (Apache-2.0); see NOTICE."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def workload(count=3, long=False):
    state = json.loads((ROOT / "examples/state.json").read_text())
    definitions = list(json.loads((ROOT / "examples/questions.json").read_text()).values())
    if long:
        state["body"] = "The customer reports duplicate billing and requests a refund today. " * 200
    questions = {f"q{i}": definitions[i % len(definitions)] for i in range(count)}
    return state, questions


def parity_cases():
    state, questions = workload()
    cases = [("email", state, questions)]
    messages = {
        "en": "I was charged twice for invoice 4411, please refund it today.",
        "zh": "发票4411被重复扣款，请今天退款。",
        "de": "Ich wurde zweimal für Rechnung 4411 belastet, bitte erstatten Sie den Betrag.",
        "fr": "J'ai été facturé deux fois pour la facture 4411, remboursez-moi s'il vous plaît.",
        "es": "Me cobraron dos veces la factura 4411, por favor devuélvanme el dinero.",
        "hi": "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया, कृपया पैसे वापस करें।",
        "ja": "請求書4411で二重に請求されました。返金してください。",
        "ru": "С меня дважды списали деньги по счёту 4411, верните деньги.",
    }
    for lang, message in messages.items():
        cases.append((lang, {"message": message}, questions))
    cases.extend(
        [
            ("empty_state", "", questions),
            ("long", *workload(3, long=True)),
            ("conversation", [{"role": "user", "content": messages["en"]}], questions),
            ("mask_literals", "[MASK] <mask> hello [MASK] <mask>", questions),
            ("many_questions", *workload(20)),
            (
                "structured",
                state,
                {
                    "choice": {
                        "type": "choice",
                        "instructions": {"task": "choose department"},
                        "criteria": {"billing": {"description": "refunds"}, "other": False},
                    },
                    "score": {
                        "type": "score",
                        "instructions": "Urgency?",
                        "criteria": [{"level": "low"}, "high"],
                    },
                    "noul": {
                        "type": "noul",
                        "instructions": "Refund?",
                        "criteria": {"true": {"reason": "money back"}},
                    },
                },
            ),
            (
                "twenty_options",
                state,
                {
                    "choice": {
                        "type": "choice",
                        "instructions": "Which department handles billing?",
                        "criteria": ["billing"] + [f"department_{i}" for i in range(19)],
                    },
                },
            ),
        ]
    )
    return cases
