"""Laya input semantics, adapted from laya-mlx; see NOTICE."""

import json

from .common import QTYPES, build_sequence, render_options


class PromptMixin:
    @staticmethod
    def _to_internal(qdef):
        if not isinstance(qdef, dict):
            raise ValueError("Each question must be a dictionary")
        kind = qdef.get("type")
        if kind not in QTYPES:
            raise ValueError(f"Unknown question type {kind!r}; expected choice, score, or noul")
        if "instructions" not in qdef:
            raise ValueError("Question is missing instructions")
        criteria = qdef.get("criteria")
        if kind == "choice":
            if isinstance(criteria, list):
                if not all(isinstance(c, str) for c in criteria):
                    raise ValueError("Choice labels must be strings")
                if len(set(criteria)) != len(criteria):
                    raise ValueError("Choice labels must be unique")
                criteria = dict.fromkeys(criteria)
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError("Choice criteria must be a nonempty dictionary or list")
            if not all(isinstance(k, str) for k in criteria):
                raise ValueError("Choice labels must be strings")
        elif kind == "score":
            if not isinstance(criteria, list) or not criteria:
                raise ValueError("Score criteria must be a nonempty list")
        elif criteria is not None and not isinstance(criteria, dict):
            raise ValueError("Noul criteria must be a dictionary with false/true descriptions")
        instructions = qdef["instructions"]
        if not isinstance(instructions, str):
            instructions = json.dumps(instructions)
        return {"t": kind, "ins": instructions, "crit": criteria}

    def prepare(self, state, questions):
        """Construct upstream-compatible CPU inputs, useful for parity and profiling."""
        if not isinstance(questions, dict):
            raise ValueError("questions must be a dictionary keyed by question id")
        items, internal = [], []
        for qid, definition in questions.items():
            q = self._to_internal(definition)
            ids, markers = build_sequence(
                self.tok, state, q, self.cfg.get("max_len", 512), self.cfg.get("head_max_len", 192)
            )
            if len(markers) != len(render_options(q)):
                raise ValueError(f"Question {qid!r} has too many options for the token budget")
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            internal.append(q)
        return items, internal
