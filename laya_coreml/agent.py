"""Core ML inference. No MLX, Transformers, or PyTorch dependency at runtime."""

import json
import math
from pathlib import Path

import numpy as np

from .prompt import PromptMixin
from .result import ResultMixin
from .tokenizer import Tokenizer

COMPUTE_UNITS = {"all": "ALL", "cpu": "CPU_ONLY", "cpu_gpu": "CPU_AND_GPU", "cpu_ne": "CPU_AND_NE"}


class Agent(PromptMixin, ResultMixin):
    def __init__(self, model_dir, *, compute_units="cpu_gpu", allow_unvalidated_gpu=False):
        if compute_units not in COMPUTE_UNITS:
            raise ValueError(f"compute_units must be one of {list(COMPUTE_UNITS)}")
        self.model_dir = Path(model_dir).expanduser()
        self.manifest = json.loads((self.model_dir / "coreml_config.json").read_text())
        if self.manifest.get("format") != "laya-coreml" or self.manifest.get("format_version") != 1:
            raise ValueError("Unsupported Core ML export format")
        self.shape = self.manifest["shape"]
        if (
            compute_units == "cpu_gpu"
            and self.shape["flexible"]
            and not self.shape.get("lengths")
            and not allow_unvalidated_gpu
        ):
            raise ValueError(
                "RangeDim + CPU_AND_GPU failed local fidelity and repeatability checks. "
                "Re-export with the default enumerated shapes, or use compute_units='cpu'. "
                "allow_unvalidated_gpu=True is for reproducing the failure only."
            )
        self.cfg = json.loads((self.model_dir / "rl_agent_config.json").read_text())
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        if len(self.temperature) != 3 or any(
            not math.isfinite(float(t)) or float(t) <= 0
            for t in [*self.temperature, *self.temperature_by_options.values()]
        ):
            raise ValueError("Calibration temperatures must be finite and positive")
        self.tok = Tokenizer(self.model_dir / "tokenizer")
        self.batch_size = self.shape["batch_size"]
        self.pad_to_multiple = 16
        import coremltools as ct

        self.compute_units = compute_units
        self.model = ct.models.MLModel(
            str(self.model_dir / "model.mlpackage"),
            compute_units=getattr(ct.ComputeUnit, COMPUTE_UNITS[compute_units]),
        )

    def forward(self, batch):
        outputs = self.model.predict(batch)
        return np.asarray(outputs["logits"], np.float32), np.asarray(
            outputs["action_logits"], np.float32
        )


def load(model_dir, **kwargs):
    return Agent(model_dir, **kwargs)
