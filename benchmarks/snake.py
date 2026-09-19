"""Compare actual Core ML and MLX decisions on the same live Snake states.

Uses laya-mlx's published game rules, features and safety policy; this is a
headless experiment, not a measurement of terminal rendering or frame rate.
"""

import argparse
import time
from pathlib import Path

from laya_mlx.snake.game import SnakeGame
from laya_mlx.snake.policy import LayaPolicy

from .common import environment, save, stats


class Policy(LayaPolicy):
    def __init__(self, agent):
        self.agent = agent
        self.guarded = True
        self.prompt = "compact"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("coreml_dir", type=Path)
    p.add_argument("mlx_source", type=Path)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 102])
    p.add_argument("--output", type=Path, default=Path("benchmarks/results/snake.json"))
    a = p.parse_args()
    from laya_mlx import Agent as MLXAgent

    from laya_coreml import Agent as CoreMLAgent

    coreml = CoreMLAgent(a.coreml_dir, compute_units="cpu_gpu")
    mlx = MLXAgent(a.mlx_source, dtype="float16", batch_size=3)
    policies = {"coreml": Policy(coreml), "mlx": Policy(mlx)}
    warmup = SnakeGame(seed=a.seeds[0])
    for policy in policies.values():
        for _ in range(10):
            policy.decide(warmup)
    report = {
        "environment": environment(),
        "export": coreml.manifest,
        "method": "Both models infer every state; order alternates each step. Core ML action advances the game. Timings exclude the other backend, game step, and rendering. Planner features and cycle safety are enabled for both.",
        "runs": [],
    }
    for seed in a.seeds:
        game = SnakeGame(seed=seed)
        samples = {"coreml": [], "mlx": []}
        traces, agree, proposed_agree, shield = [], 0, 0, 0
        max_probability_error = 0.0
        for index in range(a.steps):
            decisions = {}
            for name in ["coreml", "mlx"] if index % 2 == 0 else ["mlx", "coreml"]:
                start = time.perf_counter()
                decisions[name] = policies[name].decide(game)
                samples[name].append((time.perf_counter() - start) * 1000)
            c, m = decisions["coreml"], decisions["mlx"]
            agree += c.executed == m.executed
            proposed_agree += c.proposed == m.proposed
            shield += c.intervened
            max_probability_error = max(
                max_probability_error,
                max(abs(c.probabilities[k] - m.probabilities[k]) for k in c.probabilities),
            )
            traces.append({"tick": index, "coreml": c.to_dict(), "mlx": m.to_dict()})
            game.step(c.executed)
            if not game.alive or game.won:
                break
        run = {
            "seed": seed,
            "steps": len(traces),
            "alive": game.alive,
            "score": game.score,
            "length": len(game.body),
            "executed_agreement": agree,
            "proposed_agreement": proposed_agree,
            "coreml_shield_interventions": shield,
            "probability_max_abs_error": max_probability_error,
            "latency": {name: stats(values) for name, values in samples.items()},
            "trace": traces,
        }
        report["runs"].append(run)
        save(a.output, report)
        print(
            f"seed={seed} steps={len(traces)} score={game.score} alive={game.alive} action agreement={agree}/{len(traces)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
