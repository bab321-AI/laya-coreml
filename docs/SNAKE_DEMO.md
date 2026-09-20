# Core ML Snake: live probabilities, local inference

The terminal demo runs a real Laya model on every move. The left panel shows the
board, score and snake length; the right panel shows direction probabilities,
estimated dead-end risk, food reachability, measured prediction time, engine,
zero output tokens and offline inference status.

## Start on an Apple Silicon Mac

```bash
pip install 'laya-coreml[demo]'
hf download aac6fef/laya-multilingual-coreml-ane --local-dir models/snake
laya-coreml-snake --model ./models/snake --fps 12
```

Use a terminal at least **104 columns × 35 rows** with a monospaced font and
truecolor support. Model loading and initial Core ML compilation happen before
the game starts and can take tens of seconds on the first run. Twelve decisions
per second is a legible presentation speed, not a hardware maximum.

The model path is fully local after the explicit download. Starting the game
with an uncached Hub ID fails with instructions rather than downloading while
displaying `OFFLINE`. The runtime needs no MLX, PyTorch or Transformers.

| Control | Action |
|---|---|
| Space | Pause or resume |
| Up / Down, or `+` / `-` | Increase or decrease decision rate |
| R | Start the next seeded round |
| Q / Ctrl-C | Quit and restore the terminal |

The UI waits when the terminal is too small. `--no-alt-screen` leaves the final
frame in scrollback; `--headless` runs the same model/game without the display.

## Two deployment choices

The default ANE bundle has batch one and length 96. Each game step answers three
questions sequentially. Its **single-question ~5 ms benchmark is not a full-game
frame time**. The ordinary Snake GPU bundle uses B3/L64/K4 and batches those three
questions:

```bash
hf download aac6fef/laya-multilingual-coreml-snake --local-dir models/snake-gpu
laya-coreml-snake --model ./models/snake-gpu --fps 12
```

The engine label describes the selected Core ML compute units. The dedicated
ANE graph has separate execution-plan and hardware-trace evidence in the
[ANE report](ANE_BENCHMARKS.md); `CPU+ANE` still permits host work.

## What the model controls

Code calculates legal moves, safe progress along a Hamiltonian cycle and whether
food is reachable through empty cells. Laya receives these features and returns
direction and boolean probabilities. The displayed risk is `1 - P(safe route)`;
it is a model estimate, not a calibrated probability of dying on the next move.

The default safety shield restricts executed moves to cycle-safe progress. The
UI keeps the original model probabilities visible, shows the proposed and
executed direction, and increments a visible intervention counter when they
differ. `--unassisted` executes raw model top-1 without this shield. Zero deaths
with the shield do not establish unaided Snake intelligence or unlimited survival.

## Record a real run and export shareable media

```bash
laya-coreml-snake --model ./models/snake --fps 12 --seed 7 \
  --duration 75 --record snake.jsonl

# Requires ffmpeg: brew install ffmpeg
laya-coreml-snake export snake.jsonl --start 45 --seconds 20 \
  --output snake.mp4 --gif snake.gif --gif-seconds 15
laya-coreml-snake export snake.jsonl --start 55 --output snake.png
```

Recordings include model provenance, each board before the announced action,
real probabilities, inference latency, elapsed timestamps and a final summary.
The exporter reuses the live terminal composition and preserves **1× wall-clock
timing**. On-screen `RECORDED RUN · 1×` identifies a replay. The sidecar records
source and renderer hashes. Video FPS samples the recording; it does not change
the number of model decisions or accelerate playback.

Output files must be new paths. `--max-speed` disables pacing and waits for each
fresh prediction before moving. It does not skip model calls:

```bash
laya-coreml-snake --model ./models/snake --max-speed \
  --duration 20 --record snake-fast.jsonl
```

## Measure a stable decision rate

```bash
laya-coreml-snake benchmark --model ./models/snake \
  --rates 20,30,40,50,60 --sweep-steps 120 --soak-steps 600 \
  --seeds 101,102,103 --raw-steps 200 --output snake-benchmark.json
```

The benchmark first measures unshielded behavior and uncapped operation, sweeps
target rates, then tests passing rates on longer episodes with multiple seeds.
Passing requires zero deaths, finite outputs, preserved cycle order and food
progress, and at most 1% of ticks exceeding the active computation budget.

Timing includes model prediction, planner work, truecolor Rich composition,
ANSI serialization and game updates. Paced tests include real sleeps. Terminal
emulator painting is excluded from this in-memory rendering test. The result is
the highest passing **tested** rate on this machine and these seeds, not a
universal maximum. See [SNAKE_BENCHMARKS.md](SNAKE_BENCHMARKS.md) for the release run.
