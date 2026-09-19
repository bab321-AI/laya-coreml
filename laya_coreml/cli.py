"""Convert and run Laya Core ML packages."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("convert", help="Export original Laya weights to Core ML")
    export.add_argument("source", help="Local original checkpoint, or pinned Laya model name")
    export.add_argument("output")
    export.add_argument("--max-length", type=int)
    export.add_argument("--batch-size", type=int, default=1)
    export.add_argument("--max-options", type=int, default=32)
    export.add_argument("--fixed", action="store_true")
    export.add_argument("--precision", choices=["float16", "float32"], default="float16")
    export.add_argument("--revision")
    export.add_argument("--attention", choices=["explicit", "sdpa"], default="sdpa")
    export.add_argument("--shape-mode", choices=["enumerated", "range"], default="enumerated")
    predict = commands.add_parser("predict")
    predict.add_argument("model_dir")
    predict.add_argument("--state", required=True, help="Literal state text")
    predict.add_argument(
        "--questions", required=True, help="JSON file containing question definitions"
    )
    predict.add_argument(
        "--compute-units", choices=["all", "cpu", "cpu_gpu", "cpu_ne"], default="cpu_gpu"
    )
    args = parser.parse_args()
    if args.command == "convert":
        from .convert import convert

        convert(
            args.source,
            args.output,
            max_length=args.max_length,
            flexible=not args.fixed,
            batch_size=args.batch_size,
            max_options=args.max_options,
            precision=args.precision,
            revision=args.revision,
            attention=args.attention,
            shape_mode=args.shape_mode,
        )
    else:
        from pathlib import Path

        from .agent import Agent

        agent = Agent(args.model_dir, compute_units=args.compute_units)
        print(
            json.dumps(
                agent.predict(args.state, json.loads(Path(args.questions).read_text())),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
