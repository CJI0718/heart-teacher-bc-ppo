"""All learning commands require explicit --execute-training. No implicit run."""

import argparse
import json
from .config import TrainConfig

LEARNING = {"components", "teacher", "bc", "ppo"}


def parser():
    p = argparse.ArgumentParser(description="HEART teacher -> attention BC -> PPO")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="print the pipeline without running learning")
    check = sub.add_parser("check", help="forward/contract/math checks, no parameter updates")
    check.add_argument("--report", default=None)
    for name in ("components", "teacher", "bc", "ppo"):
        s = sub.add_parser(name)
        s.add_argument("--execute-training", action="store_true")
        s.add_argument("--output", required=True)
        s.add_argument("--seed", type=int, default=7)
        if name in ("components", "teacher"):
            s.add_argument("--deals", type=int, default=2000)
        if name == "teacher":
            s.add_argument("--features", required=True)
        if name == "bc":
            s.add_argument("--data", required=True)
            s.add_argument("--epochs", type=int, default=10)
        if name == "ppo":
            s.add_argument("--checkpoint", required=True)
            s.add_argument("--iterations", type=int, default=100)
    s = sub.add_parser("select")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--top-per-task", type=int, default=32)
    s = sub.add_parser("demos")
    s.add_argument("--teacher", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--deals", type=int, default=500)
    s = sub.add_parser("export")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--output", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for key in ("deals", "epochs", "iterations", "top_per_task"):
        if hasattr(args, key) and getattr(args, key) < 1:
            raise SystemExit(key + " must be positive")
    if args.command == "plan":
        print("components -> select -> teacher -> demos -> attention BC -> PPO -> export")
        print("No training has started. Learning requires --execute-training.")
        return
    if args.command in LEARNING and not args.execute_training:
        print(
            json.dumps(
                {"status": "plan_only", "training_executed": False, "arguments": vars(args)},
                indent=2,
            )
        )
        return
    if args.command == "check":
        from .verify import run_checks

        print(json.dumps(run_checks(args.report), indent=2))
        return
    from . import training as t

    c = TrainConfig(seed=getattr(args, "seed", 7))
    if args.command == "components":
        t.train_components(args.output, args.deals, c)
    elif args.command == "select":
        t.select_features(args.checkpoint, args.output, args.top_per_task)
    elif args.command == "teacher":
        t.train_teacher(args.features, args.output, args.deals, c)
    elif args.command == "demos":
        t.collect_demos(args.teacher, args.output, args.deals, c)
    elif args.command == "bc":
        t.train_bc(args.data, args.output, args.epochs, c)
    elif args.command == "ppo":
        t.train_ppo(args.checkpoint, args.output, args.iterations, c)
    elif args.command == "export":
        from .export import export_checkpoint

        print(export_checkpoint(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
