"""Command-line entry points for the framework's explicit capabilities."""

import argparse
import json
from pathlib import Path
from .resources import verify_assets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Check model and workbook artifact hashes")
    for name in [
        "predict",
        "validate",
        "sensitivity",
        "search",
        "feedback",
        "describe",
    ]:
        command = commands.add_parser(name, help=f"Run the {name} capability")
        command.add_argument(
            "--project", help="Root with best models/ and datasets; default: bundle"
        )
        if name == "describe":
            command.add_argument("--family", choices=["RF", "DNN"], default="DNN")
            continue
        command.add_argument("--output", default=f"results/{name}")
        if name != "predict":
            command.add_argument("--plots", action="store_true")
        if name != "search":
            command.add_argument(
                "--dataset", default="test", help="Dataset alias or CSV/XLSX path"
            )
        if name == "predict":
            command.add_argument("--family", choices=["RF", "DNN"], default="DNN")
        if name == "sensitivity":
            command.add_argument(
                "--method",
                choices=["pfi", "finite_difference", "autodiff", "compare"],
                required=True,
            )
            command.add_argument("--reference", default="train_valid")
        if name == "search":
            command.add_argument("--cg", type=float, required=True)
            command.add_argument("--cv", type=float, required=True)
            command.add_argument("--sweeps", type=int, default=1000)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--tolerance", type=float, default=0.02)
            command.add_argument("--local-scale", type=float, default=0.1)
            command.add_argument("--training", default="train_valid")
    args = parser.parse_args(argv)
    if args.command == "verify":
        manifest = verify_assets()
        print(json.dumps({"verified_model_and_dataset_files": len(manifest["files"])}))
        return
    from .framework import Framework

    framework = Framework(args.project)
    if args.command == "describe":
        print(framework.describe(args.family).to_string(index=False))
        return
    if args.command == "search":
        from .schema import GRAVIMETRIC, VOLUMETRIC

        result = framework.random_search(
            targets={GRAVIMETRIC: args.cg, VOLUMETRIC: args.cv},
            training=framework.load_dataset(args.training),
            n_sweeps=args.sweeps,
            random_seed=args.seed,
            tolerance_fraction=args.tolerance,
            local_scale_fraction=args.local_scale,
        )
    else:
        data = framework.load_dataset(args.dataset)
        if args.command == "predict":
            path = Path(args.output)
            path.mkdir(parents=True, exist_ok=True)
            framework.predict(data, families=(args.family,))[args.family].to_csv(
                path / "predictions.csv", index=False
            )
            print(path.resolve())
            return
        if args.command == "validate":
            result = framework.validate(data)
        elif args.command == "feedback":
            result = framework.disturbance_rejection(data)
        else:
            method = getattr(framework.sensitivity, args.method)
            kwargs = (
                {}
                if args.method == "pfi"
                else {"reference": framework.load_dataset(args.reference)}
            )
            result = method(data, **kwargs)
    print(result.save(args.output, plots=args.plots))


if __name__ == "__main__":
    main()
