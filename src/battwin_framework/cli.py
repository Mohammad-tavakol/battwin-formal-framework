"""Command-line access to the preserved notebook experiments."""

import argparse
import json
from .experiment import Experiment, initialize_workspace, verify_assets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Copy exact notebook, models and datasets into a workspace")
    init.add_argument("directory")
    commands.add_parser("verify", help="Verify bundled artifact and scientific source checksums")
    for name in ["run", "reproduce"]:
        command = commands.add_parser(name, help="Run preserved experiments" if name == "run" else "Run both original datasets")
        command.add_argument("directory")
        command.add_argument("--no-save", action="store_true", help="Keep original SAVE_RESULTS=False behaviour")
        command.add_argument("--quiet", action="store_true", help="Suppress notebook display objects")
        if name == "run":
            command.add_argument("--dataset", default="test_aggregated_deterministic.xlsx")
            command.add_argument("--through-cell", type=int)
    args = parser.parse_args(argv)
    if args.command == "verify":
        manifest = verify_assets()
        print(json.dumps({"verified_files": len(manifest["files"]), "unchanged_code_cells": len(manifest["cells"])}))
    elif args.command == "init":
        print(initialize_workspace(args.directory))
    else:
        datasets = (["test_aggregated_deterministic.xlsx", "train_valid_aggregated_deterministic.xlsx"]
                    if args.command == "reproduce" else [args.dataset])
        for dataset in datasets:
            run = Experiment(args.directory, dataset, save_results=not args.no_save, quiet=args.quiet)
            run.run(through_cell=getattr(args, "through_cell", None))
            print(f"Completed {dataset}: {len(run.executed_cells)} code cells")


if __name__ == "__main__":
    main()
