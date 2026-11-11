"""Run: python examples/quickstart.py --output results/example [--search]."""

import argparse
from pathlib import Path
from battwin_framework import Framework
from battwin_framework.schema import GRAVIMETRIC, VOLUMETRIC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/example"))
    parser.add_argument("--search", action="store_true")
    args = parser.parse_args()
    framework = Framework.from_bundle()
    test = framework.load_dataset("test")
    validation = framework.validate(test)
    print(validation.metrics.to_string(index=False))
    validation.save(args.output / "validation", plots=True)
    framework.sensitivity.autodiff(test).save(args.output / "autodiff", plots=True)
    if args.search:
        search = framework.random_search(
            targets={GRAVIMETRIC: 165.0, VOLUMETRIC: 480.0}
        )
        print(search.summary.to_string(index=False))
        search.save(args.output / "search", plots=True)


if __name__ == "__main__":
    main()
