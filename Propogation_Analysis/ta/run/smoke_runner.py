from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        default="test_*.py",
        help="Glob pattern for smoke tests inside the smoke/ directory.",
    )
    args = parser.parse_args()

    smoke_dir = Path(__file__).resolve().parents[1] / "smoke"
    suite = unittest.defaultTestLoader.discover(str(smoke_dir), pattern=args.pattern)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
