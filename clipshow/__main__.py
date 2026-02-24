"""Entry point for ClipShow - handles CLI arg parsing."""

import argparse
import sys

from clipshow import __version__


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="clipshow",
        description="Automatic highlight reel generator from video clips",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Video files to process",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run in automatic mode (no full UI)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Suppress all GUI (for scripting/batch use, implies --auto)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="highlight_reel.mp4",
        help="Output file path (default: highlight_reel.mp4)",
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=0,
        help="Number of parallel workers (default: auto = CPU count)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    args = parser.parse_args(argv)
    if args.headless:
        args.auto = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.auto:
        from clipshow.app import run_auto_mode

        return run_auto_mode(args.files, args.output, headless=args.headless, workers=args.workers)
    else:
        from clipshow.app import run_gui

        return run_gui(args.files)


if __name__ == "__main__":
    sys.exit(main())
