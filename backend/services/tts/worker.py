# # hindi-dubbing-platform/backend/services/tts/worker.py
# """Entry point of an isolated TTS worker process:  python -m backend.services.tts.worker --engine parler

# Started by ``backend.services.tts.isolated.IsolatedEngine``; never run heavy work on the API thread.
# """

# from __future__ import annotations

# import argparse
# import sys

# from backend.app.config import TTSEngine
# from backend.services.tts.isolated import serve


# def main() -> int:
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--engine", required=True, choices=[e.value for e in TTSEngine])
#     ap.add_argument("--device", default="cpu")
#     ap.add_argument("--data-dir", default=None)
#     a = ap.parse_args()
#     return serve(TTSEngine(a.engine), a.device, a.data_dir)


# if __name__ == "__main__":
#     sys.exit(main())




"""Entry point for an isolated TTS worker.

Example:
    python -m backend.services.tts.worker --engine parler

Started by:
    backend.services.tts.isolated.IsolatedEngine

Heavy model inference runs in this child process, not in the API process.
"""

from __future__ import annotations

import argparse
import sys

from backend.app.config import TTSEngine
from backend.services.tts.isolated import serve


def build_parser() -> argparse.ArgumentParser:
    """Build the worker command-line interface."""

    parser = argparse.ArgumentParser(
        description="Run an isolated TTS engine worker."
    )

    parser.add_argument(
        "--engine",
        required=True,
        choices=[
            TTSEngine.PARLER.value,
            TTSEngine.INDICF5.value,
        ],
        help="TTS engine to load.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device. Defaults to CPU.",
    )

    parser.add_argument(
        "--data-dir",
        default=None,
        help="Application data directory.",
    )

    return parser


def main() -> int:
    """Parse arguments and start the isolated worker."""

    parser = build_parser()

    args = parser.parse_args()

    try:
        engine = TTSEngine(
            args.engine
        )

        return serve(
            engine=engine,
            device=args.device,
            data_dir=args.data_dir,
        )

    except KeyboardInterrupt:
        return 130

    except Exception as exc:
        print(
            f"TTS worker startup error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())
