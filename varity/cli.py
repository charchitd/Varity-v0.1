"""Varity CLI — varity check / varity demo / varity batch.

No rich or click dependencies — ANSI colours only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_DIM = "\033[2m"


def _c(text: str, *codes: str) -> str:
    """Wrap *text* with ANSI *codes* if stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + _RESET


def _print_result(result: object) -> None:  # CheckResult
    from varity.models import CheckResult

    if not isinstance(result, CheckResult):
        print(str(result))
        return

    print()
    print(_c("=" * 60, _BOLD))
    print(_c("  Varity Check Result", _BOLD, _CYAN))
    print(_c("=" * 60, _BOLD))
    print(f"  Claims found    : {len(result.claims)}")
    print(f"  Flagged claims  : {len(result.flagged_claims)}")

    conf_color = _GREEN if result.overall_confidence >= 0.7 else (
        _YELLOW if result.overall_confidence >= 0.4 else _RED
    )
    vss_color = _GREEN if result.vss_score >= 0.7 else (
        _YELLOW if result.vss_score >= 0.4 else _RED
    )

    print(f"  Confidence      : {_c(f'{result.overall_confidence:.1%}', conf_color, _BOLD)}")
    print(f"  VSS             : {_c(f'{result.vss_score:.1%}', vss_color, _BOLD)}")
    print(f"  Duration        : {result.duration_ms} ms")

    total_tok = result.token_usage.get("total_tokens", 0)
    if total_tok:
        print(f"  Tokens (est.)   : {total_tok:,}")

    if result.claims:
        print()
        print(_c("  Claims", _BOLD))
        for i, claim in enumerate(result.claims, 1):
            flag = _c("!!", _RED) if claim.flagged else _c("OK", _GREEN)
            print(
                f"  {i:2}. {flag} [{claim.claim_type:10s}] "
                f"conf={claim.confidence:.2f} vss={claim.vss_score:.2f}  "
                f"{_c(claim.text[:70], _DIM)}"
            )

    if result.flagged_claims:
        print()
        print(_c("  Flagged details", _BOLD, _RED))
        for claim in result.flagged_claims:
            print(f"    - {claim.text}")
            print(f"      {_c(claim.verification_notes, _DIM)}")

    if result.corrected_response:
        print()
        print(_c("  Corrected Response", _BOLD, _YELLOW))
        print(_c("  " + result.corrected_response.replace("\n", "\n  "), _DIM))

    print(_c("=" * 60, _BOLD))
    print()


# ---------------------------------------------------------------------------
# Canned demo output (no API key required)
# ---------------------------------------------------------------------------

_DEMO_RESPONSE = (
    "The Great Wall of China was built in 221 BC under Emperor Qin Shi Huang. "
    "It stretches exactly 13,170 miles. "
    "The wall is clearly visible from space with the naked eye. "
    "Python was created by Guido van Rossum and first released in 1991."
)

_DEMO_CANNED = """\

  Varity Demo - Canned Output
  ============================================================
  Response checked:
    "The Great Wall of China was built in 221 BC under Emperor
     Qin Shi Huang. It stretches exactly 13,170 miles. The wall
     is clearly visible from space with the naked eye. Python
     was created by Guido van Rossum and first released in 1991."

  Claims found    : 4
  Flagged claims  : 2
  Confidence      : 52.3%
  VSS             : 61.0%
  Duration        : ~4200 ms

  Claims:
    1. OK [factual   ] conf=0.78 vss=1.00  The Great Wall was built in 221 BC...
    2. !! [numerical ] conf=0.31 vss=0.50  It stretches exactly 13,170 miles.
    3. !! [factual   ] conf=0.18 vss=0.33  The wall is visible from space...
    4. OK [temporal  ] conf=0.82 vss=1.00  Python first released in 1991.

  Flagged details:
    - It stretches exactly 13,170 miles.
      depth=2, flips=1, vss=0.50, cross=uncertain
    - The wall is clearly visible from space with the naked eye.
      depth=2, flips=2, vss=0.33, cross=contradicted

  Corrected Response:
    "The Great Wall of China was built in 221 BC under Emperor Qin Shi Huang.
     It reportedly stretches around 13,170 miles (though estimates vary).
     Contrary to popular belief, the wall is not clearly visible from space
     with the naked eye. Python was created by Guido van Rossum and first
     released in 1991."
  ============================================================

  To run a live check:
    varity check --provider anthropic --key sk-ant-... \\
                 --response "{response}"
"""


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def _run_check(
    response: str,
    provider_name: str,
    api_key: str,
    model: Optional[str],
    depth: int,
    threshold: float,
    json_out: bool,
) -> int:
    from varity import Varity, VarityConfig
    from varity.providers import get_provider

    kwargs: dict[str, object] = {}
    if model:
        kwargs["model"] = model

    provider = get_provider(provider_name, api_key=api_key, **kwargs)
    config = VarityConfig(depth=depth, confidence_threshold=threshold)
    varity = Varity(provider=provider, config=config)

    try:
        result = await varity.acheck(response)
    finally:
        await provider.close()

    if json_out:
        print(result.model_dump_json(indent=2))
    else:
        _print_result(result)

    return 1 if result.flagged_claims else 0


def _cmd_check(args: argparse.Namespace) -> int:
    return asyncio.run(
        _run_check(
            response=args.response,
            provider_name=args.provider,
            api_key=args.key,
            model=getattr(args, "model", None),
            depth=args.depth,
            threshold=args.threshold,
            json_out=args.json,
        )
    )


def _cmd_demo(args: argparse.Namespace) -> int:
    """Show a demo — canned output by default, live check if --key is given."""
    key = getattr(args, "key", None) or os.environ.get("VARITY_API_KEY", "")
    provider_name = getattr(args, "provider", None) or os.environ.get(
        "VARITY_PROVIDER", "anthropic"
    )

    if key:
        # Run a live check on the demo response
        print(_c(f"\n  Running live demo with provider '{provider_name}'…\n", _CYAN))
        return asyncio.run(
            _run_check(
                response=_DEMO_RESPONSE,
                provider_name=provider_name,
                api_key=key,
                model=None,
                depth=2,
                threshold=0.5,
                json_out=False,
            )
        )

    # No key — show canned output
    print(_DEMO_CANNED.format(response=_DEMO_RESPONSE[:60] + "..."))
    return 0


async def _run_batch(
    input_path: str,
    output_path: str,
    provider_name: str,
    api_key: str,
    model: Optional[str],
    depth: int,
) -> int:
    from varity import Varity, VarityConfig
    from varity.providers import get_provider

    kwargs: dict[str, object] = {}
    if model:
        kwargs["model"] = model

    provider = get_provider(provider_name, api_key=api_key, **kwargs)
    config = VarityConfig(depth=depth)
    varity = Varity(provider=provider, config=config)

    results = []
    errors = 0

    try:
        with open(input_path, encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip()]

        print(f"Processing {len(lines)} item(s)…")

        for i, line in enumerate(lines, 1):
            try:
                item = json.loads(line)
                response_text = item.get("response", item.get("text", str(item)))
                result = await varity.acheck(response_text)
                row = {
                    "input": response_text[:80],
                    "overall_confidence": result.overall_confidence,
                    "vss_score": result.vss_score,
                    "flagged": len(result.flagged_claims),
                    "total_tokens": result.token_usage.get("total_tokens", 0),
                    "duration_ms": result.duration_ms,
                }
                results.append(row)
                flag_str = _c(str(len(result.flagged_claims)), _RED if result.flagged_claims else _GREEN)
                print(
                    f"  [{i}/{len(lines)}] conf={result.overall_confidence:.2f} "
                    f"vss={result.vss_score:.2f} flagged={flag_str}"
                )
            except Exception as exc:
                errors += 1
                print(_c(f"  [{i}/{len(lines)}] ERROR — {exc}", _RED), file=sys.stderr)
    finally:
        await provider.close()

    with open(output_path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    status = _c("OK", _GREEN) if not errors else _c(f"{errors} error(s)", _RED)
    print(f"\nWrote {len(results)} results to {output_path}. Status: {status}")
    return 1 if errors else 0


def _cmd_batch(args: argparse.Namespace) -> int:
    return asyncio.run(
        _run_batch(
            input_path=args.input,
            output_path=args.output,
            provider_name=args.provider,
            api_key=args.key,
            model=getattr(args, "model", None),
            depth=args.depth,
        )
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="varity",
        description="Recursive self-checking for LLM hallucination reduction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  varity demo\n"
            "  varity demo --provider anthropic --key sk-ant-...\n"
            "  varity check --provider openai --key sk-... --response 'Paris is in Spain.'\n"
            "  varity batch --provider gemini --key AIza... --input in.jsonl --output out.jsonl\n"
        ),
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- check ---------------------------------------------------------------
    p_check = sub.add_parser("check", help="Verify a single LLM response.")
    p_check.add_argument("--response", required=True, help="LLM response text to verify.")
    p_check.add_argument(
        "--provider", default="anthropic",
        help="Provider name: anthropic | openai | gemini (default: anthropic)."
    )
    p_check.add_argument("--key", required=True, help="API key (BYOK — never stored).")
    p_check.add_argument("--model", default=None, help="Override default model for the provider.")
    p_check.add_argument("--depth", type=int, default=2, help="Verification depth (default: 2).")
    p_check.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold below which claims are flagged (default: 0.5)."
    )
    p_check.add_argument("--json", action="store_true", help="Output raw JSON.")
    p_check.set_defaults(func=_cmd_check)

    # -- demo ----------------------------------------------------------------
    p_demo = sub.add_parser(
        "demo",
        help="Show a demo. Pass --key to run a live check, otherwise shows canned output."
    )
    p_demo.add_argument(
        "--provider", default=None,
        help="Provider for live demo (default: $VARITY_PROVIDER or anthropic)."
    )
    p_demo.add_argument(
        "--key", default=None,
        help="API key for live demo (default: $VARITY_API_KEY)."
    )
    p_demo.set_defaults(func=_cmd_demo)

    # -- batch ---------------------------------------------------------------
    p_batch = sub.add_parser("batch", help="Verify a JSONL file of responses.")
    p_batch.add_argument("--input", required=True, help="Input JSONL file (one JSON object per line with 'response' or 'text' key).")
    p_batch.add_argument("--output", required=True, help="Output JSONL file path.")
    p_batch.add_argument("--provider", default="anthropic", help="Provider name.")
    p_batch.add_argument("--key", required=True, help="API key.")
    p_batch.add_argument("--model", default=None, help="Override default model.")
    p_batch.add_argument("--depth", type=int, default=2, help="Verification depth.")
    p_batch.set_defaults(func=_cmd_batch)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Varity CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
