"""Run all hardware probes and print a report.

Usage (from the repository root):
    python -m saam.hardware.diagnostics --mock
    python -m saam.hardware.diagnostics --json --save report.json
"""
import argparse
import json
import sys
from pathlib import Path

from saam.hardware import probes


def run_diagnostics(mock: bool = False) -> dict:
    return {
        "system": probes.check_system(),
        "memory": probes.check_memory(mock=mock),
        "storage": probes.check_storage(),
        "tools": probes.check_tools(),
        "audio_input": probes.check_audio_input(mock=mock),
        "audio_output": probes.check_audio_output(mock=mock),
        "camera": probes.check_camera(mock=mock),
    }


def format_report(report: dict) -> str:
    lines = [f"{'CHECK':<14}{'STATUS':<12}DETAIL"]
    for name, result in report.items():
        lines.append(f"{name:<14}{result['status']:<12}{result['detail']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="SAAMOS hardware diagnostics")
    parser.add_argument("--mock", action="store_true", help="use fixture data instead of real hardware")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument("--save", type=Path, help="also write the JSON report to this file")
    args = parser.parse_args()

    report = run_diagnostics(mock=args.mock)
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    if args.save:
        args.save.write_text(json.dumps(report, indent=2))
    return 1 if any(r["status"] == probes.FAIL for r in report.values()) else 0


if __name__ == "__main__":
    sys.exit(main())