"""Hardware probes for SAAMOS.

Every probe returns a dictionary with a "status" key that is one of
PASS, FAIL or NOT TESTED, plus a "detail" key. Probes never raise.
Pass mock=True to read fixture files instead of touching hardware.
"""
import platform
import shutil
import subprocess
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"
NOT_TESTED = "NOT TESTED"

MOCK_DIR = Path(__file__).parent / "mock_data"
REQUIRED_TOOLS = ["arecord", "aplay", "v4l2-ctl"]


def is_linux() -> bool:
    return platform.system() == "Linux"


def run_command(cmd: list[str], timeout: int = 10) -> dict:
    """Run a command safely. Returns {"ok", "output", "error"}."""
    if shutil.which(cmd[0]) is None:
        return {"ok": False, "output": "", "error": f"{cmd[0]} is not installed"}
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"ok": False, "output": "", "error": str(exc)}
    return {
        "ok": result.returncode == 0,
        "output": result.stdout,
        "error": result.stderr.strip(),
    }


def get_text(cmd: list[str], mock_file: str, mock: bool) -> dict:
    """Return command output, or fixture text when mock=True."""
    if mock:
        text = (MOCK_DIR / mock_file).read_text()
        return {"ok": True, "output": text, "error": ""}
    return run_command(cmd)


def parse_alsa_cards(text: str) -> list[str]:
    """Return the 'card N: ...' lines from arecord -l / aplay -l output."""
    return [line.strip() for line in text.splitlines() if line.startswith("card ")]


def parse_video_devices(text: str) -> list[str]:
    """Return '/dev/videoN' paths from v4l2-ctl --list-devices output."""
    return [line.strip() for line in text.splitlines() if line.strip().startswith("/dev/video")]


def parse_meminfo(text: str) -> dict:
    """Return MemTotal and MemAvailable in MB from /proc/meminfo text."""
    values = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            values[key] = int(rest.split()[0]) // 1024
    return values


def check_system() -> dict:
    return {
        "status": PASS,
        "detail": f"python {platform.python_version()}, {platform.machine()}, {platform.platform()}",
    }


def check_memory(mock: bool = False) -> dict:
    if not mock and not is_linux():
        return {"status": NOT_TESTED, "detail": "needs Linux (/proc/meminfo)"}
    if mock:
        text = (MOCK_DIR / "meminfo_ok.txt").read_text()
    else:
        try:
            text = Path("/proc/meminfo").read_text()
        except OSError as exc:
            return {"status": FAIL, "detail": str(exc)}
    info = parse_meminfo(text)
    if "MemTotal" not in info:
        return {"status": FAIL, "detail": "could not read MemTotal"}
    return {
        "status": PASS,
        "detail": f"total {info['MemTotal']} MB, available {info.get('MemAvailable', '?')} MB",
    }


def check_storage() -> dict:
    try:
        usage = shutil.disk_usage(Path("/"))
    except OSError as exc:
        return {"status": FAIL, "detail": str(exc)}
    gb = 1024 ** 3
    return {"status": PASS, "detail": f"total {usage.total / gb:.1f} GB, free {usage.free / gb:.1f} GB"}


def _alsa_probe(cmd: list[str], mock_file: str, mock: bool, label: str) -> dict:
    if not mock and not is_linux():
        return {"status": NOT_TESTED, "detail": "needs Linux with ALSA tools"}
    res = get_text(cmd, mock_file, mock)
    if not res["ok"]:
        return {"status": NOT_TESTED, "detail": res["error"] or f"{cmd[0]} failed"}
    cards = parse_alsa_cards(res["output"])
    if not cards:
        return {"status": FAIL, "detail": f"no {label} devices found"}
    return {"status": PASS, "detail": "; ".join(cards)}


def check_audio_input(mock: bool = False) -> dict:
    return _alsa_probe(["arecord", "-l"], "arecord_ok.txt", mock, "capture")


def check_audio_output(mock: bool = False) -> dict:
    return _alsa_probe(["aplay", "-l"], "aplay_ok.txt", mock, "playback")


def check_camera(mock: bool = False) -> dict:
    if not mock and not is_linux():
        return {"status": NOT_TESTED, "detail": "needs Linux with v4l-utils"}
    res = get_text(["v4l2-ctl", "--list-devices"], "v4l2_ok.txt", mock)
    if not res["ok"]:
        return {"status": NOT_TESTED, "detail": res["error"] or "v4l2-ctl failed"}
    devices = parse_video_devices(res["output"])
    if not devices:
        return {"status": FAIL, "detail": "no /dev/video devices found"}
    return {"status": PASS, "detail": ", ".join(devices)}


def check_tools() -> dict:
    if not is_linux():
        return {"status": NOT_TESTED, "detail": "needs Linux"}
    missing = [tool for tool in REQUIRED_TOOLS if shutil.which(tool) is None]
    if missing:
        return {"status": FAIL, "detail": "missing: " + ", ".join(missing)}
    return {"status": PASS, "detail": "all required tools present"}