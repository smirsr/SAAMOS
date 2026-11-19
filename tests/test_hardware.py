from saam.hardware import probes
from saam.hardware.diagnostics import run_diagnostics

VALID = {probes.PASS, probes.FAIL, probes.NOT_TESTED}


def test_mock_report_has_all_sections():
    report = run_diagnostics(mock=True)
    expected = {"system", "memory", "storage", "tools", "audio_input", "audio_output", "camera"}
    assert expected.issubset(report.keys())
    assert all(r["status"] in VALID for r in report.values())


def test_mock_devices_pass():
    report = run_diagnostics(mock=True)
    assert report["audio_input"]["status"] == probes.PASS
    assert report["audio_output"]["status"] == probes.PASS
    assert report["camera"]["status"] == probes.PASS
    assert report["memory"]["status"] == probes.PASS


def test_parsers_handle_empty_output():
    assert probes.parse_alsa_cards("**** List of CAPTURE Hardware Devices ****\n") == []
    assert probes.parse_video_devices("") == []
    assert probes.parse_meminfo("") == {}


def test_missing_command_does_not_crash():
    result = probes.run_command(["definitely_not_a_real_tool"])
    assert result["ok"] is False
    assert "not installed" in result["error"]