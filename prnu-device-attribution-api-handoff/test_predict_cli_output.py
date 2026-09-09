"""Focused CLI persistence contract for PRNU attribution."""

from __future__ import annotations

import json
import sys

from prnu_attribution import predict as cli


def test_json_output_persists_unmodified_prediction(tmp_path, monkeypatch, capsys):
    expected = {"device_attribution": {"prediction": "device_A"}, "warnings": ["native"]}
    output = tmp_path / "result.json"
    monkeypatch.setattr(cli, "predict", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["predict", "--image", "input.jpg", "--model-dir", "models", "--json", "--output", str(output)],
    )

    cli.main()

    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert json.loads(capsys.readouterr().out) == expected
