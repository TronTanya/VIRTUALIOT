import io
import json

import pandas as pd
import pytest

from reporting import (
    build_html_report,
    config_to_json,
    events_for_export,
    frame_for_export,
    to_csv_bytes,
)
from scenarios import preset_config, run_comparison
from simulation import MODEL_VERSION, SimulationConfig, run_simulation


@pytest.fixture(scope="module")
def result():
    return run_simulation(preset_config("Холодное помещение"))


def test_csv_has_bom_and_russian_headers(result):
    data = to_csv_bytes(frame_for_export(result))
    assert data.startswith(b"\xef\xbb\xbf")
    df = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
    assert "Температура (датчик), °C" in df.columns
    assert len(df) == result.config.duration_min + 1
    assert set(df["Обогреватель на интервале [t; t+1)"].dropna()) <= {"Включён", "Выключен", "—"}


def test_events_csv(result):
    data = to_csv_bytes(events_for_export(result))
    df = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
    assert list(df.columns) == ["Время, мин", "Устройство", "Новое состояние", "Показание", "Причина"]
    assert len(df) == len(result.events) > 0


def test_config_json_roundtrip(result):
    payload = json.loads(config_to_json(result.config).decode("utf-8"))
    assert payload["model_version"] == MODEL_VERSION
    assert payload["time_step_min"] == 1
    restored = SimulationConfig.from_dict(payload["config"])
    assert restored == result.config


def test_html_report_contents(result):
    comp = run_comparison(result.config)
    report = build_html_report(result, comp)
    assert report.startswith("<!DOCTYPE html>")
    assert "смоделированы" in report
    assert MODEL_VERSION in report
    assert f"зерно генератора: <b>{result.config.seed}</b>" in report
    assert "Допущения модели" in report
    assert "Сравнение сценариев" in report
    assert f"{result.total_energy_kwh:.4f}" in report


def test_html_report_escapes_content(result):
    result.events.loc[len(result.events)] = [0, "<script>alert(1)</script>", "Включён", "x & y", "r"]
    try:
        report = build_html_report(result)
    finally:
        result.events.drop(index=result.events.index[-1], inplace=True)
    assert "<script>alert(1)</script>" not in report
    assert "&lt;script&gt;" in report
    assert "x &amp; y" in report
