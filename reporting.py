"""Экспорт результатов: CSV (UTF-8 с BOM), JSON и автономный HTML-отчёт."""

from __future__ import annotations

import html
import json
from datetime import datetime
from typing import Any

import pandas as pd

from controller import DEVICE_NAMES
from scenarios import ComparisonResult
from simulation import DEVICES, MODEL_VERSION, TIME_STEP_MIN, SimulationConfig, SimulationResult

CSV_ENCODING = "utf-8-sig"

FRAME_COLUMNS = {
    "t": "Время, мин",
    "record": "Тип записи",
    "temp": "Температура (датчик), °C",
    "hum": "Влажность (датчик), %",
    "lux": "Освещённость помещения (датчик), лк",
    "lux_ambient": "Естественный свет (датчик для лампы), лк",
    "temp_true": "Температура (состояние модели), °C",
    "hum_true": "Влажность (состояние модели), %",
    "lux_true": "Освещённость (состояние модели), лк",
    "heater": "Обогреватель на интервале [t; t+1)",
    "fan": "Вентиляция на интервале [t; t+1)",
    "lamp": "Лампа на интервале [t; t+1)",
}

EVENT_COLUMNS = {
    "t": "Время, мин",
    "device": "Устройство",
    "state": "Новое состояние",
    "reading": "Показание",
    "reason": "Причина",
}

ASSUMPTIONS = [
    "Все показания смоделированы; приложение не подключено к реальному оборудованию.",
    "Шаг моделирования фиксирован: 1 минута. Модель дискретная и учебная.",
    "Температура и влажность релаксируют к уличным значениям; коэффициенты обмена постоянны.",
    "Влажность — учебное приближение: не моделируются абсолютная влажность, конденсация "
    "и зависимость относительной влажности от температуры.",
    "Шум датчиков влияет только на показания, но не на состояние среды.",
    "Лампа управляется по отдельному датчику естественного света, который не учитывает свет самой лампы.",
    "Электроэнергия оценивается при постоянной номинальной мощности: "
    "энергия, кВт·ч = мощность, Вт × время работы, мин / 60 / 1000. "
    "Это не прогноз счёта и не доказательство экономии.",
    "Модель не предназначена для точного прогноза условий в реальном здании.",
]


def _state_text(value: Any) -> str:
    if value is pd.NA or value is None:
        return "—"
    return "Включён" if bool(value) else "Выключен"


def frame_for_export(result: SimulationResult) -> pd.DataFrame:
    df = result.frame[list(FRAME_COLUMNS)].copy()
    for col in ("temp", "hum", "lux", "lux_ambient", "temp_true", "hum_true", "lux_true"):
        df[col] = df[col].round(3)
    for device in DEVICES:
        df[device] = df[device].map(_state_text).astype(object)
    return df.rename(columns=FRAME_COLUMNS)


def events_for_export(result: SimulationResult) -> pd.DataFrame:
    return result.events.rename(columns=EVENT_COLUMNS)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode(CSV_ENCODING)


def config_payload(config: SimulationConfig) -> dict[str, Any]:
    return {
        "model_version": MODEL_VERSION,
        "time_step_min": TIME_STEP_MIN,
        "note": "Параметры виртуальной модели. Данные смоделированы.",
        "config": config.to_dict(),
    }


def config_to_json(config: SimulationConfig) -> bytes:
    return json.dumps(config_payload(config), ensure_ascii=False, indent=2).encode("utf-8")


def summary_table(result: SimulationResult) -> pd.DataFrame:
    rows = []
    for device in DEVICES:
        m = result.device_metrics[device]
        rows.append(
            {
                "Устройство": DEVICE_NAMES[device],
                "Конечное состояние": _state_text(result.final_states.as_dict()[device]),
                "Время работы, мин": m.enabled_minutes,
                "Переключения, шт.": m.switches,
                "Мощность, Вт": result.config.power.watts(device),
                "Электроэнергия (оценка), кВт·ч": round(m.energy_kwh, 4),
            }
        )
    return pd.DataFrame(rows)


def _flatten(prefix: str, value: Any) -> list[tuple[str, str]]:
    if isinstance(value, dict):
        items: list[tuple[str, str]] = []
        for k, v in value.items():
            items += _flatten(f"{prefix}.{k}" if prefix else k, v)
        return items
    return [(prefix, str(value))]


def _table_html(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in df.itertuples(index=False)
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


_REPORT_CSS = """
body{font-family:Segoe UI,Arial,sans-serif;background:#F7F8FA;color:#202733;margin:0;padding:24px}
main{max-width:1100px;margin:0 auto;background:#FFFFFF;border:1px solid #E4E7EC;border-radius:6px;padding:24px 32px}
h1{margin:0 0 4px;font-size:24px} h2{font-size:18px;margin-top:28px;border-bottom:1px solid #E4E7EC;padding-bottom:4px}
p.sub,.muted{color:#667085} table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
th,td{border:1px solid #E4E7EC;padding:4px 8px;text-align:left} th{background:#F7F8FA}
.note{border-left:3px solid #B7791F;padding:6px 12px;background:#FFFFFF;color:#202733}
"""


def build_html_report(
    result: SimulationResult,
    comparison: ComparisonResult | None = None,
    generated_at: datetime | None = None,
) -> str:
    """Автономный HTML-отчёт. Все вставляемые значения экранируются."""
    cfg = result.config
    generated_at = generated_at or datetime.now()
    esc = html.escape
    params = pd.DataFrame(_flatten("", cfg.to_dict()), columns=["Параметр", "Значение"])
    in_range = pd.DataFrame(
        {
            "Величина": ["Температура", "Влажность", "Освещённость помещения"],
            "Целевой диапазон": [
                f"{cfg.targets.temp_min:g}…{cfg.targets.temp_max:g} °C",
                f"{cfg.targets.hum_min:g}…{cfg.targets.hum_max:g} %",
                f"{cfg.targets.lux_min:g}…{cfg.targets.lux_max:g} лк",
            ],
            "Доля показаний в диапазоне, %": [
                round(result.in_range_pct["temperature"], 2),
                round(result.in_range_pct["humidity"], 2),
                round(result.in_range_pct["illuminance"], 2),
            ],
        }
    )
    fr = result.final_readings
    final = pd.DataFrame(
        {
            "Показание": ["Температура", "Влажность", "Освещённость помещения", "Естественный свет (датчик для лампы)"],
            "Значение": [
                f"{fr.temperature:.2f} °C",
                f"{fr.humidity:.2f} %",
                f"{fr.illuminance_total:.1f} лк",
                f"{fr.illuminance_ambient:.1f} лк",
            ],
        }
    )
    events = events_for_export(result)
    events_html = _table_html(events) if len(events) else "<p class='muted'>Переключений не было.</p>"
    comparison_html = ""
    if comparison is not None:
        comparison_html = (
            "<h2>Сравнение сценариев</h2>"
            "<p class='muted'>Оба прогона используют одинаковые начальные условия, длительность, "
            "коэффициенты и последовательности шума датчиков. Улучшение условий может требовать "
            "дополнительной электроэнергии.</p>" + _table_html(comparison.table)
        )
    assumptions = "".join(f"<li>{esc(a)}</li>" for a in ASSUMPTIONS)
    mode = "включена" if cfg.automation_enabled else "выключена (ручные состояния)"
    return f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>VirtualIoT — отчёт о моделировании</title><style>{_REPORT_CSS}</style></head>
<body><main>
<h1>VirtualIoT</h1>
<p class="sub">Виртуальная лаборатория умного помещения — отчёт о моделировании</p>
<p class="note">Все показания смоделированы. Приложение не подключено к реальному оборудованию.</p>
<p>Версия модели: <b>{esc(MODEL_VERSION)}</b>; шаг: <b>{TIME_STEP_MIN} мин</b>;
длительность: <b>{esc(str(cfg.duration_min))} мин</b>; зерно генератора: <b>{esc(str(cfg.seed))}</b>;
автоматизация: <b>{esc(mode)}</b>; отчёт сформирован: {esc(generated_at.strftime("%Y-%m-%d %H:%M"))}.</p>
<h2>Конечные показания (момент t = {esc(str(cfg.duration_min))} мин)</h2>{_table_html(final)}
<h2>Устройства</h2>{_table_html(summary_table(result))}
<p class="muted">Электроэнергия оценена при постоянной мощности:
энергия, кВт·ч = мощность, Вт × время работы, мин / 60 / 1000. Итого: {result.total_energy_kwh:.4f} кВт·ч.</p>
<h2>Целевые диапазоны</h2>{_table_html(in_range)}
{comparison_html}
<h2>Журнал событий</h2>{events_html}
<h2>Параметры</h2>{_table_html(params)}
<h2>Допущения модели</h2><ul>{assumptions}</ul>
</main></body></html>"""
