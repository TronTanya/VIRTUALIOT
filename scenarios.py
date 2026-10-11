"""Пресеты параметров и сравнительные эксперименты.

Пресеты содержат только параметры модели. Все результаты вычисляются
при запуске моделирования.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from controller import DEVICE_NAMES, ConfigError, DeviceStates
from simulation import (
    DEVICES,
    SimulationConfig,
    SimulationResult,
    generate_noise,
    run_simulation,
)

PRESETS: dict[str, dict[str, Any]] = {
    "Холодное помещение": {
        "description": (
            "Зимний день: в помещении 15 °C, на улице −5 °C. "
            "Проверяется работа обогревателя и гистерезиса 20…22 °C."
        ),
        "overrides": {
            "initial_temperature": 15.0,
            "initial_humidity": 45.0,
            "outdoor_temperature": -5.0,
            "outdoor_humidity": 50.0,
            "natural_illuminance": 500.0,
            "natural_variation": 0.0,
            "duration_min": 240,
            "seed": 2026,
            "thresholds": {"heater_on_below": 20.0, "heater_off_above": 22.0},
            "targets": {"temp_min": 20.0, "temp_max": 23.0},
        },
    },
    "Высокая влажность": {
        "description": (
            "После влажной уборки влажность 80 %, на улице сухо (40 %). "
            "Вентиляция снижает влажность, но одновременно охлаждает помещение."
        ),
        "overrides": {
            "initial_temperature": 22.0,
            "initial_humidity": 80.0,
            "outdoor_temperature": 15.0,
            "outdoor_humidity": 40.0,
            "natural_illuminance": 500.0,
            "natural_variation": 0.0,
            "duration_min": 180,
            "seed": 7,
            "thresholds": {"fan_off_below": 50.0, "fan_on_above": 60.0},
            "targets": {"hum_min": 40.0, "hum_max": 60.0},
        },
    },
    "Недостаточное освещение": {
        "description": (
            "Пасмурный день: естественный свет колеблется около 260 лк "
            "(±50 %, период 120 мин). Лампа управляется по датчику естественного света."
        ),
        "overrides": {
            "initial_temperature": 21.0,
            "initial_humidity": 50.0,
            "outdoor_temperature": 18.0,
            "outdoor_humidity": 50.0,
            "natural_illuminance": 260.0,
            "natural_variation": 0.5,
            "duration_min": 240,
            "seed": 11,
            "thresholds": {"lamp_on_below": 250.0, "lamp_off_above": 350.0},
            "targets": {"lux_min": 300.0, "lux_max": 750.0},
            "coefficients": {"natural_period_min": 120.0, "lamp_lux": 400.0},
        },
    },
}


def preset_config(name: str, base: SimulationConfig | None = None) -> SimulationConfig:
    if name not in PRESETS:
        raise ConfigError(f"Неизвестный пресет: {name}.")
    base = base or SimulationConfig()
    config = base.with_overrides(**PRESETS[name]["overrides"])
    config.validate()
    return config


@dataclass
class ComparisonResult:
    baseline: SimulationResult
    automated: SimulationResult
    table: pd.DataFrame


def metrics_table(results: dict[str, SimulationResult]) -> pd.DataFrame:
    """Таблица показателей; столбцы — сценарии."""
    rows: list[tuple[str, str, list[float]]] = [
        ("Время в целевом диапазоне: температура", "%", [r.in_range_pct["temperature"] for r in results.values()]),
        ("Время в целевом диапазоне: влажность", "%", [r.in_range_pct["humidity"] for r in results.values()]),
        ("Время в целевом диапазоне: освещённость", "%", [r.in_range_pct["illuminance"] for r in results.values()]),
    ]
    for device in DEVICES:
        rows.append((f"Время работы: {DEVICE_NAMES[device]}", "мин",
                     [r.device_metrics[device].enabled_minutes for r in results.values()]))
    for device in DEVICES:
        rows.append((f"Переключения: {DEVICE_NAMES[device]}", "шт.",
                     [r.device_metrics[device].switches for r in results.values()]))
    for device in DEVICES:
        rows.append((f"Электроэнергия (оценка): {DEVICE_NAMES[device]}", "кВт·ч",
                     [round(r.device_metrics[device].energy_kwh, 4) for r in results.values()]))
    rows.append(("Электроэнергия (оценка): всего", "кВт·ч",
                 [round(r.total_energy_kwh, 4) for r in results.values()]))
    data = {"Показатель": [r[0] for r in rows], "Единица": [r[1] for r in rows]}
    for i, name in enumerate(results):
        data[name] = [round(float(r[2][i]), 4) for r in rows]
    return pd.DataFrame(data)


BASELINE_LABEL = "Автоматизация выключена"
AUTOMATED_LABEL = "Автоматизация включена"


def run_comparison(config: SimulationConfig) -> ComparisonResult:
    """Парное сравнение: все устройства выключены vs автоматизация.

    Оба прогона используют одинаковые начальные условия, длительность,
    коэффициенты и одни и те же заранее сгенерированные шумы датчиков.
    """
    config.validate()
    noise = generate_noise(int(config.seed), int(config.duration_min) + 1)
    baseline_cfg = config.with_overrides(automation_enabled=False, manual_states=DeviceStates())
    automated_cfg = config.with_overrides(automation_enabled=True)
    baseline = run_simulation(baseline_cfg, noise)
    automated = run_simulation(automated_cfg, noise)
    table = metrics_table({BASELINE_LABEL: baseline, AUTOMATED_LABEL: automated})
    table["Разница"] = (table[AUTOMATED_LABEL] - table[BASELINE_LABEL]).round(4)
    return ComparisonResult(baseline, automated, table)


_THRESHOLD_FIELDS = {
    "heater": ("heater_on_below", "heater_off_above", "temperature"),
    "fan": ("fan_off_below", "fan_on_above", "humidity"),
    "lamp": ("lamp_on_below", "lamp_off_above", "illuminance"),
}


@dataclass
class HysteresisExperiment:
    device: str
    narrow: SimulationResult
    wide: SimulationResult
    table: pd.DataFrame


def run_hysteresis_experiment(
    config: SimulationConfig, device: str, center: float, narrow_width: float, wide_width: float
) -> HysteresisExperiment:
    """Сравнивает узкую и широкую полосу гистерезиса с общим центром и общим шумом."""
    if device not in _THRESHOLD_FIELDS:
        raise ConfigError(f"Неизвестное устройство: {device}.")
    if not 0 < narrow_width < wide_width:
        raise ConfigError("Ширина узкой полосы должна быть больше 0 и меньше ширины широкой полосы.")
    lo_field, hi_field, quantity = _THRESHOLD_FIELDS[device]
    noise = generate_noise(int(config.seed), int(config.duration_min) + 1)
    results: dict[str, SimulationResult] = {}
    for label, width in (("Узкая полоса", narrow_width), ("Широкая полоса", wide_width)):
        cfg = config.with_overrides(
            automation_enabled=True,
            thresholds={lo_field: center - width / 2, hi_field: center + width / 2},
        )
        results[f"{label} ({width:g})"] = run_simulation(cfg, noise)
    narrow, wide = results.values()
    names = list(results)
    table = pd.DataFrame(
        {
            "Показатель": [
                f"Переключения: {DEVICE_NAMES[device]}",
                f"Время работы: {DEVICE_NAMES[device]}",
                "Время в целевом диапазоне",
                f"Электроэнергия (оценка): {DEVICE_NAMES[device]}",
            ],
            "Единица": ["шт.", "мин", "%", "кВт·ч"],
            names[0]: [
                narrow.device_metrics[device].switches,
                narrow.device_metrics[device].enabled_minutes,
                round(narrow.in_range_pct[quantity], 2),
                round(narrow.device_metrics[device].energy_kwh, 4),
            ],
            names[1]: [
                wide.device_metrics[device].switches,
                wide.device_metrics[device].enabled_minutes,
                round(wide.in_range_pct[quantity], 2),
                round(wide.device_metrics[device].energy_kwh, 4),
            ],
        }
    )
    return HysteresisExperiment(device, narrow, wide, table)
