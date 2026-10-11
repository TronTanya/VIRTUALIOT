"""Дискретная модель умного помещения (шаг 1 минута).

Модель учебная: она объясняет качественное поведение системы
«датчики — контроллер — устройства — среда» и не предназначена для
точного прогноза параметров реального здания.

Уравнения (t — номер минуты, Δt = 1 мин):

    T[t+1] = T[t] + (k_T + k_TV·v[t]) · (T_out − T[t]) + h · q[t]
    H[t+1] = H[t] + (k_H + k_HV·v[t]) · (H_out − H[t])
    E_nat[t] = E_0 · (1 + a · sin(2π·t / P))
    E[t]     = E_nat[t] + L · l[t−1]

где q, v, l ∈ {0, 1} — состояния обогревателя, вентиляции и лампы.
Показания датчиков = состояние среды + шум; шум не меняет состояние среды.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from controller import (
    DEVICE_NAMES,
    AutomationController,
    ConfigError,
    DeviceStates,
    ManualController,
    SensorReadings,
    Thresholds,
)

MODEL_VERSION = "1.0.0"
TIME_STEP_MIN = 1
MIN_DURATION_MIN = 10
MAX_DURATION_MIN = 1440
DEVICES = ("heater", "fan", "lamp")
SENSOR_STREAMS = ("temperature", "humidity", "illuminance_total", "illuminance_ambient")


@dataclass(frozen=True)
class ModelCoefficients:
    """Коэффициенты модели. Все скорости заданы на один шаг (1 мин)."""

    k_temp: float = 0.01  # 1/мин, обмен теплом с улицей без вентиляции
    k_temp_vent: float = 0.03  # 1/мин, добавка к обмену при работе вентиляции
    heater_gain: float = 0.4  # °C/мин, нагрев при включённом обогревателе
    k_hum: float = 0.005  # 1/мин, обмен влагой с улицей без вентиляции
    k_hum_vent: float = 0.05  # 1/мин, добавка к обмену при работе вентиляции
    lamp_lux: float = 400.0  # лк, вклад лампы в освещённость
    natural_period_min: float = 120.0  # мин, период колебаний естественного света
    noise_temp: float = 0.1  # °C, СКО шума датчика температуры
    noise_hum: float = 0.5  # %, СКО шума датчика влажности
    noise_lux: float = 5.0  # лк, СКО шума датчиков освещённости

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not 0.0 < self.k_temp < 1.0:
            errors.append("k_T должен быть в интервале (0; 1) 1/мин.")
        if not 0.0 <= self.k_temp_vent < 1.0:
            errors.append("k_TV должен быть в интервале [0; 1) 1/мин.")
        if self.k_temp + self.k_temp_vent > 1.0:
            errors.append(
                "k_T + k_TV не должна превышать 1: иначе температура перескакивает "
                "через уличное значение за один шаг."
            )
        if not 0.0 < self.k_hum < 1.0:
            errors.append("k_H должен быть в интервале (0; 1) 1/мин.")
        if not 0.0 <= self.k_hum_vent < 1.0:
            errors.append("k_HV должен быть в интервале [0; 1) 1/мин.")
        if self.k_hum + self.k_hum_vent > 1.0:
            errors.append(
                "k_H + k_HV не должна превышать 1: иначе влажность перескакивает "
                "через уличное значение за один шаг."
            )
        if not 0.0 <= self.heater_gain <= 2.0:
            errors.append("Нагрев обогревателя должен быть в диапазоне 0…2 °C/мин.")
        if not 0.0 <= self.lamp_lux <= 5000.0:
            errors.append("Вклад лампы должен быть в диапазоне 0…5000 лк.")
        if not 10.0 <= self.natural_period_min <= 1440.0:
            errors.append("Период колебаний естественного света: 10…1440 мин.")
        if not 0.0 <= self.noise_temp <= 2.0:
            errors.append("Шум датчика температуры: 0…2 °C.")
        if not 0.0 <= self.noise_hum <= 10.0:
            errors.append("Шум датчика влажности: 0…10 %.")
        if not 0.0 <= self.noise_lux <= 200.0:
            errors.append("Шум датчиков освещённости: 0…200 лк.")
        return errors


@dataclass(frozen=True)
class DevicePower:
    """Номинальная мощность устройств, Вт (принимается постоянной)."""

    heater_w: float = 1500.0
    fan_w: float = 40.0
    lamp_w: float = 60.0

    def validate(self) -> list[str]:
        errors = []
        for name, value in (("обогревателя", self.heater_w), ("вентиляции", self.fan_w), ("лампы", self.lamp_w)):
            if not 0.0 <= value <= 5000.0:
                errors.append(f"Мощность {name} должна быть в диапазоне 0…5000 Вт.")
        return errors

    def watts(self, device: str) -> float:
        return {"heater": self.heater_w, "fan": self.fan_w, "lamp": self.lamp_w}[device]


@dataclass(frozen=True)
class TargetRanges:
    """Целевые диапазоны для оценки условий (границы включаются)."""

    temp_min: float = 20.0
    temp_max: float = 23.0
    hum_min: float = 40.0
    hum_max: float = 60.0
    lux_min: float = 300.0
    lux_max: float = 750.0

    def validate(self) -> list[str]:
        errors = []
        for name, lo, hi in (
            ("температуры", self.temp_min, self.temp_max),
            ("влажности", self.hum_min, self.hum_max),
            ("освещённости", self.lux_min, self.lux_max),
        ):
            if not lo < hi:
                errors.append(f"Целевой диапазон {name}: минимум должен быть меньше максимума.")
        if self.hum_min < 0 or self.hum_max > 100:
            errors.append("Целевой диапазон влажности должен лежать в пределах 0…100 %.")
        if self.lux_min < 0:
            errors.append("Целевая освещённость не может быть отрицательной.")
        return errors


@dataclass(frozen=True)
class SimulationConfig:
    initial_temperature: float = 18.0  # °C
    initial_humidity: float = 65.0  # %
    outdoor_temperature: float = 5.0  # °C
    outdoor_humidity: float = 45.0  # %
    natural_illuminance: float = 400.0  # лк, среднее значение
    natural_variation: float = 0.0  # доля 0…1, амплитуда колебаний естественного света
    duration_min: int = 180
    seed: int = 42
    noise_enabled: bool = True
    automation_enabled: bool = True
    thresholds: Thresholds = field(default_factory=Thresholds)
    manual_states: DeviceStates = field(default_factory=DeviceStates)
    targets: TargetRanges = field(default_factory=TargetRanges)
    coefficients: ModelCoefficients = field(default_factory=ModelCoefficients)
    power: DevicePower = field(default_factory=DevicePower)

    def validate(self) -> None:
        errors: list[str] = []
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                errors.append(f"Параметр {name} должен быть конечным числом.")
        if not -20.0 <= self.initial_temperature <= 50.0:
            errors.append("Начальная температура должна быть в диапазоне −20…50 °C.")
        if not 0.0 <= self.initial_humidity <= 100.0:
            errors.append("Начальная влажность должна быть в диапазоне 0…100 %.")
        if not -40.0 <= self.outdoor_temperature <= 50.0:
            errors.append("Уличная температура должна быть в диапазоне −40…50 °C.")
        if not 0.0 <= self.outdoor_humidity <= 100.0:
            errors.append("Уличная влажность должна быть в диапазоне 0…100 %.")
        if not 0.0 <= self.natural_illuminance <= 100_000.0:
            errors.append("Естественная освещённость должна быть в диапазоне 0…100 000 лк.")
        if not 0.0 <= self.natural_variation <= 1.0:
            errors.append("Амплитуда колебаний естественного света должна быть 0…100 %.")
        if isinstance(self.duration_min, bool) or not isinstance(self.duration_min, (int, np.integer)):
            errors.append("Длительность должна быть целым числом минут.")
        elif not MIN_DURATION_MIN <= self.duration_min <= MAX_DURATION_MIN:
            errors.append(f"Длительность должна быть {MIN_DURATION_MIN}…{MAX_DURATION_MIN} мин.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or not 0 <= self.seed < 2**32:
            errors.append("Зерно генератора должно быть целым числом 0…4 294 967 295.")
        errors += self.thresholds.validate()
        errors += self.targets.validate()
        errors += self.coefficients.validate()
        errors += self.power.validate()
        if errors:
            raise ConfigError(errors)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationConfig":
        nested = {
            "thresholds": Thresholds,
            "manual_states": DeviceStates,
            "targets": TargetRanges,
            "coefficients": ModelCoefficients,
            "power": DevicePower,
        }
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Неизвестные параметры конфигурации: {', '.join(sorted(unknown))}.")
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in nested and isinstance(value, dict):
                kwargs[key] = nested[key](**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def with_overrides(self, **overrides: Any) -> "SimulationConfig":
        """Создаёт копию; вложенные объекты можно переопределять словарём."""
        kwargs: dict[str, Any] = {}
        for key, value in overrides.items():
            current = getattr(self, key)
            if is_dataclass(current) and isinstance(value, dict):
                kwargs[key] = replace(current, **value)
            else:
                kwargs[key] = value
        return replace(self, **kwargs)


@dataclass(frozen=True)
class NoiseSequences:
    """Стандартные нормальные шумы, по одному независимому потоку на датчик.

    Масштабирование на СКО выполняется при моделировании, поэтому одни и те
    же последовательности можно использовать в парных сравнениях.
    """

    temperature: np.ndarray
    humidity: np.ndarray
    illuminance_total: np.ndarray
    illuminance_ambient: np.ndarray

    def __len__(self) -> int:
        return len(self.temperature)


def generate_noise(seed: int, n_observations: int) -> NoiseSequences:
    children = np.random.SeedSequence(seed).spawn(len(SENSOR_STREAMS))
    streams = {
        name: np.random.default_rng(child).standard_normal(n_observations)
        for name, child in zip(SENSOR_STREAMS, children)
    }
    return NoiseSequences(**streams)


@dataclass
class DeviceMetrics:
    enabled_minutes: int
    switches: int
    energy_kwh: float


@dataclass
class SimulationResult:
    config: SimulationConfig
    frame: pd.DataFrame
    events: pd.DataFrame
    device_metrics: dict[str, DeviceMetrics]
    in_range_pct: dict[str, float]
    final_readings: SensorReadings
    final_states: DeviceStates

    @property
    def total_energy_kwh(self) -> float:
        return sum(m.energy_kwh for m in self.device_metrics.values())


def energy_kwh(power_w: float, enabled_minutes: float) -> float:
    """energy_kWh = power_W × enabled_minutes / 60 / 1000."""
    return power_w * enabled_minutes / 60.0 / 1000.0


def natural_illuminance(config: SimulationConfig, t: int) -> float:
    phase = 2.0 * math.pi * t / config.coefficients.natural_period_min
    return max(0.0, config.natural_illuminance * (1.0 + config.natural_variation * math.sin(phase)))


def step_environment(
    temperature: float, humidity: float, states: DeviceStates, config: SimulationConfig
) -> tuple[float, float]:
    """Продвигает состояние среды на один шаг (1 мин)."""
    c = config.coefficients
    vent = 1.0 if states.fan else 0.0
    heat = 1.0 if states.heater else 0.0
    k_t = c.k_temp + c.k_temp_vent * vent
    k_h = c.k_hum + c.k_hum_vent * vent
    new_t = temperature + k_t * (config.outdoor_temperature - temperature) + c.heater_gain * heat
    new_h = humidity + k_h * (config.outdoor_humidity - humidity)
    return new_t, min(100.0, max(0.0, new_h))


def measure(
    temperature: float,
    humidity: float,
    ambient_lux: float,
    total_lux: float,
    noise: NoiseSequences,
    t: int,
    config: SimulationConfig,
) -> SensorReadings:
    c = config.coefficients
    scale = 1.0 if config.noise_enabled else 0.0
    return SensorReadings(
        temperature=temperature + scale * c.noise_temp * float(noise.temperature[t]),
        humidity=min(100.0, max(0.0, humidity + scale * c.noise_hum * float(noise.humidity[t]))),
        illuminance_total=max(0.0, total_lux + scale * c.noise_lux * float(noise.illuminance_total[t])),
        illuminance_ambient=max(0.0, ambient_lux + scale * c.noise_lux * float(noise.illuminance_ambient[t])),
    )


def _in_range(values: pd.Series, lo: float, hi: float) -> float:
    return float(((values >= lo) & (values <= hi)).mean() * 100.0)


def run_simulation(config: SimulationConfig, noise: NoiseSequences | None = None) -> SimulationResult:
    """Выполняет моделирование.

    Наблюдения выполняются в моменты t = 0…N (N + 1 строк). Строка t < N
    описывает показания в момент t и состояния устройств на интервале
    [t; t+1). Последняя строка — конечное наблюдение без следующего интервала.
    """
    config.validate()
    n = int(config.duration_min)
    if noise is None:
        noise = generate_noise(int(config.seed), n + 1)
    elif len(noise) < n + 1:
        raise ConfigError("Длина последовательностей шума меньше числа наблюдений.")

    controller = (
        AutomationController(config.thresholds)
        if config.automation_enabled
        else ManualController(config.manual_states)
    )
    lamp_lux = config.coefficients.lamp_lux

    temperature = float(config.initial_temperature)
    humidity = float(config.initial_humidity)
    states = DeviceStates()  # до начала прогона все устройства выключены

    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    readings = None
    for t in range(n + 1):
        ambient = natural_illuminance(config, t)
        total = ambient + (lamp_lux if states.lamp else 0.0)
        # 1. Показания датчиков
        readings = measure(temperature, humidity, ambient, total, noise, t, config)
        row: dict[str, Any] = {
            "t": t * TIME_STEP_MIN,
            "record": "Начальное наблюдение" if t == 0 else ("Конечное наблюдение" if t == n else "Наблюдение"),
            "temp_true": temperature,
            "hum_true": humidity,
            "lux_true": total,
            "temp": readings.temperature,
            "hum": readings.humidity,
            "lux": readings.illuminance_total,
            "lux_ambient": readings.illuminance_ambient,
        }
        if t < n:
            # 2. Правила автоматизации
            states, decisions = controller.evaluate(readings, states)
            for d in decisions:
                events.append(
                    {
                        "t": t * TIME_STEP_MIN,
                        "device": DEVICE_NAMES[d.device],
                        "state": "Включён" if d.new_state else "Выключен",
                        "reading": d.reading_label,
                        "reason": d.reason,
                    }
                )
            # 3–4. Воздействие устройств на интервале и продвижение среды
            for device, value in states.as_dict().items():
                row[device] = value
            temperature, humidity = step_environment(temperature, humidity, states, config)
        else:
            for device in DEVICES:
                row[device] = pd.NA
        # 5. Запись результата
        rows.append(row)

    frame = pd.DataFrame(rows)
    for device in DEVICES:
        frame[device] = frame[device].astype("boolean")
    events_df = pd.DataFrame(events, columns=["t", "device", "state", "reading", "reason"])

    device_metrics: dict[str, DeviceMetrics] = {}
    for device in DEVICES:
        minutes = int(frame[device].iloc[:n].sum()) * TIME_STEP_MIN
        switches = int((events_df["device"] == DEVICE_NAMES[device]).sum())
        device_metrics[device] = DeviceMetrics(minutes, switches, energy_kwh(config.power.watts(device), minutes))

    tg = config.targets
    in_range = {
        "temperature": _in_range(frame["temp"], tg.temp_min, tg.temp_max),
        "humidity": _in_range(frame["hum"], tg.hum_min, tg.hum_max),
        "illuminance": _in_range(frame["lux"], tg.lux_min, tg.lux_max),
    }
    assert readings is not None
    return SimulationResult(config, frame, events_df, device_metrics, in_range, readings, states)
