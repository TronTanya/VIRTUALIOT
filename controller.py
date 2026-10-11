"""Правила автоматизации с гистерезисом.

Модуль не зависит от интерфейса и от модели среды: контроллер получает
измеренные показания датчиков и текущие состояния устройств и возвращает
новые состояния вместе с объяснением причин переключения.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConfigError(ValueError):
    """Ошибка конфигурации. Содержит список сообщений на русском языке."""

    def __init__(self, messages: list[str] | str):
        if isinstance(messages, str):
            messages = [messages]
        self.messages = list(messages)
        super().__init__("; ".join(self.messages))


class Mode(str, Enum):
    """Направление срабатывания правила."""

    ON_BELOW = "on_below"  # включить ниже нижнего порога (обогреватель, лампа)
    ON_ABOVE = "on_above"  # включить выше верхнего порога (вентиляция)


DEVICE_NAMES: dict[str, str] = {
    "heater": "Обогреватель",
    "fan": "Вентиляция",
    "lamp": "Лампа",
}

STATE_LABELS: dict[bool, str] = {True: "Включён", False: "Выключен"}


@dataclass(frozen=True)
class HysteresisRule:
    """Двухпороговое правило.

    Граничное поведение: сравнения строгие. Значение, в точности равное
    порогу, считается находящимся внутри полосы гистерезиса, поэтому
    состояние устройства сохраняется.
    """

    lower: float
    upper: float
    mode: Mode

    def validate(self, name: str) -> list[str]:
        if not self.lower < self.upper:
            return [
                f"{name}: нижний порог ({self.lower:g}) должен быть строго меньше "
                f"верхнего ({self.upper:g})."
            ]
        return []

    def next_state(self, current: bool, reading: float) -> tuple[bool, str | None]:
        """Возвращает новое состояние и причину, если состояние изменилось."""
        if self.mode is Mode.ON_BELOW:
            if reading < self.lower:
                new, reason = True, f"ниже порога включения {self.lower:g}"
            elif reading > self.upper:
                new, reason = False, f"выше порога выключения {self.upper:g}"
            else:
                return current, None
        else:
            if reading > self.upper:
                new, reason = True, f"выше порога включения {self.upper:g}"
            elif reading < self.lower:
                new, reason = False, f"ниже порога выключения {self.lower:g}"
            else:
                return current, None
        return new, (reason if new != current else None)


@dataclass(frozen=True)
class Thresholds:
    """Пороги автоматизации (в единицах соответствующих датчиков)."""

    heater_on_below: float = 20.0  # °C
    heater_off_above: float = 22.0  # °C
    fan_off_below: float = 50.0  # %
    fan_on_above: float = 60.0  # %
    lamp_on_below: float = 250.0  # лк, по датчику естественного света
    lamp_off_above: float = 350.0  # лк, по датчику естественного света

    def rules(self) -> dict[str, HysteresisRule]:
        return {
            "heater": HysteresisRule(self.heater_on_below, self.heater_off_above, Mode.ON_BELOW),
            "fan": HysteresisRule(self.fan_off_below, self.fan_on_above, Mode.ON_ABOVE),
            "lamp": HysteresisRule(self.lamp_on_below, self.lamp_off_above, Mode.ON_BELOW),
        }

    def validate(self) -> list[str]:
        errors: list[str] = []
        for key, rule in self.rules().items():
            errors.extend(rule.validate(f"Пороги «{DEVICE_NAMES[key]}»"))
        if not -20.0 <= self.heater_on_below <= 50.0 or not -20.0 <= self.heater_off_above <= 50.0:
            errors.append("Пороги обогревателя должны лежать в диапазоне −20…50 °C.")
        for value in (self.fan_off_below, self.fan_on_above):
            if not 0.0 <= value <= 100.0:
                errors.append("Пороги вентиляции должны лежать в диапазоне 0…100 %.")
                break
        for value in (self.lamp_on_below, self.lamp_off_above):
            if not 0.0 <= value <= 100_000.0:
                errors.append("Пороги лампы должны лежать в диапазоне 0…100 000 лк.")
                break
        return errors


@dataclass(frozen=True)
class DeviceStates:
    heater: bool = False
    fan: bool = False
    lamp: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {"heater": self.heater, "fan": self.fan, "lamp": self.lamp}


@dataclass(frozen=True)
class SensorReadings:
    """Измеренные (смоделированные) показания датчиков в один момент времени."""

    temperature: float  # °C
    humidity: float  # %
    illuminance_total: float  # лк, общая освещённость помещения
    illuminance_ambient: float  # лк, естественный свет без вклада лампы


@dataclass(frozen=True)
class Decision:
    """Изменение состояния устройства с объяснением."""

    device: str
    new_state: bool
    reading_label: str
    reason: str


_READING_FOR_DEVICE = {
    "heater": ("temperature", "Температура", "°C", "Температура"),
    "fan": ("humidity", "Влажность", "%", "Влажность"),
    "lamp": ("illuminance_ambient", "Естественный свет", "лк", "Естественный свет"),
}


def format_reading(device: str, readings: SensorReadings) -> str:
    attr, label, unit, _ = _READING_FOR_DEVICE[device]
    return f"{label}: {getattr(readings, attr):.1f} {unit}"


class AutomationController:
    """Контроллер с гистерезисом для трёх устройств.

    Лампа управляется по датчику естественного света, который не видит
    собственный свет лампы. Иначе включённая лампа поднимала бы показание
    выше верхнего порога и сразу выключалась бы, вызывая циклические
    переключения.
    """

    def __init__(self, thresholds: Thresholds):
        errors = thresholds.validate()
        if errors:
            raise ConfigError(errors)
        self.thresholds = thresholds
        self._rules = thresholds.rules()

    def evaluate(
        self, readings: SensorReadings, current: DeviceStates
    ) -> tuple[DeviceStates, list[Decision]]:
        new_states = current.as_dict()
        decisions: list[Decision] = []
        for device, rule in self._rules.items():
            attr, _, _, subject = _READING_FOR_DEVICE[device]
            state, reason = rule.next_state(new_states[device], getattr(readings, attr))
            if reason is not None:
                new_states[device] = state
                decisions.append(
                    Decision(device, state, format_reading(device, readings), f"{subject} {reason}")
                )
        return DeviceStates(**new_states), decisions


class ManualController:
    """Фиксированные состояния устройств на весь прогон (автоматизация выключена)."""

    def __init__(self, states: DeviceStates):
        self.states = states

    def evaluate(
        self, readings: SensorReadings, current: DeviceStates
    ) -> tuple[DeviceStates, list[Decision]]:
        decisions = [
            Decision(
                device,
                self.states.as_dict()[device],
                format_reading(device, readings),
                "Ручной режим: фиксированное состояние",
            )
            for device, value in current.as_dict().items()
            if self.states.as_dict()[device] != value
        ]
        return self.states, decisions
