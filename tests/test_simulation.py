import numpy as np
import pandas as pd
import pytest

from controller import ConfigError, DeviceStates
from scenarios import PRESETS, preset_config, run_comparison, run_hysteresis_experiment
from simulation import (
    SimulationConfig,
    energy_kwh,
    generate_noise,
    run_simulation,
    step_environment,
)


def quiet(**overrides):
    """Конфигурация без шума для детерминированных проверок."""
    return SimulationConfig(noise_enabled=False).with_overrides(**overrides)


# ---------------------------------------------------------------- валидация

@pytest.mark.parametrize(
    "overrides",
    [
        {"duration_min": 9},
        {"duration_min": 1441},
        {"duration_min": 30.5},
        {"initial_humidity": 101.0},
        {"outdoor_humidity": -1.0},
        {"initial_temperature": 80.0},
        {"natural_illuminance": -5.0},
        {"natural_variation": 1.5},
        {"seed": -1},
        {"initial_temperature": float("nan")},
        {"targets": {"temp_min": 23.0, "temp_max": 20.0}},
        {"thresholds": {"heater_on_below": 22.0, "heater_off_above": 22.0}},
        {"coefficients": {"k_temp": 0.0}},
        {"coefficients": {"k_temp": 0.6, "k_temp_vent": 0.5}},
        {"coefficients": {"k_hum": 0.7, "k_hum_vent": 0.4}},
        {"power": {"heater_w": -1.0}},
    ],
)
def test_invalid_config_rejected(overrides):
    with pytest.raises(ConfigError):
        run_simulation(SimulationConfig().with_overrides(**overrides))


def test_all_presets_valid():
    for name in PRESETS:
        preset_config(name).validate()


# ---------------------------------------------------------------- структура результата

def test_row_count_and_timestamps():
    res = run_simulation(SimulationConfig(duration_min=37))
    df = res.frame
    assert len(df) == 38
    assert df["t"].tolist() == list(range(38))
    assert df["record"].iloc[0] == "Начальное наблюдение"
    assert df["record"].iloc[-1] == "Конечное наблюдение"
    assert (df["record"].iloc[1:-1] == "Наблюдение").all()
    # у конечного наблюдения нет следующего интервала
    assert df[["heater", "fan", "lamp"]].iloc[-1].isna().all()
    assert df[["heater", "fan", "lamp"]].iloc[:-1].notna().all().all()


def test_reproducible_with_fixed_seed():
    cfg = SimulationConfig(seed=123)
    a, b = run_simulation(cfg), run_simulation(cfg)
    pd.testing.assert_frame_equal(a.frame, b.frame)
    pd.testing.assert_frame_equal(a.events, b.events)
    c = run_simulation(cfg.with_overrides(seed=124))
    assert not np.allclose(a.frame["temp"], c.frame["temp"])


def test_noise_streams_independent_and_reproducible():
    n1, n2 = generate_noise(5, 100), generate_noise(5, 100)
    np.testing.assert_array_equal(n1.temperature, n2.temperature)
    assert not np.allclose(n1.temperature, n1.humidity)
    assert not np.allclose(n1.illuminance_total, n1.illuminance_ambient)


def test_noise_does_not_alter_environment_state():
    # Ручной режим: состояния устройств не зависят от показаний,
    # поэтому состояние среды должно совпадать с прогоном без шума.
    base = dict(automation_enabled=False, manual_states=DeviceStates(heater=True, fan=True, lamp=True))
    noisy = run_simulation(SimulationConfig(noise_enabled=True).with_overrides(**base))
    clean = run_simulation(SimulationConfig(noise_enabled=False).with_overrides(**base))
    for col in ("temp_true", "hum_true", "lux_true"):
        np.testing.assert_allclose(noisy.frame[col], clean.frame[col], rtol=0, atol=1e-12)
    assert not np.allclose(noisy.frame["temp"], noisy.frame["temp_true"])
    np.testing.assert_allclose(clean.frame["temp"], clean.frame["temp_true"], atol=1e-12)


def test_humidity_and_illuminance_bounds():
    cfg = SimulationConfig().with_overrides(
        initial_humidity=99.5, outdoor_humidity=100.0, natural_illuminance=1.0, natural_variation=1.0,
        coefficients={"noise_hum": 10.0, "noise_lux": 200.0},
        automation_enabled=False, manual_states=DeviceStates(fan=True),
    )
    df = run_simulation(cfg).frame
    assert df["hum"].between(0.0, 100.0).all()
    assert (df["lux"] >= 0).all() and (df["lux_ambient"] >= 0).all()
    assert df["hum"].max() == pytest.approx(100.0)  # ограничение действительно срабатывало
    assert df["lux_ambient"].min() == pytest.approx(0.0)


# ---------------------------------------------------------------- физика

def test_temperature_relaxes_toward_outdoor_without_devices():
    cfg = quiet(initial_temperature=20.0, outdoor_temperature=0.0, duration_min=60,
                automation_enabled=False, coefficients={"k_temp": 0.02})
    temp = run_simulation(cfg).frame["temp"].to_numpy()
    assert np.all(np.diff(temp) < 0)
    assert temp[-1] > 0.0
    assert temp[-1] == pytest.approx(20.0 * 0.98**60, rel=1e-9)


def test_heater_and_ventilation_effects_single_step():
    cfg = quiet(outdoor_temperature=0.0, outdoor_humidity=40.0,
                coefficients={"k_temp": 0.01, "k_temp_vent": 0.03, "heater_gain": 0.4,
                              "k_hum": 0.005, "k_hum_vent": 0.05})
    t, h = step_environment(20.0, 80.0, DeviceStates(heater=True), cfg)
    assert t == pytest.approx(20.0 - 0.01 * 20.0 + 0.4)
    assert h == pytest.approx(80.0 - 0.005 * 40.0)
    t, h = step_environment(20.0, 80.0, DeviceStates(fan=True), cfg)
    assert t == pytest.approx(20.0 - 0.04 * 20.0)
    assert h == pytest.approx(80.0 - 0.055 * 40.0)


def test_no_overshoot_at_maximum_exchange():
    cfg = quiet(outdoor_temperature=0.0, coefficients={"k_temp": 0.5, "k_temp_vent": 0.5})
    t, _ = step_environment(20.0, 50.0, DeviceStates(fan=True), cfg)
    assert t == pytest.approx(0.0, abs=1e-12)


def test_lamp_does_not_cycle_from_own_light():
    # Естественный свет постоянно ниже порога; лампа добавляет 400 лк,
    # но управляется по естественному свету и не должна выключаться.
    cfg = quiet(natural_illuminance=200.0, duration_min=60,
                thresholds={"lamp_on_below": 250.0, "lamp_off_above": 350.0})
    res = run_simulation(cfg)
    assert res.device_metrics["lamp"].switches == 1
    assert res.device_metrics["lamp"].enabled_minutes == 60
    assert res.frame["lux"].iloc[1] == pytest.approx(600.0)
    assert res.frame["lux_ambient"].iloc[1] == pytest.approx(200.0)


# ---------------------------------------------------------------- метрики

def test_enabled_durations_and_energy_manual():
    cfg = quiet(duration_min=90, automation_enabled=False,
                manual_states=DeviceStates(heater=True, fan=False, lamp=True),
                power={"heater_w": 1200.0, "lamp_w": 60.0})
    res = run_simulation(cfg)
    m = res.device_metrics
    assert m["heater"].enabled_minutes == 90
    assert m["fan"].enabled_minutes == 0
    assert m["heater"].switches == 1 and m["fan"].switches == 0
    assert m["heater"].energy_kwh == pytest.approx(1.8)
    assert m["lamp"].energy_kwh == pytest.approx(0.09)
    assert res.total_energy_kwh == pytest.approx(1.89)


def test_energy_formula():
    assert energy_kwh(1500.0, 60) == pytest.approx(1.5)
    assert energy_kwh(40.0, 30) == pytest.approx(0.02)
    assert energy_kwh(0.0, 1440) == 0.0


def test_switch_counts_and_durations_consistent_with_frame():
    res = run_simulation(preset_config("Холодное помещение"))
    n = res.config.duration_min
    for device in ("heater", "fan", "lamp"):
        states = [False] + res.frame[device].iloc[:n].astype(bool).tolist()
        transitions = sum(a != b for a, b in zip(states, states[1:]))
        assert res.device_metrics[device].switches == transitions
        assert 0 <= res.device_metrics[device].enabled_minutes <= n
        assert res.device_metrics[device].enabled_minutes == sum(states[1:])
    assert res.device_metrics["heater"].switches >= 3  # гистерезис действительно работал


def test_temperature_hysteresis_cycle_no_noise():
    cfg = preset_config("Холодное помещение").with_overrides(noise_enabled=False)
    res = run_simulation(cfg)
    df = res.frame
    # после первого достижения нижнего порога температура держится около полосы
    after = df["temp"].iloc[60:]
    assert after.min() > 19.5 and after.max() < 22.5
    assert res.in_range_pct["temperature"] > 50.0


def test_in_range_percentage():
    cfg = quiet(initial_temperature=21.0, outdoor_temperature=21.0, duration_min=19,
                automation_enabled=False, targets={"temp_min": 20.0, "temp_max": 22.0})
    assert run_simulation(cfg).in_range_pct["temperature"] == pytest.approx(100.0)


# ---------------------------------------------------------------- сценарии

def test_paired_comparison_uses_identical_noise():
    cfg = preset_config("Холодное помещение")
    comp = run_comparison(cfg)
    b, a = comp.baseline.frame, comp.automated.frame
    assert len(a) == len(b)
    # шум = показание − состояние; совпадает поэлементно в обоих прогонах
    np.testing.assert_allclose(a["temp"] - a["temp_true"], b["temp"] - b["temp_true"], atol=1e-9)
    np.testing.assert_allclose(a["hum"] - a["hum_true"], b["hum"] - b["hum_true"], atol=1e-9)
    assert comp.baseline.total_energy_kwh == 0.0
    assert not comp.baseline.config.automation_enabled
    assert comp.baseline.config.manual_states == DeviceStates()
    assert comp.automated.config.automation_enabled
    assert comp.automated.in_range_pct["temperature"] > comp.baseline.in_range_pct["temperature"]


def test_hysteresis_experiment_narrow_switches_more():
    cfg = preset_config("Холодное помещение")
    exp = run_hysteresis_experiment(cfg, "heater", 21.0, 0.4, 3.0)
    assert exp.narrow.device_metrics["heater"].switches > exp.wide.device_metrics["heater"].switches
    with pytest.raises(ConfigError):
        run_hysteresis_experiment(cfg, "heater", 21.0, 3.0, 0.4)
